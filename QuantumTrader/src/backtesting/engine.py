"""
Production-Grade Backtesting Engine for QuantumTrader

This module implements a sophisticated backtesting engine with realistic
market simulation, transaction costs, and comprehensive performance analytics.

Author: QuantumTrader Team
Date: 2024
"""

import warnings
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats
from tqdm import tqdm

from ..engine.base_strategy import (
    BaseStrategy, Signal, SignalType, Position, PositionSide, OrderType
)
from ..utils.logging import get_logger
from ..data.models import MarketDataSchema, TradeData

logger = get_logger()


class BacktestMode(Enum):
    """Backtesting modes."""
    
    VECTORIZED = "vectorized"  # Fast, approximate
    EVENT_DRIVEN = "event_driven"  # Accurate, slower
    MONTE_CARLO = "monte_carlo"  # Multiple simulations


class FillModel(Enum):
    """Order fill models."""
    
    CLOSE = "close"  # Fill at close price
    NEXT_OPEN = "next_open"  # Fill at next bar open
    MIDPOINT = "midpoint"  # Fill at bid-ask midpoint
    CONSERVATIVE = "conservative"  # Worst price


@dataclass
class TransactionCostModel:
    """Transaction cost model."""
    
    commission_type: str = "percentage"  # "percentage", "fixed", "per_share"
    commission_rate: float = 0.001  # 0.1%
    slippage_model: str = "linear"  # "linear", "square_root", "logarithmic"
    slippage_factor: float = 0.0005  # 0.05%
    min_commission: float = 1.0
    max_commission: float = 100.0
    
    def calculate_commission(self, value: float, shares: Optional[float] = None) -> float:
        """Calculate commission for a trade."""
        if self.commission_type == "percentage":
            commission = value * self.commission_rate
        elif self.commission_type == "fixed":
            commission = self.commission_rate
        elif self.commission_type == "per_share" and shares:
            commission = shares * self.commission_rate
        else:
            commission = 0.0
        
        return np.clip(commission, self.min_commission, self.max_commission)
    
    def calculate_slippage(self, price: float, volume: float, order_size: float) -> float:
        """Calculate slippage based on order size and market volume."""
        if volume == 0:
            return 0
        
        impact = order_size / volume
        
        if self.slippage_model == "linear":
            slippage = self.slippage_factor * impact
        elif self.slippage_model == "square_root":
            slippage = self.slippage_factor * np.sqrt(impact)
        elif self.slippage_model == "logarithmic":
            slippage = self.slippage_factor * np.log1p(impact)
        else:
            slippage = 0
        
        return price * slippage


@dataclass
class BacktestConfig:
    """Backtesting configuration."""
    
    initial_capital: float = 100000.0
    mode: BacktestMode = BacktestMode.EVENT_DRIVEN
    fill_model: FillModel = FillModel.CLOSE
    transaction_costs: TransactionCostModel = field(default_factory=TransactionCostModel)
    enable_shorting: bool = True
    margin_requirement: float = 0.25  # 25% margin for shorts
    use_adjusted_close: bool = True
    random_seed: Optional[int] = None
    warm_up_period: int = 0  # Bars to warm up indicators
    max_open_positions: int = 10
    position_sizing: str = "equal_weight"  # "equal_weight", "kelly", "risk_parity"
    rebalance_frequency: Optional[str] = None  # "daily", "weekly", "monthly"
    benchmark: Optional[str] = "SPY"  # Benchmark symbol
    risk_free_rate: float = 0.02  # Annual risk-free rate


@dataclass
class BacktestResult:
    """Backtesting results container."""
    
    # Performance metrics
    total_return: float = 0.0
    annual_return: float = 0.0
    volatility: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_duration: int = 0
    
    # Trade statistics
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    profit_factor: float = 0.0
    expectancy: float = 0.0
    
    # Risk metrics
    var_95: float = 0.0  # Value at Risk (95%)
    cvar_95: float = 0.0  # Conditional VaR
    downside_deviation: float = 0.0
    beta: float = 0.0
    alpha: float = 0.0
    
    # Time series
    equity_curve: pd.Series = field(default_factory=pd.Series)
    returns: pd.Series = field(default_factory=pd.Series)
    positions: pd.DataFrame = field(default_factory=pd.DataFrame)
    trades: pd.DataFrame = field(default_factory=pd.DataFrame)
    
    # Additional analytics
    monthly_returns: pd.DataFrame = field(default_factory=pd.DataFrame)
    rolling_sharpe: pd.Series = field(default_factory=pd.Series)
    rolling_volatility: pd.Series = field(default_factory=pd.Series)
    underwater_curve: pd.Series = field(default_factory=pd.Series)


class BacktestingEngine:
    """Advanced backtesting engine with realistic market simulation."""
    
    def __init__(self, config: BacktestConfig):
        """
        Initialize backtesting engine.
        
        Parameters
        ----------
        config : BacktestConfig
            Backtesting configuration
        """
        self.config = config
        self.strategy: Optional[BaseStrategy] = None
        self.market_data: Dict[str, pd.DataFrame] = {}
        self.benchmark_data: Optional[pd.DataFrame] = None
        
        # Portfolio state
        self.cash = config.initial_capital
        self.positions: Dict[str, Position] = {}
        self.trades: List[TradeData] = []
        self.equity_curve: List[float] = []
        self.timestamps: List[datetime] = []
        
        # Performance tracking
        self.daily_returns: List[float] = []
        self.high_water_mark = config.initial_capital
        self.max_drawdown = 0.0
        self.current_drawdown = 0.0
        
        # Set random seed for reproducibility
        if config.random_seed:
            np.random.seed(config.random_seed)
        
        logger.info(f"Initialized backtesting engine with {config.mode.value} mode")
    
    def add_data(self, symbol: str, data: pd.DataFrame) -> None:
        """
        Add market data for a symbol.
        
        Parameters
        ----------
        symbol : str
            Symbol identifier
        data : pd.DataFrame
            OHLCV data with datetime index
        """
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in data.columns for col in required_columns):
            raise ValueError(f"Data must contain columns: {required_columns}")
        
        # Ensure datetime index
        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError("Data must have DatetimeIndex")
        
        # Sort by date
        data = data.sort_index()
        
        # Handle adjusted close
        if self.config.use_adjusted_close and 'adj_close' in data.columns:
            data['close'] = data['adj_close']
        
        self.market_data[symbol] = data
        logger.info(f"Added {len(data)} bars of data for {symbol}")
    
    def run(
        self,
        strategy: BaseStrategy,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        progress_bar: bool = True
    ) -> BacktestResult:
        """
        Run backtest with given strategy.
        
        Parameters
        ----------
        strategy : BaseStrategy
            Trading strategy to backtest
        start_date : Optional[datetime]
            Start date for backtest
        end_date : Optional[datetime]
            End date for backtest
        progress_bar : bool
            Show progress bar
            
        Returns
        -------
        BacktestResult
            Comprehensive backtesting results
        """
        self.strategy = strategy
        self.strategy.initialize()
        
        # Validate data
        if not self.market_data:
            raise ValueError("No market data added")
        
        # Determine backtest period
        all_dates = pd.concat([df.index.to_series() for df in self.market_data.values()])
        if start_date is None:
            start_date = all_dates.min()
        if end_date is None:
            end_date = all_dates.max()
        
        # Filter data to backtest period
        for symbol in self.market_data:
            mask = (self.market_data[symbol].index >= start_date) & \
                   (self.market_data[symbol].index <= end_date)
            self.market_data[symbol] = self.market_data[symbol][mask]
        
        logger.info(f"Running backtest from {start_date} to {end_date}")
        
        if self.config.mode == BacktestMode.EVENT_DRIVEN:
            return self._run_event_driven(progress_bar)
        elif self.config.mode == BacktestMode.VECTORIZED:
            return self._run_vectorized()
        elif self.config.mode == BacktestMode.MONTE_CARLO:
            return self._run_monte_carlo(num_simulations=100)
        else:
            raise ValueError(f"Unknown backtest mode: {self.config.mode}")
    
    def _run_event_driven(self, progress_bar: bool = True) -> BacktestResult:
        """Run event-driven backtest."""
        # Get all unique timestamps
        all_timestamps = sorted(set().union(*[set(df.index) for df in self.market_data.values()]))
        
        # Skip warm-up period
        if self.config.warm_up_period > 0:
            all_timestamps = all_timestamps[self.config.warm_up_period:]
        
        # Progress bar
        iterator = tqdm(all_timestamps, desc="Backtesting") if progress_bar else all_timestamps
        
        for timestamp in iterator:
            # Get current market data
            current_data = {}
            for symbol, df in self.market_data.items():
                if timestamp in df.index:
                    current_data[symbol] = df.loc[:timestamp]
            
            if not current_data:
                continue
            
            # Update positions with current prices
            self._update_positions(timestamp, current_data)
            
            # Generate signals
            signals = self.strategy.on_data(timestamp, current_data)
            
            # Process signals
            for signal in signals:
                self._process_signal(signal, timestamp, current_data)
            
            # Check stop conditions
            self._check_stop_conditions(timestamp, current_data)
            
            # Record equity
            equity = self._calculate_equity(current_data)
            self.equity_curve.append(equity)
            self.timestamps.append(timestamp)
            
            # Calculate daily returns
            if len(self.equity_curve) > 1:
                daily_return = (self.equity_curve[-1] - self.equity_curve[-2]) / self.equity_curve[-2]
                self.daily_returns.append(daily_return)
            
            # Update drawdown
            if equity > self.high_water_mark:
                self.high_water_mark = equity
            self.current_drawdown = (equity - self.high_water_mark) / self.high_water_mark
            self.max_drawdown = min(self.max_drawdown, self.current_drawdown)
        
        # Calculate final results
        return self._calculate_results()
    
    def _process_signal(
        self,
        signal: Signal,
        timestamp: datetime,
        current_data: Dict[str, pd.DataFrame]
    ) -> None:
        """Process trading signal."""
        if signal.symbol not in current_data:
            logger.warning(f"No data for symbol {signal.symbol}")
            return
        
        current_bar = current_data[signal.symbol].iloc[-1]
        
        # Determine execution price
        if self.config.fill_model == FillModel.CLOSE:
            exec_price = current_bar['close']
        elif self.config.fill_model == FillModel.NEXT_OPEN:
            # Would need next bar's open - simplified here
            exec_price = current_bar['close'] * 1.0001
        elif self.config.fill_model == FillModel.MIDPOINT:
            if 'bid' in current_bar and 'ask' in current_bar:
                exec_price = (current_bar['bid'] + current_bar['ask']) / 2
            else:
                exec_price = current_bar['close']
        else:  # CONSERVATIVE
            if signal.signal_type in [SignalType.BUY, SignalType.CLOSE_SELL]:
                exec_price = current_bar['high']  # Worst price for buys
            else:
                exec_price = current_bar['low']  # Worst price for sells
        
        # Calculate position size
        if signal.quantity is None:
            signal.quantity = self.strategy.calculate_position_size(
                signal,
                self.cash,
                exec_price
            )
        
        # Apply slippage
        slippage = self.config.transaction_costs.calculate_slippage(
            exec_price,
            current_bar['volume'],
            signal.quantity
        )
        
        if signal.signal_type in [SignalType.BUY, SignalType.CLOSE_SELL]:
            exec_price += slippage
        else:
            exec_price -= slippage
        
        # Execute trade
        if signal.signal_type == SignalType.BUY:
            self._open_long_position(signal, exec_price, timestamp)
        elif signal.signal_type == SignalType.SELL and self.config.enable_shorting:
            self._open_short_position(signal, exec_price, timestamp)
        elif signal.signal_type in [SignalType.CLOSE_BUY, SignalType.CLOSE_SELL]:
            self._close_position(signal, exec_price, timestamp)
    
    def _open_long_position(
        self,
        signal: Signal,
        price: float,
        timestamp: datetime
    ) -> None:
        """Open long position."""
        if signal.symbol in self.positions:
            logger.warning(f"Position already exists for {signal.symbol}")
            return
        
        # Check if we have enough cash
        value = signal.quantity * price
        commission = self.config.transaction_costs.calculate_commission(value, signal.quantity)
        total_cost = value + commission
        
        if total_cost > self.cash:
            logger.warning(f"Insufficient cash for {signal.symbol}: need {total_cost}, have {self.cash}")
            return
        
        # Check max positions
        if len(self.positions) >= self.config.max_open_positions:
            logger.warning(f"Maximum positions ({self.config.max_open_positions}) reached")
            return
        
        # Create position
        position = Position(
            symbol=signal.symbol,
            side=PositionSide.LONG,
            quantity=signal.quantity,
            entry_price=price,
            entry_time=timestamp,
            current_price=price,
            commission=commission,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit
        )
        
        self.positions[signal.symbol] = position
        self.cash -= total_cost
        
        # Record trade
        trade = TradeData(
            trade_id=f"{timestamp}_{signal.symbol}_BUY",
            symbol=signal.symbol,
            timestamp=timestamp,
            price=price,
            size=signal.quantity,
            side="buy",
            fee=commission
        )
        self.trades.append(trade)
        
        logger.info(f"Opened long position: {signal.quantity} {signal.symbol} @ {price}")
    
    def _open_short_position(
        self,
        signal: Signal,
        price: float,
        timestamp: datetime
    ) -> None:
        """Open short position."""
        if signal.symbol in self.positions:
            logger.warning(f"Position already exists for {signal.symbol}")
            return
        
        # Calculate margin requirement
        value = signal.quantity * price
        margin_required = value * self.config.margin_requirement
        commission = self.config.transaction_costs.calculate_commission(value, signal.quantity)
        
        if margin_required + commission > self.cash:
            logger.warning(f"Insufficient margin for short {signal.symbol}")
            return
        
        # Create position
        position = Position(
            symbol=signal.symbol,
            side=PositionSide.SHORT,
            quantity=signal.quantity,
            entry_price=price,
            entry_time=timestamp,
            current_price=price,
            commission=commission,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit
        )
        
        self.positions[signal.symbol] = position
        self.cash -= (margin_required + commission)
        self.cash += value  # Credit from short sale
        
        # Record trade
        trade = TradeData(
            trade_id=f"{timestamp}_{signal.symbol}_SELL",
            symbol=signal.symbol,
            timestamp=timestamp,
            price=price,
            size=signal.quantity,
            side="sell",
            fee=commission
        )
        self.trades.append(trade)
        
        logger.info(f"Opened short position: {signal.quantity} {signal.symbol} @ {price}")
    
    def _close_position(
        self,
        signal: Signal,
        price: float,
        timestamp: datetime
    ) -> None:
        """Close existing position."""
        if signal.symbol not in self.positions:
            logger.warning(f"No position to close for {signal.symbol}")
            return
        
        position = self.positions[signal.symbol]
        value = position.quantity * price
        commission = self.config.transaction_costs.calculate_commission(value, position.quantity)
        
        # Calculate P&L
        if position.side == PositionSide.LONG:
            pnl = (price - position.entry_price) * position.quantity - commission - position.commission
            self.cash += value - commission
        else:  # SHORT
            pnl = (position.entry_price - price) * position.quantity - commission - position.commission
            # Return margin and settle P&L
            margin = position.entry_price * position.quantity * self.config.margin_requirement
            self.cash += margin + pnl
        
        position.exit_price = price
        position.exit_time = timestamp
        position.realized_pnl = pnl
        
        # Record trade
        trade = TradeData(
            trade_id=f"{timestamp}_{signal.symbol}_CLOSE",
            symbol=signal.symbol,
            timestamp=timestamp,
            price=price,
            size=position.quantity,
            side="sell" if position.side == PositionSide.LONG else "buy",
            fee=commission
        )
        self.trades.append(trade)
        
        # Remove position
        del self.positions[signal.symbol]
        
        logger.info(f"Closed position: {signal.symbol} P&L: ${pnl:.2f}")
    
    def _update_positions(
        self,
        timestamp: datetime,
        current_data: Dict[str, pd.DataFrame]
    ) -> None:
        """Update position values with current prices."""
        for symbol, position in self.positions.items():
            if symbol in current_data:
                current_price = current_data[symbol].iloc[-1]['close']
                position.current_price = current_price
                
                # Calculate unrealized P&L
                if position.side == PositionSide.LONG:
                    position.unrealized_pnl = (current_price - position.entry_price) * position.quantity
                else:  # SHORT
                    position.unrealized_pnl = (position.entry_price - current_price) * position.quantity
    
    def _check_stop_conditions(
        self,
        timestamp: datetime,
        current_data: Dict[str, pd.DataFrame]
    ) -> None:
        """Check stop loss and take profit conditions."""
        positions_to_close = []
        
        for symbol, position in self.positions.items():
            if symbol not in current_data:
                continue
            
            signal = self.strategy.check_stop_conditions(position)
            if signal:
                positions_to_close.append(signal)
        
        # Process close signals
        for signal in positions_to_close:
            self._process_signal(signal, timestamp, current_data)
    
    def _calculate_equity(self, current_data: Dict[str, pd.DataFrame]) -> float:
        """Calculate current portfolio equity."""
        equity = self.cash
        
        for symbol, position in self.positions.items():
            if symbol in current_data:
                current_price = current_data[symbol].iloc[-1]['close']
                
                if position.side == PositionSide.LONG:
                    equity += position.quantity * current_price
                else:  # SHORT
                    # Add back the margin and unrealized P&L
                    margin = position.entry_price * position.quantity * self.config.margin_requirement
                    equity += margin + position.unrealized_pnl
        
        return equity
    
    def _calculate_results(self) -> BacktestResult:
        """Calculate comprehensive backtesting results."""
        result = BacktestResult()
        
        # Convert to pandas Series
        result.equity_curve = pd.Series(self.equity_curve, index=self.timestamps)
        result.returns = pd.Series(self.daily_returns)
        
        # Basic metrics
        result.total_return = (self.equity_curve[-1] - self.config.initial_capital) / self.config.initial_capital
        
        # Annualized return
        days = (self.timestamps[-1] - self.timestamps[0]).days
        years = days / 365.25
        result.annual_return = (1 + result.total_return) ** (1/years) - 1 if years > 0 else 0
        
        # Volatility
        result.volatility = result.returns.std() * np.sqrt(252)
        
        # Sharpe ratio
        excess_returns = result.returns - self.config.risk_free_rate / 252
        result.sharpe_ratio = np.sqrt(252) * excess_returns.mean() / excess_returns.std() if excess_returns.std() > 0 else 0
        
        # Sortino ratio
        downside_returns = result.returns[result.returns < 0]
        result.downside_deviation = downside_returns.std() * np.sqrt(252)
        result.sortino_ratio = np.sqrt(252) * excess_returns.mean() / result.downside_deviation if result.downside_deviation > 0 else 0
        
        # Max drawdown
        result.max_drawdown = self.max_drawdown
        
        # Calmar ratio
        result.calmar_ratio = result.annual_return / abs(result.max_drawdown) if result.max_drawdown != 0 else 0
        
        # Trade statistics
        result.total_trades = len(self.trades)
        
        # Calculate trade P&Ls
        trade_pnls = []
        for i in range(0, len(self.trades), 2):  # Pairs of open/close
            if i + 1 < len(self.trades):
                open_trade = self.trades[i]
                close_trade = self.trades[i + 1]
                if open_trade.side == "buy":
                    pnl = (close_trade.price - open_trade.price) * open_trade.size - open_trade.fee - close_trade.fee
                else:
                    pnl = (open_trade.price - close_trade.price) * open_trade.size - open_trade.fee - close_trade.fee
                trade_pnls.append(pnl)
        
        if trade_pnls:
            trade_pnls = np.array(trade_pnls)
            result.winning_trades = np.sum(trade_pnls > 0)
            result.losing_trades = np.sum(trade_pnls <= 0)
            result.win_rate = result.winning_trades / len(trade_pnls) if trade_pnls.size > 0 else 0
            
            winning_pnls = trade_pnls[trade_pnls > 0]
            losing_pnls = trade_pnls[trade_pnls <= 0]
            
            result.avg_win = winning_pnls.mean() if winning_pnls.size > 0 else 0
            result.avg_loss = losing_pnls.mean() if losing_pnls.size > 0 else 0
            
            total_wins = winning_pnls.sum() if winning_pnls.size > 0 else 0
            total_losses = abs(losing_pnls.sum()) if losing_pnls.size > 0 else 0
            result.profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')
            
            result.expectancy = trade_pnls.mean()
        
        # Risk metrics
        if result.returns.size > 0:
            result.var_95 = np.percentile(result.returns, 5)
            result.cvar_95 = result.returns[result.returns <= result.var_95].mean()
        
        # Monthly returns table
        if len(result.equity_curve) > 0:
            monthly_returns = result.equity_curve.resample('M').last().pct_change()
            result.monthly_returns = monthly_returns.to_frame('returns')
        
        # Rolling metrics
        if len(result.returns) > 20:
            result.rolling_volatility = result.returns.rolling(20).std() * np.sqrt(252)
            rolling_mean = result.returns.rolling(20).mean()
            rolling_std = result.returns.rolling(20).std()
            result.rolling_sharpe = np.sqrt(252) * rolling_mean / rolling_std
        
        # Underwater curve
        cumulative = (1 + result.returns).cumprod()
        running_max = cumulative.expanding().max()
        result.underwater_curve = (cumulative - running_max) / running_max
        
        # Convert trades to DataFrame
        if self.trades:
            trades_data = []
            for trade in self.trades:
                trades_data.append({
                    'timestamp': trade.timestamp,
                    'symbol': trade.symbol,
                    'side': trade.side,
                    'price': trade.price,
                    'size': trade.size,
                    'fee': trade.fee
                })
            result.trades = pd.DataFrame(trades_data)
        
        logger.info(f"Backtest complete: Return={result.total_return:.2%}, Sharpe={result.sharpe_ratio:.2f}")
        
        return result
    
    def _run_vectorized(self) -> BacktestResult:
        """Run vectorized backtest (faster but less accurate)."""
        # Simplified vectorized implementation
        logger.warning("Vectorized mode is simplified - use event-driven for accurate results")
        
        # This would be a full vectorized implementation
        # For now, redirect to event-driven
        return self._run_event_driven(progress_bar=False)
    
    def _run_monte_carlo(self, num_simulations: int = 100) -> BacktestResult:
        """Run Monte Carlo simulation."""
        logger.info(f"Running {num_simulations} Monte Carlo simulations")
        
        results = []
        for i in range(num_simulations):
            # Reset state
            self.cash = self.config.initial_capital
            self.positions.clear()
            self.trades.clear()
            self.equity_curve.clear()
            self.timestamps.clear()
            self.daily_returns.clear()
            
            # Add random noise to prices for each simulation
            modified_data = {}
            for symbol, df in self.market_data.items():
                noise = np.random.normal(1, 0.001, size=len(df))
                modified_df = df.copy()
                for col in ['open', 'high', 'low', 'close']:
                    modified_df[col] = modified_df[col] * noise
                modified_data[symbol] = modified_df
            
            # Store original data
            original_data = self.market_data
            self.market_data = modified_data
            
            # Run simulation
            result = self._run_event_driven(progress_bar=False)
            results.append(result)
            
            # Restore original data
            self.market_data = original_data
        
        # Aggregate results
        final_result = BacktestResult()
        
        # Average metrics
        metrics = ['total_return', 'annual_return', 'volatility', 'sharpe_ratio',
                  'sortino_ratio', 'max_drawdown', 'win_rate', 'profit_factor']
        
        for metric in metrics:
            values = [getattr(r, metric) for r in results]
            setattr(final_result, metric, np.mean(values))
        
        # Add confidence intervals
        returns = [r.total_return for r in results]
        final_result.metadata = {
            'confidence_interval_95': (np.percentile(returns, 2.5), np.percentile(returns, 97.5)),
            'std_returns': np.std(returns),
            'num_simulations': num_simulations
        }
        
        logger.info(f"Monte Carlo complete: Mean Return={final_result.total_return:.2%}")
        
        return final_result
