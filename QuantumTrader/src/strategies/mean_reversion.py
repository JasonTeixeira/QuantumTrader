"""
Mean Reversion Trading Strategy

Implements a sophisticated mean reversion strategy with Bollinger Bands,
RSI, and statistical arbitrage techniques.

Author: QuantumTrader Team
Date: 2024
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands

from ..engine.base_strategy import (
    BaseStrategy, Signal, SignalType, StrategyConfig, Position
)
from ..utils.logging import get_logger

logger = get_logger()


@dataclass
class MeanReversionConfig(StrategyConfig):
    """Mean reversion strategy configuration."""
    
    # Bollinger Bands parameters
    bb_period: int = 20
    bb_std: float = 2.0
    
    # RSI parameters
    rsi_period: int = 14
    rsi_oversold: float = 30.0
    rsi_overbought: float = 70.0
    
    # Z-score parameters
    zscore_period: int = 20
    zscore_entry: float = 2.0
    zscore_exit: float = 0.5
    
    # Cointegration parameters
    use_cointegration: bool = False
    cointegration_period: int = 60
    
    # Risk parameters
    max_holding_period: int = 10  # Maximum days to hold position
    use_stop_loss: bool = True
    stop_loss_pct: float = 0.05  # 5% stop loss
    use_take_profit: bool = True
    take_profit_pct: float = 0.03  # 3% take profit
    
    # Signal confirmation
    require_volume_confirmation: bool = True
    volume_threshold: float = 1.5  # 1.5x average volume
    
    # Position sizing
    position_sizing_method: str = "kelly"  # "fixed", "kelly", "volatility"
    kelly_fraction: float = 0.25  # Conservative Kelly


class MeanReversionStrategy(BaseStrategy):
    """
    Advanced mean reversion strategy using multiple indicators.
    
    This strategy identifies overbought/oversold conditions using:
    - Bollinger Bands for price extremes
    - RSI for momentum confirmation
    - Z-score for statistical significance
    - Volume confirmation for liquidity
    """
    
    def __init__(self, config: MeanReversionConfig):
        """
        Initialize mean reversion strategy.
        
        Parameters
        ----------
        config : MeanReversionConfig
            Strategy configuration
        """
        super().__init__(config)
        self.config = config
        
        # Indicator storage
        self.bollinger_bands = {}
        self.rsi_indicators = {}
        self.z_scores = {}
        self.rolling_stats = {}
        
        # Performance tracking
        self.entry_signals = []
        self.exit_signals = []
        self.win_rate_estimate = 0.6  # Initial estimate
        self.avg_win_loss_ratio = 1.5  # Initial estimate
    
    def _setup_indicators(self) -> None:
        """Set up technical indicators."""
        logger.info("Setting up mean reversion indicators")
        
        for symbol in self.config.symbols:
            self.bollinger_bands[symbol] = None
            self.rsi_indicators[symbol] = None
            self.z_scores[symbol] = None
            self.rolling_stats[symbol] = {
                'mean': None,
                'std': None,
                'volume_mean': None
            }
    
    def _setup_risk_management(self) -> None:
        """Set up risk management parameters."""
        logger.info("Setting up risk management for mean reversion")
        
        self.max_portfolio_risk = 0.02  # 2% max risk per trade
        self.correlation_threshold = 0.7  # Max correlation between positions
    
    def calculate_signals(
        self,
        market_data: pd.DataFrame,
        **kwargs
    ) -> List[Signal]:
        """
        Calculate trading signals based on mean reversion indicators.
        
        Parameters
        ----------
        market_data : pd.DataFrame
            Historical market data
        **kwargs
            Additional parameters
            
        Returns
        -------
        List[Signal]
            List of trading signals
        """
        if len(market_data) < max(self.config.bb_period, self.config.rsi_period, self.config.zscore_period):
            return []
        
        signals = []
        symbol = kwargs.get('symbol', 'UNKNOWN')
        
        # Calculate indicators
        bb_upper, bb_middle, bb_lower = self._calculate_bollinger_bands(market_data)
        rsi = self._calculate_rsi(market_data)
        z_score = self._calculate_z_score(market_data)
        
        # Get current values
        current_price = market_data['close'].iloc[-1]
        current_volume = market_data['volume'].iloc[-1]
        avg_volume = market_data['volume'].rolling(20).mean().iloc[-1]
        
        # Volume confirmation
        volume_confirmed = True
        if self.config.require_volume_confirmation:
            volume_confirmed = current_volume > avg_volume * self.config.volume_threshold
        
        # Generate signals
        signal_strength = 0.0
        signal_type = SignalType.HOLD
        
        # Check for oversold condition (potential buy)
        if (current_price < bb_lower.iloc[-1] and 
            rsi.iloc[-1] < self.config.rsi_oversold and
            z_score.iloc[-1] < -self.config.zscore_entry and
            volume_confirmed):
            
            signal_type = SignalType.BUY
            signal_strength = self._calculate_signal_strength(
                current_price, bb_lower.iloc[-1], rsi.iloc[-1], z_score.iloc[-1]
            )
            
            # Calculate stop loss and take profit
            stop_loss = current_price * (1 - self.config.stop_loss_pct) if self.config.use_stop_loss else None
            take_profit = current_price * (1 + self.config.take_profit_pct) if self.config.use_take_profit else None
            
            signal = Signal(
                timestamp=market_data.index[-1],
                symbol=symbol,
                signal_type=signal_type,
                strength=signal_strength,
                stop_loss=stop_loss,
                take_profit=take_profit,
                confidence=self._calculate_confidence(market_data),
                metadata={
                    'rsi': rsi.iloc[-1],
                    'z_score': z_score.iloc[-1],
                    'bb_position': (current_price - bb_lower.iloc[-1]) / (bb_upper.iloc[-1] - bb_lower.iloc[-1]),
                    'volume_ratio': current_volume / avg_volume
                }
            )
            signals.append(signal)
        
        # Check for overbought condition (potential sell/short)
        elif (current_price > bb_upper.iloc[-1] and 
              rsi.iloc[-1] > self.config.rsi_overbought and
              z_score.iloc[-1] > self.config.zscore_entry and
              volume_confirmed and
              self.config.enable_shorts):
            
            signal_type = SignalType.SELL
            signal_strength = self._calculate_signal_strength(
                current_price, bb_upper.iloc[-1], rsi.iloc[-1], z_score.iloc[-1]
            )
            
            # Calculate stop loss and take profit for short
            stop_loss = current_price * (1 + self.config.stop_loss_pct) if self.config.use_stop_loss else None
            take_profit = current_price * (1 - self.config.take_profit_pct) if self.config.use_take_profit else None
            
            signal = Signal(
                timestamp=market_data.index[-1],
                symbol=symbol,
                signal_type=signal_type,
                strength=signal_strength,
                stop_loss=stop_loss,
                take_profit=take_profit,
                confidence=self._calculate_confidence(market_data),
                metadata={
                    'rsi': rsi.iloc[-1],
                    'z_score': z_score.iloc[-1],
                    'bb_position': (current_price - bb_lower.iloc[-1]) / (bb_upper.iloc[-1] - bb_lower.iloc[-1]),
                    'volume_ratio': current_volume / avg_volume
                }
            )
            signals.append(signal)
        
        # Check for exit conditions
        if symbol in self.positions:
            position = self.positions[symbol]
            
            # Exit on mean reversion
            if abs(z_score.iloc[-1]) < self.config.zscore_exit:
                if position.side == position.side.LONG:
                    signal_type = SignalType.CLOSE_BUY
                else:
                    signal_type = SignalType.CLOSE_SELL
                
                signal = Signal(
                    timestamp=market_data.index[-1],
                    symbol=symbol,
                    signal_type=signal_type,
                    strength=1.0,
                    metadata={'reason': 'mean_reversion', 'z_score': z_score.iloc[-1]}
                )
                signals.append(signal)
            
            # Exit on max holding period
            elif self._check_holding_period(position):
                if position.side == position.side.LONG:
                    signal_type = SignalType.CLOSE_BUY
                else:
                    signal_type = SignalType.CLOSE_SELL
                
                signal = Signal(
                    timestamp=market_data.index[-1],
                    symbol=symbol,
                    signal_type=signal_type,
                    strength=0.5,
                    metadata={'reason': 'max_holding_period'}
                )
                signals.append(signal)
        
        return signals
    
    def on_data(
        self,
        timestamp: datetime,
        data: Dict[str, pd.DataFrame]
    ) -> List[Signal]:
        """
        Process new market data and generate signals.
        
        Parameters
        ----------
        timestamp : datetime
            Current timestamp
        data : Dict[str, pd.DataFrame]
            Market data for each symbol
            
        Returns
        -------
        List[Signal]
            List of trading signals
        """
        all_signals = []
        
        for symbol, df in data.items():
            if symbol not in self.config.symbols:
                continue
            
            # Update indicators
            self._update_indicators(symbol, df)
            
            # Generate signals
            signals = self.calculate_signals(df, symbol=symbol)
            
            # Filter signals based on portfolio risk
            filtered_signals = self._filter_signals_by_risk(signals)
            
            all_signals.extend(filtered_signals)
        
        # Check for pairs trading opportunities if enabled
        if self.config.use_cointegration and len(data) > 1:
            pairs_signals = self._check_pairs_trading(timestamp, data)
            all_signals.extend(pairs_signals)
        
        return all_signals
    
    def calculate_position_size(
        self,
        signal: Signal,
        account_balance: float,
        current_price: float
    ) -> float:
        """
        Calculate position size using Kelly Criterion or other methods.
        
        Parameters
        ----------
        signal : Signal
            Trading signal
        account_balance : float
            Current account balance
        current_price : float
            Current asset price
            
        Returns
        -------
        float
            Position size in units
        """
        if self.config.position_sizing_method == "fixed":
            # Fixed percentage of capital
            position_value = account_balance * self.config.position_size
            return position_value / current_price
        
        elif self.config.position_sizing_method == "kelly":
            # Kelly Criterion
            kelly_percentage = self._calculate_kelly_percentage()
            position_value = account_balance * kelly_percentage * self.config.kelly_fraction
            return position_value / current_price
        
        elif self.config.position_sizing_method == "volatility":
            # Volatility-based sizing
            volatility = signal.metadata.get('volatility', 0.02)
            target_risk = self.config.risk_per_trade
            position_value = (account_balance * target_risk) / volatility
            return position_value / current_price
        
        else:
            # Default to fixed
            position_value = account_balance * self.config.position_size
            return position_value / current_price
    
    def _calculate_bollinger_bands(self, data: pd.DataFrame) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Bollinger Bands."""
        bb = BollingerBands(
            close=data['close'],
            window=self.config.bb_period,
            window_dev=self.config.bb_std
        )
        
        return (
            bb.bollinger_hband(),
            bb.bollinger_mavg(),
            bb.bollinger_lband()
        )
    
    def _calculate_rsi(self, data: pd.DataFrame) -> pd.Series:
        """Calculate RSI."""
        rsi = RSIIndicator(
            close=data['close'],
            window=self.config.rsi_period
        )
        return rsi.rsi()
    
    def _calculate_z_score(self, data: pd.DataFrame) -> pd.Series:
        """Calculate Z-score."""
        mean = data['close'].rolling(self.config.zscore_period).mean()
        std = data['close'].rolling(self.config.zscore_period).std()
        z_score = (data['close'] - mean) / std
        return z_score
    
    def _calculate_signal_strength(
        self,
        price: float,
        band_level: float,
        rsi: float,
        z_score: float
    ) -> float:
        """
        Calculate signal strength based on multiple factors.
        
        Parameters
        ----------
        price : float
            Current price
        band_level : float
            Bollinger band level
        rsi : float
            RSI value
        z_score : float
            Z-score value
            
        Returns
        -------
        float
            Signal strength between -1 and 1
        """
        # Distance from band
        band_strength = min(abs(price - band_level) / band_level, 1.0)
        
        # RSI extremity
        if rsi < 50:
            rsi_strength = (50 - rsi) / 50
        else:
            rsi_strength = (rsi - 50) / 50
        
        # Z-score extremity
        z_strength = min(abs(z_score) / 3, 1.0)
        
        # Weighted average
        strength = (band_strength * 0.3 + rsi_strength * 0.3 + z_strength * 0.4)
        
        return np.clip(strength, -1.0, 1.0)
    
    def _calculate_confidence(self, data: pd.DataFrame) -> float:
        """
        Calculate confidence level based on indicator agreement.
        
        Parameters
        ----------
        data : pd.DataFrame
            Market data
            
        Returns
        -------
        float
            Confidence level between 0 and 1
        """
        # Calculate various indicators
        bb_upper, bb_middle, bb_lower = self._calculate_bollinger_bands(data)
        rsi = self._calculate_rsi(data)
        z_score = self._calculate_z_score(data)
        
        current_price = data['close'].iloc[-1]
        
        # Check indicator agreement
        indicators_agree = 0
        total_indicators = 3
        
        # Bollinger Bands
        if current_price < bb_lower.iloc[-1] or current_price > bb_upper.iloc[-1]:
            indicators_agree += 1
        
        # RSI
        if rsi.iloc[-1] < self.config.rsi_oversold or rsi.iloc[-1] > self.config.rsi_overbought:
            indicators_agree += 1
        
        # Z-score
        if abs(z_score.iloc[-1]) > self.config.zscore_entry:
            indicators_agree += 1
        
        # Calculate confidence
        confidence = indicators_agree / total_indicators
        
        # Adjust for market regime
        volatility = data['close'].pct_change().std()
        if volatility > 0.03:  # High volatility reduces confidence
            confidence *= 0.8
        
        return min(confidence, 1.0)
    
    def _calculate_kelly_percentage(self) -> float:
        """
        Calculate Kelly Criterion percentage.
        
        Returns
        -------
        float
            Kelly percentage
        """
        if self.avg_win_loss_ratio == 0:
            return 0.0
        
        # Kelly formula: f = (p * b - q) / b
        # where p = win probability, q = loss probability, b = win/loss ratio
        p = self.win_rate_estimate
        q = 1 - p
        b = self.avg_win_loss_ratio
        
        kelly = (p * b - q) / b
        
        # Cap at reasonable levels
        return max(0, min(kelly, 0.25))  # Cap at 25%
    
    def _update_indicators(self, symbol: str, data: pd.DataFrame) -> None:
        """Update indicators for a symbol."""
        if len(data) < self.config.bb_period:
            return
        
        # Update Bollinger Bands
        bb = BollingerBands(
            close=data['close'],
            window=self.config.bb_period,
            window_dev=self.config.bb_std
        )
        self.bollinger_bands[symbol] = bb
        
        # Update RSI
        rsi = RSIIndicator(
            close=data['close'],
            window=self.config.rsi_period
        )
        self.rsi_indicators[symbol] = rsi
        
        # Update rolling statistics
        self.rolling_stats[symbol]['mean'] = data['close'].rolling(self.config.zscore_period).mean()
        self.rolling_stats[symbol]['std'] = data['close'].rolling(self.config.zscore_period).std()
        self.rolling_stats[symbol]['volume_mean'] = data['volume'].rolling(20).mean()
    
    def _filter_signals_by_risk(self, signals: List[Signal]) -> List[Signal]:
        """Filter signals based on portfolio risk constraints."""
        filtered = []
        
        for signal in signals:
            # Check portfolio exposure
            current_exposure = sum(p.value for p in self.positions.values())
            
            if signal.signal_type in [SignalType.BUY, SignalType.SELL]:
                # Check if adding this position would exceed risk limits
                potential_risk = self.max_portfolio_risk * len(self.positions)
                
                if potential_risk < 0.1:  # Max 10% portfolio risk
                    filtered.append(signal)
            else:
                # Always allow closing positions
                filtered.append(signal)
        
        return filtered
    
    def _check_holding_period(self, position: Position) -> bool:
        """Check if position has exceeded max holding period."""
        if not hasattr(position, 'entry_time'):
            return False
        
        holding_days = (datetime.now() - position.entry_time).days
        return holding_days >= self.config.max_holding_period
    
    def _check_pairs_trading(
        self,
        timestamp: datetime,
        data: Dict[str, pd.DataFrame]
    ) -> List[Signal]:
        """
        Check for pairs trading opportunities using cointegration.
        
        Parameters
        ----------
        timestamp : datetime
            Current timestamp
        data : Dict[str, pd.DataFrame]
            Market data for all symbols
            
        Returns
        -------
        List[Signal]
            Pairs trading signals
        """
        signals = []
        
        # Simple pairs trading implementation
        symbols = list(data.keys())
        
        for i in range(len(symbols)):
            for j in range(i + 1, len(symbols)):
                symbol1, symbol2 = symbols[i], symbols[j]
                
                if len(data[symbol1]) < self.config.cointegration_period:
                    continue
                
                # Calculate spread
                price1 = data[symbol1]['close'].iloc[-self.config.cointegration_period:]
                price2 = data[symbol2]['close'].iloc[-self.config.cointegration_period:]
                
                # Normalize prices
                norm_price1 = price1 / price1.iloc[0]
                norm_price2 = price2 / price2.iloc[0]
                
                # Calculate spread
                spread = norm_price1 - norm_price2
                spread_mean = spread.mean()
                spread_std = spread.std()
                
                if spread_std == 0:
                    continue
                
                # Calculate z-score of spread
                current_spread = spread.iloc[-1]
                z_score = (current_spread - spread_mean) / spread_std
                
                # Generate signals based on spread
                if z_score > 2:  # Spread is too wide, expect convergence
                    # Short symbol1, long symbol2
                    if self.config.enable_shorts:
                        signals.append(Signal(
                            timestamp=timestamp,
                            symbol=symbol1,
                            signal_type=SignalType.SELL,
                            strength=min(z_score / 3, 1.0),
                            metadata={'pair': symbol2, 'spread_zscore': z_score}
                        ))
                    
                    signals.append(Signal(
                        timestamp=timestamp,
                        symbol=symbol2,
                        signal_type=SignalType.BUY,
                        strength=min(z_score / 3, 1.0),
                        metadata={'pair': symbol1, 'spread_zscore': z_score}
                    ))
                
                elif z_score < -2:  # Spread is too narrow
                    # Long symbol1, short symbol2
                    signals.append(Signal(
                        timestamp=timestamp,
                        symbol=symbol1,
                        signal_type=SignalType.BUY,
                        strength=min(abs(z_score) / 3, 1.0),
                        metadata={'pair': symbol2, 'spread_zscore': z_score}
                    ))
                    
                    if self.config.enable_shorts:
                        signals.append(Signal(
                            timestamp=timestamp,
                            symbol=symbol2,
                            signal_type=SignalType.SELL,
                            strength=min(abs(z_score) / 3, 1.0),
                            metadata={'pair': symbol1, 'spread_zscore': z_score}
                        ))
        
        return signals
