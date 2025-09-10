"""
Risk Management System for QuantumTrader

Implements comprehensive risk management including position sizing,
portfolio optimization, risk metrics calculation, and exposure management.

Author: QuantumTrader Team
Date: 2024
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf

from ..engine.base_strategy import Position, PositionSide, Signal, SignalType
from ..utils.logging import get_logger

logger = get_logger()


class RiskMetric(Enum):
    """Risk metric types."""
    
    VAR = "value_at_risk"
    CVAR = "conditional_value_at_risk"
    SHARPE = "sharpe_ratio"
    SORTINO = "sortino_ratio"
    MAX_DRAWDOWN = "max_drawdown"
    VOLATILITY = "volatility"
    BETA = "beta"
    CORRELATION = "correlation"


@dataclass
class RiskLimits:
    """Risk limits configuration."""
    
    # Portfolio level limits
    max_portfolio_var: float = 0.05  # 5% VaR
    max_portfolio_leverage: float = 2.0
    max_drawdown: float = 0.20  # 20% max drawdown
    max_concentration: float = 0.25  # 25% max in single position
    
    # Position level limits
    max_position_size: float = 0.20  # 20% of portfolio
    min_position_size: float = 0.01  # 1% of portfolio
    max_position_risk: float = 0.02  # 2% risk per position
    
    # Sector/Asset class limits
    max_sector_exposure: float = 0.40  # 40% in one sector
    max_asset_class_exposure: float = 0.60  # 60% in one asset class
    max_correlation: float = 0.80  # Max correlation between positions
    
    # Trading limits
    max_daily_trades: int = 50
    max_daily_turnover: float = 2.0  # 200% daily turnover
    max_intraday_exposure: float = 1.5
    
    # Greeks limits (for options)
    max_delta: float = 100000
    max_gamma: float = 10000
    max_vega: float = 50000
    max_theta: float = -10000


@dataclass
class PortfolioRiskMetrics:
    """Portfolio risk metrics."""
    
    timestamp: datetime = field(default_factory=datetime.now)
    
    # Risk metrics
    portfolio_var_95: float = 0.0
    portfolio_var_99: float = 0.0
    portfolio_cvar_95: float = 0.0
    expected_shortfall: float = 0.0
    
    # Performance metrics
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    information_ratio: float = 0.0
    
    # Exposure metrics
    gross_exposure: float = 0.0
    net_exposure: float = 0.0
    leverage: float = 0.0
    concentration: float = 0.0
    
    # Greeks (if applicable)
    portfolio_delta: float = 0.0
    portfolio_gamma: float = 0.0
    portfolio_vega: float = 0.0
    portfolio_theta: float = 0.0
    
    # Correlation metrics
    avg_correlation: float = 0.0
    max_correlation: float = 0.0
    diversification_ratio: float = 0.0
    
    # Stress test results
    stress_scenarios: Dict[str, float] = field(default_factory=dict)


class RiskManager:
    """
    Comprehensive risk management system.
    
    Features:
    - Position sizing algorithms (Kelly, Risk Parity, Vol Target)
    - Portfolio optimization (Mean-Variance, Black-Litterman)
    - Risk metrics calculation (VaR, CVaR, Greeks)
    - Exposure management and limits monitoring
    - Stress testing and scenario analysis
    """
    
    def __init__(
        self,
        risk_limits: Optional[RiskLimits] = None,
        risk_free_rate: float = 0.02
    ):
        """
        Initialize risk manager.
        
        Parameters
        ----------
        risk_limits : Optional[RiskLimits]
            Risk limits configuration
        risk_free_rate : float
            Annual risk-free rate
        """
        self.risk_limits = risk_limits or RiskLimits()
        self.risk_free_rate = risk_free_rate
        
        # Portfolio state
        self.positions: Dict[str, Position] = {}
        self.historical_returns: pd.DataFrame = pd.DataFrame()
        self.correlation_matrix: Optional[pd.DataFrame] = None
        self.covariance_matrix: Optional[pd.DataFrame] = None
        
        # Risk metrics history
        self.risk_metrics_history: List[PortfolioRiskMetrics] = []
        self.daily_trades_count = 0
        self.daily_turnover = 0.0
        
        # Optimization parameters
        self.optimization_frequency = "weekly"
        self.rebalance_threshold = 0.05  # 5% deviation triggers rebalance
        
        logger.info("Initialized Risk Manager")
    
    def calculate_position_size(
        self,
        signal: Signal,
        portfolio_value: float,
        current_price: float,
        method: str = "risk_parity",
        **kwargs
    ) -> float:
        """
        Calculate optimal position size.
        
        Parameters
        ----------
        signal : Signal
            Trading signal
        portfolio_value : float
            Total portfolio value
        current_price : float
            Current asset price
        method : str
            Sizing method: 'kelly', 'risk_parity', 'equal_weight', 'volatility_target'
        **kwargs
            Additional parameters
            
        Returns
        -------
        float
            Position size in units
        """
        if method == "kelly":
            return self._kelly_position_size(signal, portfolio_value, current_price, **kwargs)
        elif method == "risk_parity":
            return self._risk_parity_position_size(signal, portfolio_value, current_price, **kwargs)
        elif method == "volatility_target":
            return self._volatility_target_position_size(signal, portfolio_value, current_price, **kwargs)
        elif method == "equal_weight":
            return self._equal_weight_position_size(portfolio_value, current_price, **kwargs)
        else:
            # Default to equal weight
            return self._equal_weight_position_size(portfolio_value, current_price, **kwargs)
    
    def _kelly_position_size(
        self,
        signal: Signal,
        portfolio_value: float,
        current_price: float,
        win_rate: float = 0.55,
        avg_win: float = 0.02,
        avg_loss: float = 0.01,
        kelly_fraction: float = 0.25
    ) -> float:
        """
        Calculate position size using Kelly Criterion.
        
        Kelly formula: f = (p*b - q) / b
        where:
        - f = fraction of capital to bet
        - p = probability of winning
        - q = probability of losing (1-p)
        - b = ratio of win to loss
        """
        if avg_loss == 0:
            return 0
        
        p = win_rate
        q = 1 - p
        b = avg_win / avg_loss
        
        # Calculate Kelly percentage
        kelly_pct = (p * b - q) / b
        
        # Apply Kelly fraction (conservative approach)
        kelly_pct = kelly_pct * kelly_fraction
        
        # Apply limits
        kelly_pct = np.clip(kelly_pct, 0, self.risk_limits.max_position_size)
        
        # Calculate position size
        position_value = portfolio_value * kelly_pct
        position_size = position_value / current_price
        
        logger.info(f"Kelly position size: {position_size:.2f} units ({kelly_pct:.2%} of portfolio)")
        
        return position_size
    
    def _risk_parity_position_size(
        self,
        signal: Signal,
        portfolio_value: float,
        current_price: float,
        target_risk: float = 0.01,
        lookback_period: int = 60
    ) -> float:
        """
        Calculate position size using Risk Parity approach.
        
        Each position contributes equally to portfolio risk.
        """
        symbol = signal.symbol
        
        # Get historical volatility
        if symbol in self.historical_returns.columns:
            returns = self.historical_returns[symbol].tail(lookback_period)
            volatility = returns.std() * np.sqrt(252)
        else:
            # Use default volatility if no history
            volatility = 0.20  # 20% annual volatility
        
        # Calculate position size for target risk contribution
        if volatility > 0:
            position_pct = target_risk / volatility
        else:
            position_pct = 0.05  # Default 5%
        
        # Apply limits
        position_pct = np.clip(position_pct, 
                              self.risk_limits.min_position_size,
                              self.risk_limits.max_position_size)
        
        # Calculate position size
        position_value = portfolio_value * position_pct
        position_size = position_value / current_price
        
        logger.info(f"Risk parity position size: {position_size:.2f} units (vol={volatility:.2%})")
        
        return position_size
    
    def _volatility_target_position_size(
        self,
        signal: Signal,
        portfolio_value: float,
        current_price: float,
        target_volatility: float = 0.15,
        lookback_period: int = 20
    ) -> float:
        """
        Calculate position size targeting specific volatility.
        """
        symbol = signal.symbol
        
        # Get historical volatility
        if symbol in self.historical_returns.columns:
            returns = self.historical_returns[symbol].tail(lookback_period)
            realized_vol = returns.std() * np.sqrt(252)
        else:
            realized_vol = 0.20  # Default
        
        # Calculate scaling factor
        if realized_vol > 0:
            vol_scalar = target_volatility / realized_vol
        else:
            vol_scalar = 1.0
        
        # Base position size
        base_position_pct = 0.10  # 10% base
        
        # Scaled position
        position_pct = base_position_pct * vol_scalar
        
        # Apply limits
        position_pct = np.clip(position_pct,
                              self.risk_limits.min_position_size,
                              self.risk_limits.max_position_size)
        
        # Calculate position size
        position_value = portfolio_value * position_pct
        position_size = position_value / current_price
        
        return position_size
    
    def _equal_weight_position_size(
        self,
        portfolio_value: float,
        current_price: float,
        num_positions: int = 10
    ) -> float:
        """
        Calculate equal weight position size.
        """
        position_pct = 1.0 / num_positions
        
        # Apply limits
        position_pct = np.clip(position_pct,
                              self.risk_limits.min_position_size,
                              self.risk_limits.max_position_size)
        
        position_value = portfolio_value * position_pct
        position_size = position_value / current_price
        
        return position_size
    
    def optimize_portfolio(
        self,
        expected_returns: pd.Series,
        covariance_matrix: pd.DataFrame,
        method: str = "mean_variance",
        constraints: Optional[Dict] = None
    ) -> pd.Series:
        """
        Optimize portfolio weights.
        
        Parameters
        ----------
        expected_returns : pd.Series
            Expected returns for each asset
        covariance_matrix : pd.DataFrame
            Covariance matrix
        method : str
            Optimization method: 'mean_variance', 'min_variance', 'max_sharpe', 'risk_parity'
        constraints : Optional[Dict]
            Additional constraints
            
        Returns
        -------
        pd.Series
            Optimal weights
        """
        n_assets = len(expected_returns)
        
        if method == "mean_variance":
            return self._mean_variance_optimization(expected_returns, covariance_matrix, constraints)
        elif method == "min_variance":
            return self._min_variance_optimization(covariance_matrix, constraints)
        elif method == "max_sharpe":
            return self._max_sharpe_optimization(expected_returns, covariance_matrix, constraints)
        elif method == "risk_parity":
            return self._risk_parity_optimization(covariance_matrix)
        else:
            # Default to equal weight
            weights = pd.Series(1.0 / n_assets, index=expected_returns.index)
            return weights
    
    def _mean_variance_optimization(
        self,
        expected_returns: pd.Series,
        covariance_matrix: pd.DataFrame,
        constraints: Optional[Dict] = None
    ) -> pd.Series:
        """
        Mean-variance optimization (Markowitz).
        """
        n_assets = len(expected_returns)
        
        # Objective function (negative Sharpe ratio)
        def objective(weights):
            portfolio_return = np.dot(weights, expected_returns)
            portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(covariance_matrix, weights)))
            sharpe = (portfolio_return - self.risk_free_rate) / portfolio_vol
            return -sharpe
        
        # Constraints
        cons = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}  # Weights sum to 1
        ]
        
        # Bounds
        bounds = tuple((0, self.risk_limits.max_concentration) for _ in range(n_assets))
        
        # Initial guess (equal weight)
        x0 = np.array([1.0 / n_assets] * n_assets)
        
        # Optimize
        result = minimize(
            objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=cons
        )
        
        if result.success:
            weights = pd.Series(result.x, index=expected_returns.index)
            logger.info(f"Optimization successful. Sharpe: {-result.fun:.3f}")
        else:
            logger.warning("Optimization failed. Using equal weights.")
            weights = pd.Series(1.0 / n_assets, index=expected_returns.index)
        
        return weights
    
    def _min_variance_optimization(
        self,
        covariance_matrix: pd.DataFrame,
        constraints: Optional[Dict] = None
    ) -> pd.Series:
        """
        Minimum variance portfolio optimization.
        """
        n_assets = len(covariance_matrix)
        
        # Objective function (portfolio variance)
        def objective(weights):
            return np.dot(weights.T, np.dot(covariance_matrix, weights))
        
        # Constraints
        cons = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
        ]
        
        # Bounds
        bounds = tuple((0, self.risk_limits.max_concentration) for _ in range(n_assets))
        
        # Initial guess
        x0 = np.array([1.0 / n_assets] * n_assets)
        
        # Optimize
        result = minimize(
            objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=cons
        )
        
        if result.success:
            weights = pd.Series(result.x, index=covariance_matrix.index)
        else:
            weights = pd.Series(1.0 / n_assets, index=covariance_matrix.index)
        
        return weights
    
    def _max_sharpe_optimization(
        self,
        expected_returns: pd.Series,
        covariance_matrix: pd.DataFrame,
        constraints: Optional[Dict] = None
    ) -> pd.Series:
        """
        Maximum Sharpe ratio optimization.
        """
        return self._mean_variance_optimization(expected_returns, covariance_matrix, constraints)
    
    def _risk_parity_optimization(
        self,
        covariance_matrix: pd.DataFrame
    ) -> pd.Series:
        """
        Risk parity portfolio optimization.
        
        Each asset contributes equally to portfolio risk.
        """
        n_assets = len(covariance_matrix)
        
        # Objective function (minimize risk concentration)
        def objective(weights):
            portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(covariance_matrix, weights)))
            marginal_contrib = np.dot(covariance_matrix, weights) / portfolio_vol
            contrib = weights * marginal_contrib
            
            # Minimize variance of risk contributions
            target_contrib = portfolio_vol / n_assets
            return np.sum((contrib - target_contrib) ** 2)
        
        # Constraints
        cons = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
        ]
        
        # Bounds (all positive weights)
        bounds = tuple((0.001, 1) for _ in range(n_assets))
        
        # Initial guess
        x0 = np.array([1.0 / n_assets] * n_assets)
        
        # Optimize
        result = minimize(
            objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=cons
        )
        
        if result.success:
            weights = pd.Series(result.x, index=covariance_matrix.index)
        else:
            weights = pd.Series(1.0 / n_assets, index=covariance_matrix.index)
        
        return weights
    
    def calculate_portfolio_metrics(
        self,
        positions: Dict[str, Position],
        market_prices: Dict[str, float]
    ) -> PortfolioRiskMetrics:
        """
        Calculate comprehensive portfolio risk metrics.
        
        Parameters
        ----------
        positions : Dict[str, Position]
            Current positions
        market_prices : Dict[str, float]
            Current market prices
            
        Returns
        -------
        PortfolioRiskMetrics
            Portfolio risk metrics
        """
        metrics = PortfolioRiskMetrics()
        
        if not positions:
            return metrics
        
        # Calculate portfolio value and weights
        portfolio_value = sum(
            pos.quantity * market_prices.get(symbol, pos.current_price)
            for symbol, pos in positions.items()
        )
        
        weights = {}
        for symbol, pos in positions.items():
            position_value = pos.quantity * market_prices.get(symbol, pos.current_price)
            weights[symbol] = position_value / portfolio_value if portfolio_value > 0 else 0
        
        # Calculate exposures
        long_exposure = sum(
            weights[symbol] for symbol, pos in positions.items()
            if pos.side == PositionSide.LONG
        )
        short_exposure = sum(
            weights[symbol] for symbol, pos in positions.items()
            if pos.side == PositionSide.SHORT
        )
        
        metrics.gross_exposure = long_exposure + short_exposure
        metrics.net_exposure = long_exposure - short_exposure
        metrics.leverage = metrics.gross_exposure
        
        # Calculate concentration
        if weights:
            metrics.concentration = max(abs(w) for w in weights.values())
        
        # Calculate VaR and CVaR
        if not self.historical_returns.empty:
            portfolio_returns = self._calculate_portfolio_returns(weights)
            
            if len(portfolio_returns) > 20:
                metrics.portfolio_var_95 = np.percentile(portfolio_returns, 5)
                metrics.portfolio_var_99 = np.percentile(portfolio_returns, 1)
                metrics.portfolio_cvar_95 = portfolio_returns[
                    portfolio_returns <= metrics.portfolio_var_95
                ].mean()
                
                # Performance metrics
                excess_returns = portfolio_returns - self.risk_free_rate / 252
                if excess_returns.std() > 0:
                    metrics.sharpe_ratio = np.sqrt(252) * excess_returns.mean() / excess_returns.std()
                
                downside_returns = portfolio_returns[portfolio_returns < 0]
                if len(downside_returns) > 0 and downside_returns.std() > 0:
                    metrics.sortino_ratio = np.sqrt(252) * excess_returns.mean() / downside_returns.std()
        
        # Calculate correlations
        if self.correlation_matrix is not None and len(positions) > 1:
            symbols = list(positions.keys())
            if all(s in self.correlation_matrix.index for s in symbols):
                corr_subset = self.correlation_matrix.loc[symbols, symbols]
                
                # Average correlation (excluding diagonal)
                n = len(symbols)
                if n > 1:
                    metrics.avg_correlation = (corr_subset.sum().sum() - n) / (n * (n - 1))
                    
                    # Max correlation
                    np.fill_diagonal(corr_subset.values, 0)
                    metrics.max_correlation = corr_subset.max().max()
        
        # Store metrics
        self.risk_metrics_history.append(metrics)
        
        return metrics
    
    def _calculate_portfolio_returns(
        self,
        weights: Dict[str, float]
    ) -> pd.Series:
        """
        Calculate historical portfolio returns.
        """
        portfolio_returns = pd.Series(dtype=float)
        
        for symbol, weight in weights.items():
            if symbol in self.historical_returns.columns:
                if portfolio_returns.empty:
                    portfolio_returns = self.historical_returns[symbol] * weight
                else:
                    portfolio_returns += self.historical_returns[symbol] * weight
        
        return portfolio_returns
    
    def check_risk_limits(
        self,
        signal: Signal,
        positions: Dict[str, Position],
        portfolio_value: float
    ) -> Tuple[bool, List[str]]:
        """
        Check if signal violates risk limits.
        
        Parameters
        ----------
        signal : Signal
            Trading signal
        positions : Dict[str, Position]
            Current positions
        portfolio_value : float
            Total portfolio value
            
        Returns
        -------
        Tuple[bool, List[str]]
            (is_allowed, list_of_violations)
        """
        violations = []
        
        # Check position size limit
        if signal.quantity:
            position_value = signal.quantity * signal.price if signal.price else 0
            position_pct = position_value / portfolio_value
            
            if position_pct > self.risk_limits.max_position_size:
                violations.append(f"Position size {position_pct:.1%} exceeds limit {self.risk_limits.max_position_size:.1%}")
        
        # Check concentration
        current_positions = len(positions)
        if signal.signal_type in [SignalType.BUY, SignalType.SELL]:
            if signal.symbol not in positions:
                current_positions += 1
        
        if current_positions > 0:
            avg_position_size = 1.0 / current_positions
            if avg_position_size > self.risk_limits.max_concentration:
                violations.append(f"Concentration {avg_position_size:.1%} exceeds limit")
        
        # Check correlation
        if self.correlation_matrix is not None and signal.symbol in self.correlation_matrix.index:
            for pos_symbol in positions.keys():
                if pos_symbol in self.correlation_matrix.index:
                    correlation = self.correlation_matrix.loc[signal.symbol, pos_symbol]
                    if abs(correlation) > self.risk_limits.max_correlation:
                        violations.append(f"Correlation with {pos_symbol} ({correlation:.2f}) exceeds limit")
        
        # Check daily trades limit
        if self.daily_trades_count >= self.risk_limits.max_daily_trades:
            violations.append(f"Daily trades limit ({self.risk_limits.max_daily_trades}) reached")
        
        # Check leverage
        total_exposure = sum(abs(pos.value) for pos in positions.values())
        leverage = total_exposure / portfolio_value if portfolio_value > 0 else 0
        
        if leverage > self.risk_limits.max_portfolio_leverage:
            violations.append(f"Leverage {leverage:.1f}x exceeds limit {self.risk_limits.max_portfolio_leverage:.1f}x")
        
        is_allowed = len(violations) == 0
        
        if not is_allowed:
            logger.warning(f"Risk limits violated for {signal.symbol}: {violations}")
        
        return is_allowed, violations
    
    def update_historical_data(
        self,
        returns_data: pd.DataFrame
    ) -> None:
        """
        Update historical returns data.
        
        Parameters
        ----------
        returns_data : pd.DataFrame
            Historical returns data
        """
        self.historical_returns = returns_data
        
        # Update correlation and covariance matrices
        if len(returns_data) > 20:
            self.correlation_matrix = returns_data.corr()
            
            # Use Ledoit-Wolf shrinkage for more stable covariance
            lw = LedoitWolf()
            self.covariance_matrix = pd.DataFrame(
                lw.fit(returns_data.dropna()).covariance_,
                index=returns_data.columns,
                columns=returns_data.columns
            )
    
    def stress_test(
        self,
        positions: Dict[str, Position],
        scenarios: Optional[Dict[str, Dict[str, float]]] = None
    ) -> Dict[str, float]:
        """
        Run stress tests on portfolio.
        
        Parameters
        ----------
        positions : Dict[str, Position]
            Current positions
        scenarios : Optional[Dict[str, Dict[str, float]]]
            Stress scenarios (symbol -> price_change_pct)
            
        Returns
        -------
        Dict[str, float]
            Scenario results
        """
        if scenarios is None:
            # Default stress scenarios
            scenarios = {
                "market_crash": {symbol: -0.20 for symbol in positions.keys()},
                "market_rally": {symbol: 0.15 for symbol in positions.keys()},
                "high_volatility": {symbol: np.random.normal(0, 0.05) for symbol in positions.keys()},
                "correlation_breakdown": {symbol: np.random.normal(0, 0.10) for symbol in positions.keys()}
            }
        
        results = {}
        
        for scenario_name, price_changes in scenarios.items():
            portfolio_pnl = 0.0
            
            for symbol, position in positions.items():
                if symbol in price_changes:
                    price_change = price_changes[symbol]
                    
                    if position.side == PositionSide.LONG:
                        position_pnl = position.quantity * position.current_price * price_change
                    else:  # SHORT
                        position_pnl = -position.quantity * position.current_price * price_change
                    
                    portfolio_pnl += position_pnl
            
            results[scenario_name] = portfolio_pnl
            
            logger.info(f"Stress test '{scenario_name}': P&L = ${portfolio_pnl:,.2f}")
        
        return results
    
    def calculate_var(
        self,
        positions: Dict[str, Position],
        confidence_level: float = 0.95,
        time_horizon: int = 1,
        method: str = "historical"
    ) -> float:
        """
        Calculate Value at Risk.
        
        Parameters
        ----------
        positions : Dict[str, Position]
            Current positions
        confidence_level : float
            Confidence level (e.g., 0.95 for 95%)
        time_horizon : int
            Time horizon in days
        method : str
            VaR method: 'historical', 'parametric', 'monte_carlo'
            
        Returns
        -------
        float
            VaR value
        """
        if not positions or self.historical_returns.empty:
            return 0.0
        
        # Calculate portfolio weights
        total_value = sum(pos.value for pos in positions.values())
        weights = {
            symbol: pos.value / total_value
            for symbol, pos in positions.items()
            if total_value > 0
        }
        
        if method == "historical":
            # Historical VaR
            portfolio_returns = self._calculate_portfolio_returns(weights)
            
            if len(portfolio_returns) > 20:
                # Scale to time horizon
                scaled_returns = portfolio_returns * np.sqrt(time_horizon)
                var = np.percentile(scaled_returns, (1 - confidence_level) * 100)
                return abs(var) * total_value
        
        elif method == "parametric":
            # Parametric VaR (assumes normal distribution)
            portfolio_returns = self._calculate_portfolio_returns(weights)
            
            if len(portfolio_returns) > 20:
                mean_return = portfolio_returns.mean() * time_horizon
                std_return = portfolio_returns.std() * np.sqrt(time_horizon)
                
                z_score = stats.norm.ppf(1 - confidence_level)
                var = abs(mean_return + z_score * std_return) * total_value
                return var
        
        elif method == "monte_carlo":
            # Monte Carlo VaR
            n_simulations = 10000
            
            if self.covariance_matrix is not None:
                # Generate random returns
                symbols = list(weights.keys())
                weights_array = np.array([weights.get(s, 0) for s in symbols])
                
                # Filter covariance matrix
                cov_subset = self.covariance_matrix.loc[symbols, symbols].values
                
                # Simulate returns
                simulated_returns = np.random.multivariate_normal(
                    mean=np.zeros(len(symbols)),
                    cov=cov_subset * time_horizon,
                    size=n_simulations
                )
                
                # Calculate portfolio returns
                portfolio_simulated = simulated_returns @ weights_array
                
                # Calculate VaR
                var = np.percentile(portfolio_simulated, (1 - confidence_level) * 100)
                return abs(var) * total_value
        
        return 0.0
    
    def reset_daily_counters(self) -> None:
        """Reset daily risk counters."""
        self.daily_trades_count = 0
        self.daily_turnover = 0.0
        logger.info("Daily risk counters reset")
