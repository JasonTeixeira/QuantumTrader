"""
Professional Performance Analytics Module for QuantumTrader

Institutional-grade performance measurement, attribution analysis,
and comprehensive reporting capabilities.

Author: QuantumTrader Team
Date: 2024
"""

import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.optimize import minimize
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

from ..utils.logging import get_logger

logger = get_logger(__name__)
warnings.filterwarnings('ignore')


@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics."""
    
    # Basic Returns
    total_return: float
    annual_return: float
    monthly_return: float
    daily_return: float
    ytd_return: float
    
    # Risk Metrics
    volatility: float
    annual_volatility: float
    downside_deviation: float
    max_drawdown: float
    max_drawdown_duration: int
    var_95: float
    cvar_95: float
    
    # Risk-Adjusted Returns
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    information_ratio: float
    treynor_ratio: float
    
    # Trading Statistics
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    expectancy: float
    payoff_ratio: float
    
    # Drawdown Analysis
    drawdown_periods: List[Dict]
    recovery_times: List[int]
    underwater_curve: pd.Series
    
    # Advanced Metrics
    skewness: float
    kurtosis: float
    tail_ratio: float
    omega_ratio: float
    ulcer_index: float
    
    # Time-based Analysis
    best_month: float
    worst_month: float
    positive_months: float
    rolling_sharpe: pd.Series
    
    # Benchmark Comparison
    alpha: float
    beta: float
    correlation: float
    tracking_error: float
    active_return: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "returns": {
                "total": self.total_return,
                "annual": self.annual_return,
                "monthly": self.monthly_return,
                "ytd": self.ytd_return
            },
            "risk": {
                "volatility": self.annual_volatility,
                "max_drawdown": self.max_drawdown,
                "var_95": self.var_95,
                "cvar_95": self.cvar_95
            },
            "ratios": {
                "sharpe": self.sharpe_ratio,
                "sortino": self.sortino_ratio,
                "calmar": self.calmar_ratio
            },
            "trading": {
                "total_trades": self.total_trades,
                "win_rate": self.win_rate,
                "profit_factor": self.profit_factor,
                "expectancy": self.expectancy
            }
        }


@dataclass
class AttributionAnalysis:
    """Performance attribution analysis."""
    
    # Factor Attribution
    factor_returns: Dict[str, float]
    factor_exposures: Dict[str, float]
    factor_contributions: Dict[str, float]
    
    # Sector Attribution
    sector_allocation: Dict[str, float]
    sector_selection: Dict[str, float]
    sector_interaction: Dict[str, float]
    
    # Time Attribution
    timing_return: float
    selection_return: float
    allocation_return: float
    
    # Risk Attribution
    systematic_risk: float
    specific_risk: float
    total_risk: float
    
    # Style Attribution
    style_factors: Dict[str, float]
    style_contributions: Dict[str, float]
    
    # Brinson Attribution
    allocation_effect: float
    selection_effect: float
    interaction_effect: float
    total_effect: float


class PerformanceAnalyzer:
    """Advanced performance analysis engine."""
    
    def __init__(self, risk_free_rate: float = 0.02):
        self.risk_free_rate = risk_free_rate
        self.cache = {}
        
    def calculate_metrics(
        self,
        returns: pd.Series,
        positions: pd.DataFrame = None,
        trades: pd.DataFrame = None,
        benchmark: pd.Series = None
    ) -> PerformanceMetrics:
        """Calculate comprehensive performance metrics."""
        
        # Ensure returns is a Series with datetime index
        if not isinstance(returns.index, pd.DatetimeIndex):
            returns.index = pd.to_datetime(returns.index)
        
        # Basic returns
        total_return = self._calculate_total_return(returns)
        annual_return = self._calculate_annual_return(returns)
        monthly_return = self._calculate_period_return(returns, 'M')
        daily_return = returns.mean()
        ytd_return = self._calculate_ytd_return(returns)
        
        # Risk metrics
        volatility = returns.std()
        annual_volatility = volatility * np.sqrt(252)
        downside_deviation = self._calculate_downside_deviation(returns)
        max_dd, max_dd_duration = self._calculate_max_drawdown(returns)
        var_95 = self._calculate_var(returns, 0.95)
        cvar_95 = self._calculate_cvar(returns, 0.95)
        
        # Risk-adjusted returns
        sharpe = self._calculate_sharpe_ratio(returns, self.risk_free_rate)
        sortino = self._calculate_sortino_ratio(returns, self.risk_free_rate)
        calmar = self._calculate_calmar_ratio(annual_return, max_dd)
        
        # Trading statistics
        trading_stats = self._calculate_trading_statistics(trades) if trades is not None else {}
        
        # Drawdown analysis
        dd_periods = self._analyze_drawdowns(returns)
        underwater = self._calculate_underwater_curve(returns)
        
        # Advanced metrics
        skewness = stats.skew(returns)
        kurtosis = stats.kurtosis(returns)
        tail_ratio = self._calculate_tail_ratio(returns)
        omega = self._calculate_omega_ratio(returns, self.risk_free_rate)
        ulcer = self._calculate_ulcer_index(returns)
        
        # Time-based analysis
        monthly_returns = returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
        best_month = monthly_returns.max()
        worst_month = monthly_returns.min()
        positive_months = (monthly_returns > 0).mean()
        rolling_sharpe = self._calculate_rolling_sharpe(returns)
        
        # Benchmark comparison
        if benchmark is not None:
            alpha, beta = self._calculate_alpha_beta(returns, benchmark)
            correlation = returns.corr(benchmark)
            tracking_error = (returns - benchmark).std() * np.sqrt(252)
            active_return = annual_return - self._calculate_annual_return(benchmark)
            info_ratio = active_return / tracking_error if tracking_error > 0 else 0
            treynor = (annual_return - self.risk_free_rate) / beta if beta != 0 else 0
        else:
            alpha = beta = correlation = tracking_error = active_return = 0
            info_ratio = treynor = 0
        
        return PerformanceMetrics(
            total_return=total_return,
            annual_return=annual_return,
            monthly_return=monthly_return,
            daily_return=daily_return,
            ytd_return=ytd_return,
            volatility=volatility,
            annual_volatility=annual_volatility,
            downside_deviation=downside_deviation,
            max_drawdown=max_dd,
            max_drawdown_duration=max_dd_duration,
            var_95=var_95,
            cvar_95=cvar_95,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            calmar_ratio=calmar,
            information_ratio=info_ratio,
            treynor_ratio=treynor,
            total_trades=trading_stats.get('total_trades', 0),
            winning_trades=trading_stats.get('winning_trades', 0),
            losing_trades=trading_stats.get('losing_trades', 0),
            win_rate=trading_stats.get('win_rate', 0),
            avg_win=trading_stats.get('avg_win', 0),
            avg_loss=trading_stats.get('avg_loss', 0),
            profit_factor=trading_stats.get('profit_factor', 0),
            expectancy=trading_stats.get('expectancy', 0),
            payoff_ratio=trading_stats.get('payoff_ratio', 0),
            drawdown_periods=dd_periods,
            recovery_times=[p['recovery_days'] for p in dd_periods if p['recovery_days']],
            underwater_curve=underwater,
            skewness=skewness,
            kurtosis=kurtosis,
            tail_ratio=tail_ratio,
            omega_ratio=omega,
            ulcer_index=ulcer,
            best_month=best_month,
            worst_month=worst_month,
            positive_months=positive_months,
            rolling_sharpe=rolling_sharpe,
            alpha=alpha,
            beta=beta,
            correlation=correlation,
            tracking_error=tracking_error,
            active_return=active_return
        )
    
    def calculate_attribution(
        self,
        returns: pd.Series,
        positions: pd.DataFrame,
        factor_returns: pd.DataFrame = None,
        benchmark_weights: pd.DataFrame = None
    ) -> AttributionAnalysis:
        """Perform performance attribution analysis."""
        
        # Factor attribution
        factor_attr = self._factor_attribution(returns, positions, factor_returns) if factor_returns is not None else {}
        
        # Sector attribution
        sector_attr = self._sector_attribution(returns, positions, benchmark_weights) if benchmark_weights is not None else {}
        
        # Time attribution
        timing, selection, allocation = self._time_attribution(returns, positions)
        
        # Risk attribution
        systematic, specific = self._risk_attribution(returns, factor_returns) if factor_returns is not None else (0, 0)
        
        # Style attribution
        style_attr = self._style_attribution(returns, positions)
        
        # Brinson attribution
        brinson = self._brinson_attribution(returns, positions, benchmark_weights) if benchmark_weights is not None else {}
        
        return AttributionAnalysis(
            factor_returns=factor_attr.get('returns', {}),
            factor_exposures=factor_attr.get('exposures', {}),
            factor_contributions=factor_attr.get('contributions', {}),
            sector_allocation=sector_attr.get('allocation', {}),
            sector_selection=sector_attr.get('selection', {}),
            sector_interaction=sector_attr.get('interaction', {}),
            timing_return=timing,
            selection_return=selection,
            allocation_return=allocation,
            systematic_risk=systematic,
            specific_risk=specific,
            total_risk=systematic + specific,
            style_factors=style_attr.get('factors', {}),
            style_contributions=style_attr.get('contributions', {}),
            allocation_effect=brinson.get('allocation', 0),
            selection_effect=brinson.get('selection', 0),
            interaction_effect=brinson.get('interaction', 0),
            total_effect=brinson.get('total', 0)
        )
    
    def _calculate_total_return(self, returns: pd.Series) -> float:
        """Calculate total return."""
        return (1 + returns).prod() - 1
    
    def _calculate_annual_return(self, returns: pd.Series) -> float:
        """Calculate annualized return."""
        total_return = self._calculate_total_return(returns)
        n_years = len(returns) / 252
        return (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0
    
    def _calculate_period_return(self, returns: pd.Series, period: str) -> float:
        """Calculate period return."""
        period_returns = returns.resample(period).apply(lambda x: (1 + x).prod() - 1)
        return period_returns.mean()
    
    def _calculate_ytd_return(self, returns: pd.Series) -> float:
        """Calculate year-to-date return."""
        current_year = datetime.now().year
        ytd_returns = returns[returns.index.year == current_year]
        return self._calculate_total_return(ytd_returns) if len(ytd_returns) > 0 else 0
    
    def _calculate_downside_deviation(self, returns: pd.Series, target: float = 0) -> float:
        """Calculate downside deviation."""
        downside_returns = returns[returns < target]
        return np.sqrt(np.mean(downside_returns ** 2)) if len(downside_returns) > 0 else 0
    
    def _calculate_max_drawdown(self, returns: pd.Series) -> Tuple[float, int]:
        """Calculate maximum drawdown and duration."""
        cum_returns = (1 + returns).cumprod()
        running_max = cum_returns.expanding().max()
        drawdown = (cum_returns - running_max) / running_max
        
        max_drawdown = drawdown.min()
        
        # Calculate duration
        drawdown_start = drawdown[drawdown == max_drawdown].index[0]
        if drawdown[drawdown_start:].max() >= 0:
            recovery_date = drawdown[drawdown_start:][drawdown[drawdown_start:] >= 0].index[0]
            duration = (recovery_date - drawdown_start).days
        else:
            duration = (returns.index[-1] - drawdown_start).days
        
        return abs(max_drawdown), duration
    
    def _calculate_var(self, returns: pd.Series, confidence: float) -> float:
        """Calculate Value at Risk."""
        return np.percentile(returns, (1 - confidence) * 100)
    
    def _calculate_cvar(self, returns: pd.Series, confidence: float) -> float:
        """Calculate Conditional Value at Risk."""
        var = self._calculate_var(returns, confidence)
        return returns[returns <= var].mean()
    
    def _calculate_sharpe_ratio(self, returns: pd.Series, risk_free: float) -> float:
        """Calculate Sharpe ratio."""
        excess_returns = returns - risk_free / 252
        return np.sqrt(252) * excess_returns.mean() / returns.std() if returns.std() > 0 else 0
    
    def _calculate_sortino_ratio(self, returns: pd.Series, risk_free: float) -> float:
        """Calculate Sortino ratio."""
        excess_returns = returns - risk_free / 252
        downside_dev = self._calculate_downside_deviation(excess_returns)
        return np.sqrt(252) * excess_returns.mean() / downside_dev if downside_dev > 0 else 0
    
    def _calculate_calmar_ratio(self, annual_return: float, max_drawdown: float) -> float:
        """Calculate Calmar ratio."""
        return annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
    
    def _calculate_trading_statistics(self, trades: pd.DataFrame) -> Dict[str, float]:
        """Calculate trading statistics."""
        if trades is None or trades.empty:
            return {}
        
        # Calculate P&L for each trade
        if 'pnl' in trades.columns:
            pnl = trades['pnl']
        elif 'exit_price' in trades.columns and 'entry_price' in trades.columns:
            pnl = (trades['exit_price'] - trades['entry_price']) * trades.get('quantity', 1)
        else:
            return {}
        
        winning_trades = pnl[pnl > 0]
        losing_trades = pnl[pnl < 0]
        
        total_trades = len(pnl)
        num_winners = len(winning_trades)
        num_losers = len(losing_trades)
        
        win_rate = num_winners / total_trades if total_trades > 0 else 0
        avg_win = winning_trades.mean() if num_winners > 0 else 0
        avg_loss = abs(losing_trades.mean()) if num_losers > 0 else 0
        
        gross_profit = winning_trades.sum() if num_winners > 0 else 0
        gross_loss = abs(losing_trades.sum()) if num_losers > 0 else 0
        
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        expectancy = pnl.mean()
        payoff_ratio = avg_win / avg_loss if avg_loss > 0 else float('inf')
        
        return {
            'total_trades': total_trades,
            'winning_trades': num_winners,
            'losing_trades': num_losers,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'expectancy': expectancy,
            'payoff_ratio': payoff_ratio,
            'gross_profit': gross_profit,
            'gross_loss': gross_loss
        }
    
    def _analyze_drawdowns(self, returns: pd.Series) -> List[Dict]:
        """Analyze all drawdown periods."""
        cum_returns = (1 + returns).cumprod()
        running_max = cum_returns.expanding().max()
        drawdown = (cum_returns - running_max) / running_max
        
        # Find drawdown periods
        is_drawdown = drawdown < 0
        drawdown_starts = (~is_drawdown).shift(1) & is_drawdown
        drawdown_ends = is_drawdown.shift(1) & (~is_drawdown)
        
        periods = []
        start_dates = drawdown_starts[drawdown_starts].index
        
        for start_date in start_dates:
            # Find end date
            future_ends = drawdown_ends[drawdown_ends.index > start_date]
            if len(future_ends) > 0:
                end_date = future_ends.index[0]
                recovery_days = (end_date - start_date).days
            else:
                end_date = returns.index[-1]
                recovery_days = None
            
            # Calculate drawdown magnitude
            period_drawdown = drawdown[start_date:end_date].min()
            
            periods.append({
                'start': start_date,
                'end': end_date,
                'drawdown': abs(period_drawdown),
                'recovery_days': recovery_days
            })
        
        return periods
    
    def _calculate_underwater_curve(self, returns: pd.Series) -> pd.Series:
        """Calculate underwater curve."""
        cum_returns = (1 + returns).cumprod()
        running_max = cum_returns.expanding().max()
        return (cum_returns - running_max) / running_max
    
    def _calculate_tail_ratio(self, returns: pd.Series) -> float:
        """Calculate tail ratio."""
        right_tail = abs(returns.quantile(0.95))
        left_tail = abs(returns.quantile(0.05))
        return right_tail / left_tail if left_tail > 0 else 0
    
    def _calculate_omega_ratio(self, returns: pd.Series, threshold: float) -> float:
        """Calculate Omega ratio."""
        excess = returns - threshold / 252
        gains = excess[excess > 0].sum()
        losses = abs(excess[excess < 0].sum())
        return gains / losses if losses > 0 else float('inf')
    
    def _calculate_ulcer_index(self, returns: pd.Series) -> float:
        """Calculate Ulcer Index."""
        cum_returns = (1 + returns).cumprod()
        running_max = cum_returns.expanding().max()
        drawdown = (cum_returns - running_max) / running_max
        return np.sqrt(np.mean(drawdown ** 2))
    
    def _calculate_rolling_sharpe(self, returns: pd.Series, window: int = 252) -> pd.Series:
        """Calculate rolling Sharpe ratio."""
        rolling_mean = returns.rolling(window).mean()
        rolling_std = returns.rolling(window).std()
        return np.sqrt(252) * rolling_mean / rolling_std
    
    def _calculate_alpha_beta(self, returns: pd.Series, benchmark: pd.Series) -> Tuple[float, float]:
        """Calculate alpha and beta."""
        # Align series
        aligned = pd.DataFrame({'returns': returns, 'benchmark': benchmark}).dropna()
        
        if len(aligned) < 2:
            return 0, 0
        
        # Calculate beta
        covariance = aligned['returns'].cov(aligned['benchmark'])
        benchmark_var = aligned['benchmark'].var()
        beta = covariance / benchmark_var if benchmark_var > 0 else 0
        
        # Calculate alpha
        returns_mean = aligned['returns'].mean() * 252
        benchmark_mean = aligned['benchmark'].mean() * 252
        alpha = returns_mean - beta * benchmark_mean
        
        return alpha, beta
    
    def _factor_attribution(
        self,
        returns: pd.Series,
        positions: pd.DataFrame,
        factor_returns: pd.DataFrame
    ) -> Dict[str, Any]:
        """Perform factor attribution."""
        if factor_returns is None or factor_returns.empty:
            return {}
        
        # Align data
        aligned = pd.concat([returns, factor_returns], axis=1).dropna()
        
        if len(aligned) < 20:  # Need sufficient data
            return {}
        
        # Run factor regression
        X = aligned[factor_returns.columns]
        y = aligned[returns.name] if returns.name else aligned.iloc[:, 0]
        
        model = LinearRegression()
        model.fit(X, y)
        
        # Calculate factor exposures (betas)
        exposures = dict(zip(factor_returns.columns, model.coef_))
        
        # Calculate factor returns
        factor_period_returns = {}
        for factor in factor_returns.columns:
            factor_period_returns[factor] = factor_returns[factor].mean() * 252
        
        # Calculate factor contributions
        contributions = {}
        for factor, exposure in exposures.items():
            contributions[factor] = exposure * factor_period_returns[factor]
        
        return {
            'returns': factor_period_returns,
            'exposures': exposures,
            'contributions': contributions,
            'r_squared': model.score(X, y)
        }
    
    def _sector_attribution(
        self,
        returns: pd.Series,
        positions: pd.DataFrame,
        benchmark_weights: pd.DataFrame
    ) -> Dict[str, Any]:
        """Perform sector attribution."""
        if positions is None or benchmark_weights is None:
            return {}
        
        # This would require sector classification of positions
        # Simplified implementation
        return {
            'allocation': {},
            'selection': {},
            'interaction': {}
        }
    
    def _time_attribution(
        self,
        returns: pd.Series,
        positions: pd.DataFrame
    ) -> Tuple[float, float, float]:
        """Perform time attribution."""
        if positions is None or positions.empty:
            return 0, 0, 0
        
        # Simplified time attribution
        # Would need intraday data for accurate timing attribution
        timing_return = 0
        selection_return = returns.mean() * 252
        allocation_return = 0
        
        return timing_return, selection_return, allocation_return
    
    def _risk_attribution(
        self,
        returns: pd.Series,
        factor_returns: pd.DataFrame
    ) -> Tuple[float, float]:
        """Perform risk attribution."""
        if factor_returns is None or factor_returns.empty:
            total_risk = returns.std() * np.sqrt(252)
            return 0, total_risk
        
        # Factor model risk decomposition
        factor_attr = self._factor_attribution(returns, None, factor_returns)
        
        if 'r_squared' in factor_attr:
            r_squared = factor_attr['r_squared']
            total_risk = returns.std() * np.sqrt(252)
            systematic_risk = total_risk * np.sqrt(r_squared)
            specific_risk = total_risk * np.sqrt(1 - r_squared)
            return systematic_risk, specific_risk
        
        return 0, returns.std() * np.sqrt(252)
    
    def _style_attribution(
        self,
        returns: pd.Series,
        positions: pd.DataFrame
    ) -> Dict[str, Any]:
        """Perform style attribution."""
        # Style factors: Value, Growth, Momentum, Quality, Size
        style_factors = {
            'value': 0,
            'growth': 0,
            'momentum': 0,
            'quality': 0,
            'size': 0
        }
        
        style_contributions = {
            'value': 0,
            'growth': 0,
            'momentum': 0,
            'quality': 0,
            'size': 0
        }
        
        return {
            'factors': style_factors,
            'contributions': style_contributions
        }
    
    def _brinson_attribution(
        self,
        returns: pd.Series,
        positions: pd.DataFrame,
        benchmark_weights: pd.DataFrame
    ) -> Dict[str, float]:
        """Perform Brinson attribution."""
        if positions is None or benchmark_weights is None:
            return {}
        
        # Simplified Brinson attribution
        # Would need sector/asset returns and weights
        allocation_effect = 0
        selection_effect = 0
        interaction_effect = 0
        
        return {
            'allocation': allocation_effect,
            'selection': selection_effect,
            'interaction': interaction_effect,
            'total': allocation_effect + selection_effect + interaction_effect
        }


class ReportGenerator:
    """Generate professional performance reports."""
    
    def __init__(self):
        self.analyzer = PerformanceAnalyzer()
    
    def generate_tearsheet(
        self,
        returns: pd.Series,
        positions: pd.DataFrame = None,
        trades: pd.DataFrame = None,
        benchmark: pd.Series = None
    ) -> Dict[str, Any]:
        """Generate comprehensive tearsheet."""
        
        # Calculate metrics
        metrics = self.analyzer.calculate_metrics(returns, positions, trades, benchmark)
        
        # Generate sections
        tearsheet = {
            'summary': self._generate_summary(metrics),
            'returns': self._generate_returns_analysis(returns, metrics),
            'risk': self._generate_risk_analysis(metrics),
            'trading': self._generate_trading_analysis(metrics),
            'drawdown': self._generate_drawdown_analysis(metrics),
            'rolling': self._generate_rolling_analysis(returns, metrics),
            'distribution': self._generate_distribution_analysis(returns, metrics)
        }
        
        # Add attribution if positions available
        if positions is not None:
            attribution = self.analyzer.calculate_attribution(returns, positions)
            tearsheet['attribution'] = self._generate_attribution_analysis(attribution)
        
        return tearsheet
    
    def _generate_summary(self, metrics: PerformanceMetrics) -> Dict[str, Any]:
        """Generate summary section."""
        return {
            'total_return': f"{metrics.total_return:.2%}",
            'annual_return': f"{metrics.annual_return:.2%}",
            'volatility': f"{metrics.annual_volatility:.2%}",
            'sharpe_ratio': f"{metrics.sharpe_ratio:.2f}",
            'max_drawdown': f"{metrics.max_drawdown:.2%}",
            'win_rate': f"{metrics.win_rate:.2%}",
            'profit_factor': f"{metrics.profit_factor:.2f}"
        }
    
    def _generate_returns_analysis(self, returns: pd.Series, metrics: PerformanceMetrics) -> Dict[str, Any]:
        """Generate returns analysis."""
        return {
            'periods': {
                'daily': f"{metrics.daily_return:.2%}",
                'monthly': f"{metrics.monthly_return:.2%}",
                'yearly': f"{metrics.annual_return:.2%}",
                'ytd': f"{metrics.ytd_return:.2%}"
            },
            'best_worst': {
                'best_day': f"{returns.max():.2%}",
                'worst_day': f"{returns.min():.2%}",
                'best_month': f"{metrics.best_month:.2%}",
                'worst_month': f"{metrics.worst_month:.2%}"
            },
            'consistency': {
                'positive_days': f"{(returns > 0).mean():.1%}",
                'positive_months': f"{metrics.positive_months:.1%}"
            }
        }
    
    def _generate_risk_analysis(self, metrics: PerformanceMetrics) -> Dict[str, Any]:
        """Generate risk analysis."""
        return {
            'volatility': {
                'daily': f"{metrics.volatility:.2%}",
                'annual': f"{metrics.annual_volatility:.2%}",
                'downside': f"{metrics.downside_deviation:.2%}"
            },
            'var_cvar': {
                'var_95': f"{metrics.var_95:.2%}",
                'cvar_95': f"{metrics.cvar_95:.2%}"
            },
            'ratios': {
                'sharpe': f"{metrics.sharpe_ratio:.2f}",
                'sortino': f"{metrics.sortino_ratio:.2f}",
                'calmar': f"{metrics.calmar_ratio:.2f}",
                'omega': f"{metrics.omega_ratio:.2f}"
            },
            'tail_risk': {
                'skewness': f"{metrics.skewness:.2f}",
                'kurtosis': f"{metrics.kurtosis:.2f}",
                'tail_ratio': f"{metrics.tail_ratio:.2f}"
            }
        }
    
    def _generate_trading_analysis(self, metrics: PerformanceMetrics) -> Dict[str, Any]:
        """Generate trading analysis."""
        return {
            'summary': {
                'total_trades': metrics.total_trades,
                'win_rate': f"{metrics.win_rate:.1%}",
                'profit_factor': f"{metrics.profit_factor:.2f}",
                'expectancy': f"${metrics.expectancy:.2f}"
            },
            'winners_losers': {
                'winning_trades': metrics.winning_trades,
                'losing_trades': metrics.losing_trades,
                'avg_win': f"${metrics.avg_win:.2f}",
                'avg_loss': f"${metrics.avg_loss:.2f}",
                'payoff_ratio': f"{metrics.payoff_ratio:.2f}"
            }
        }
    
    def _generate_drawdown_analysis(self, metrics: PerformanceMetrics) -> Dict[str, Any]:
        """Generate drawdown analysis."""
        return {
            'maximum': {
                'depth': f"{metrics.max_drawdown:.2%}",
                'duration_days': metrics.max_drawdown_duration
            },
            'periods': len(metrics.drawdown_periods),
            'recovery': {
                'avg_days': np.mean(metrics.recovery_times) if metrics.recovery_times else 0,
                'max_days': max(metrics.recovery_times) if metrics.recovery_times else 0
            },
            'ulcer_index': f"{metrics.ulcer_index:.4f}"
        }
    
    def _generate_rolling_analysis(self, returns: pd.Series, metrics: PerformanceMetrics) -> Dict[str, Any]:
        """Generate rolling analysis."""
        return {
            'rolling_sharpe': {
                'current': f"{metrics.rolling_sharpe.iloc[-1]:.2f}" if len(metrics.rolling_sharpe) > 0 else "N/A",
                'mean': f"{metrics.rolling_sharpe.mean():.2f}" if len(metrics.rolling_sharpe) > 0 else "N/A",
                'std': f"{metrics.rolling_sharpe.std():.2f}" if len(metrics.rolling_sharpe) > 0 else "N/A"
            }
        }
    
    def _generate_distribution_analysis(self, returns: pd.Series, metrics: PerformanceMetrics) -> Dict[str, Any]:
        """Generate distribution analysis."""
        return {
            'moments': {
                'mean': f"{returns.mean():.4f}",
                'std': f"{returns.std():.4f}",
                'skewness': f"{metrics.skewness:.2f}",
                'kurtosis': f"{metrics.kurtosis:.2f}"
            },
            'percentiles': {
                '1%': f"{returns.quantile(0.01):.2%}",
                '5%': f"{returns.quantile(0.05):.2%}",
                '25%': f"{returns.quantile(0.25):.2%}",
                '50%': f"{returns.quantile(0.50):.2%}",
                '75%': f"{returns.quantile(0.75):.2%}",
                '95%': f"{returns.quantile(0.95):.2%}",
                '99%': f"{returns.quantile(0.99):.2%}"
            }
        }
    
    def _generate_attribution_analysis(self, attribution: AttributionAnalysis) -> Dict[str, Any]:
        """Generate attribution analysis."""
        return {
            'factor': {
                'contributions': attribution.factor_contributions,
                'exposures': attribution.factor_exposures
            },
            'time': {
                'timing': f"{attribution.timing_return:.2%}",
                'selection': f"{attribution.selection_return:.2%}",
                'allocation': f"{attribution.allocation_return:.2%}"
            },
            'risk': {
                'systematic': f"{attribution.systematic_risk:.2%}",
                'specific': f"{attribution.specific_risk:.2%}",
                'total': f"{attribution.total_risk:.2%}"
            }
        }
    
    def export_html_report(
        self,
        tearsheet: Dict[str, Any],
        filepath: str = "performance_report.html"
    ) -> None:
        """Export tearsheet as HTML report."""
        html_content = self._generate_html(tearsheet)
        
        with open(filepath, 'w') as f:
            f.write(html_content)
        
        logger.info(f"HTML report exported to {filepath}")
    
    def _generate_html(self, tearsheet: Dict[str, Any]) -> str:
        """Generate HTML content for report."""
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Performance Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                h1 { color: #333; }
                h2 { color: #666; border-bottom: 2px solid #ddd; padding-bottom: 5px; }
                table { border-collapse: collapse; width: 100%; margin-bottom: 20px; }
                th, td { padding: 10px; text-align: left; border: 1px solid #ddd; }
                th { background-color: #f4f4f4; }
                .metric { font-weight: bold; color: #2e7d32; }
                .negative { color: #d32f2f; }
            </style>
        </head>
        <body>
            <h1>Performance Report</h1>
        """
        
        # Add sections
        for section_name, section_data in tearsheet.items():
            html += f"<h2>{section_name.replace('_', ' ').title()}</h2>"
            html += self._dict_to_html_table(section_data)
        
        html += """
        </body>
        </html>
        """
        
        return html
    
    def _dict_to_html_table(self, data: Dict) -> str:
        """Convert dictionary to HTML table."""
        if not data:
            return "<p>No data available</p>"
        
        html = "<table>"
        
        for key, value in data.items():
            if isinstance(value, dict):
                html += f"<tr><th colspan='2'>{key.replace('_', ' ').title()}</th></tr>"
                for sub_key, sub_value in value.items():
                    html += f"<tr><td>{sub_key.replace('_', ' ').title()}</td>"
                    html += f"<td class='metric'>{sub_value}</td></tr>"
            else:
                html += f"<tr><td>{key.replace('_', ' ').title()}</td>"
                html += f"<td class='metric'>{value}</td></tr>"
        
        html += "</table>"
        return html


# Example usage
if __name__ == "__main__":
    # Generate sample data
    np.random.seed(42)
    dates = pd.date_range(start='2022-01-01', end='2024-01-01', freq='D')
    returns = pd.Series(np.random.randn(len(dates)) * 0.02, index=dates)
    
    # Create analyzer
    analyzer = PerformanceAnalyzer()
    
    # Calculate metrics
    metrics = analyzer.calculate_metrics(returns)
    
    # Generate tearsheet
    generator = ReportGenerator()
    tearsheet = generator.generate_tearsheet(returns)
    
    # Export report
    generator.export_html_report(tearsheet)
    
    print("Performance Analysis Complete")
    print(f"Total Return: {metrics.total_return:.2%}")
    print(f"Sharpe Ratio: {metrics.sharpe_ratio:.2f}")
    print(f"Max Drawdown: {metrics.max_drawdown:.2%}")
