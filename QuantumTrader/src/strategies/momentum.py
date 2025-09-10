"""
Momentum Trading Strategy

Implements sophisticated momentum strategies including trend following,
breakout detection, and relative strength analysis.

Author: QuantumTrader Team
Date: 2024
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import MACD, ADXIndicator, EMAIndicator, SMAIndicator
from ta.volatility import AverageTrueRange

from ..engine.base_strategy import (
    BaseStrategy, Signal, SignalType, StrategyConfig, Position
)
from ..utils.logging import get_logger

logger = get_logger()


@dataclass
class MomentumConfig(StrategyConfig):
    """Momentum strategy configuration."""
    
    # Trend parameters
    fast_ma_period: int = 20
    slow_ma_period: int = 50
    trend_ma_period: int = 200
    ma_type: str = "ema"  # "sma", "ema", "wma"
    
    # MACD parameters
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    
    # ADX parameters
    adx_period: int = 14
    adx_threshold: float = 25.0  # Minimum ADX for trend strength
    
    # Breakout parameters
    breakout_period: int = 20
    breakout_volume_factor: float = 1.5
    use_atr_stops: bool = True
    atr_multiplier: float = 2.0
    
    # Relative strength
    use_relative_strength: bool = True
    rs_benchmark: str = "SPY"
    rs_period: int = 60
    
    # Risk parameters
    momentum_threshold: float = 0.02  # 2% minimum momentum
    max_correlation: float = 0.8  # Maximum correlation between positions
    trailing_stop_pct: float = 0.05  # 5% trailing stop
    
    # Position management
    scale_in: bool = True
    scale_out: bool = True
    max_position_age: int = 30  # Maximum days to hold
    
    # Market regime
    use_regime_filter: bool = True
    volatility_threshold: float = 0.03  # Max volatility for entry


class MomentumStrategy(BaseStrategy):
    """
    Advanced momentum strategy with multiple confirmation signals.
    
    Features:
    - Dual moving average crossover
    - MACD confirmation
    - ADX trend strength filter
    - Breakout detection
    - Relative strength analysis
    - Dynamic position sizing based on momentum strength
    """
    
    def __init__(self, config: MomentumConfig):
        """
        Initialize momentum strategy.
        
        Parameters
        ----------
        config : MomentumConfig
            Strategy configuration
        """
        super().__init__(config)
        self.config = config
        
        # Indicator storage
        self.indicators = {}
        self.momentum_scores = {}
        self.regime_states = {}
        
        # Performance tracking
        self.momentum_history = []
        self.breakout_success_rate = 0.65
        self.trend_accuracy = 0.7
        
        # Position tracking
        self.position_momentum = {}
        self.trailing_stops = {}
    
    def _setup_indicators(self) -> None:
        """Set up technical indicators."""
        logger.info("Setting up momentum indicators")
        
        for symbol in self.config.symbols:
            self.indicators[symbol] = {
                'fast_ma': None,
                'slow_ma': None,
                'trend_ma': None,
                'macd': None,
                'adx': None,
                'atr': None,
                'rsi': None,
                'volume_ma': None
            }
            self.momentum_scores[symbol] = 0.0
            self.regime_states[symbol] = 'neutral'
    
    def _setup_risk_management(self) -> None:
        """Set up risk management parameters."""
        logger.info("Setting up momentum risk management")
        
        self.max_sector_exposure = 0.3  # Max 30% in one sector
        self.max_drawdown_limit = 0.15  # 15% max drawdown
        self.position_heat_map = {}
    
    def calculate_signals(
        self,
        market_data: pd.DataFrame,
        **kwargs
    ) -> List[Signal]:
        """
        Calculate momentum trading signals.
        
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
        if len(market_data) < self.config.slow_ma_period:
            return []
        
        signals = []
        symbol = kwargs.get('symbol', 'UNKNOWN')
        
        # Calculate indicators
        indicators = self._calculate_all_indicators(market_data)
        
        # Calculate momentum score
        momentum_score = self._calculate_momentum_score(market_data, indicators)
        self.momentum_scores[symbol] = momentum_score
        
        # Determine market regime
        regime = self._determine_market_regime(market_data, indicators)
        self.regime_states[symbol] = regime
        
        # Generate entry signals
        if symbol not in self.positions:
            # Check for long entry
            if self._check_long_entry(market_data, indicators, momentum_score, regime):
                signal = self._generate_long_signal(market_data, indicators, symbol)
                signals.append(signal)
            
            # Check for short entry
            elif self.config.enable_shorts and \
                 self._check_short_entry(market_data, indicators, momentum_score, regime):
                signal = self._generate_short_signal(market_data, indicators, symbol)
                signals.append(signal)
        
        # Manage existing positions
        else:
            position = self.positions[symbol]
            
            # Update trailing stop
            self._update_trailing_stop(symbol, market_data['close'].iloc[-1])
            
            # Check exit conditions
            exit_signal = self._check_exit_conditions(
                position, market_data, indicators, momentum_score
            )
            if exit_signal:
                signals.append(exit_signal)
            
            # Check for scaling
            elif self.config.scale_in or self.config.scale_out:
                scale_signal = self._check_scaling_conditions(
                    position, market_data, indicators, momentum_score
                )
                if scale_signal:
                    signals.append(scale_signal)
        
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
        
        # Calculate relative strength if enabled
        if self.config.use_relative_strength:
            self._update_relative_strength(data)
        
        # Process each symbol
        for symbol, df in data.items():
            if symbol not in self.config.symbols:
                continue
            
            # Generate signals
            signals = self.calculate_signals(df, symbol=symbol)
            
            # Apply portfolio-level filters
            filtered_signals = self._apply_portfolio_filters(signals, data)
            
            all_signals.extend(filtered_signals)
        
        # Rank signals by momentum strength
        all_signals = self._rank_signals(all_signals)
        
        return all_signals
    
    def calculate_position_size(
        self,
        signal: Signal,
        account_balance: float,
        current_price: float
    ) -> float:
        """
        Calculate position size based on momentum strength.
        
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
        # Base position size
        base_size = account_balance * self.config.position_size
        
        # Adjust by momentum strength
        momentum_factor = signal.metadata.get('momentum_score', 0.5)
        momentum_multiplier = 0.5 + momentum_factor  # 0.5x to 1.5x
        
        # Adjust by volatility
        volatility = signal.metadata.get('volatility', 0.02)
        volatility_multiplier = min(0.02 / volatility, 1.5)  # Inverse volatility sizing
        
        # Adjust by ADX (trend strength)
        adx_value = signal.metadata.get('adx', 25)
        adx_multiplier = min(adx_value / 25, 1.5) if adx_value > 25 else 0.5
        
        # Calculate final position size
        position_value = base_size * momentum_multiplier * volatility_multiplier * adx_multiplier
        
        # Apply maximum position size limit
        max_position = account_balance * 0.2  # Max 20% per position
        position_value = min(position_value, max_position)
        
        return position_value / current_price
    
    def _calculate_all_indicators(self, data: pd.DataFrame) -> Dict:
        """Calculate all technical indicators."""
        indicators = {}
        
        # Moving averages
        if self.config.ma_type == "ema":
            indicators['fast_ma'] = EMAIndicator(
                close=data['close'], window=self.config.fast_ma_period
            ).ema_indicator()
            indicators['slow_ma'] = EMAIndicator(
                close=data['close'], window=self.config.slow_ma_period
            ).ema_indicator()
            indicators['trend_ma'] = EMAIndicator(
                close=data['close'], window=self.config.trend_ma_period
            ).ema_indicator()
        else:
            indicators['fast_ma'] = SMAIndicator(
                close=data['close'], window=self.config.fast_ma_period
            ).sma_indicator()
            indicators['slow_ma'] = SMAIndicator(
                close=data['close'], window=self.config.slow_ma_period
            ).sma_indicator()
            indicators['trend_ma'] = SMAIndicator(
                close=data['close'], window=self.config.trend_ma_period
            ).sma_indicator()
        
        # MACD
        macd = MACD(
            close=data['close'],
            window_slow=self.config.macd_slow,
            window_fast=self.config.macd_fast,
            window_sign=self.config.macd_signal
        )
        indicators['macd'] = macd.macd()
        indicators['macd_signal'] = macd.macd_signal()
        indicators['macd_diff'] = macd.macd_diff()
        
        # ADX
        adx = ADXIndicator(
            high=data['high'],
            low=data['low'],
            close=data['close'],
            window=self.config.adx_period
        )
        indicators['adx'] = adx.adx()
        indicators['adx_pos'] = adx.adx_pos()
        indicators['adx_neg'] = adx.adx_neg()
        
        # ATR
        atr = AverageTrueRange(
            high=data['high'],
            low=data['low'],
            close=data['close'],
            window=14
        )
        indicators['atr'] = atr.average_true_range()
        
        # RSI
        rsi = RSIIndicator(close=data['close'], window=14)
        indicators['rsi'] = rsi.rsi()
        
        # Volume
        indicators['volume_ma'] = data['volume'].rolling(20).mean()
        indicators['volume_ratio'] = data['volume'] / indicators['volume_ma']
        
        # Price momentum
        indicators['returns'] = data['close'].pct_change()
        indicators['momentum'] = data['close'].pct_change(self.config.breakout_period)
        
        # Breakout levels
        indicators['high_breakout'] = data['high'].rolling(self.config.breakout_period).max()
        indicators['low_breakout'] = data['low'].rolling(self.config.breakout_period).min()
        
        return indicators
    
    def _calculate_momentum_score(
        self,
        data: pd.DataFrame,
        indicators: Dict
    ) -> float:
        """
        Calculate comprehensive momentum score.
        
        Returns
        -------
        float
            Momentum score between -1 and 1
        """
        score = 0.0
        weights = 0.0
        
        # Moving average momentum (30% weight)
        if not indicators['fast_ma'].empty:
            ma_momentum = (indicators['fast_ma'].iloc[-1] - indicators['slow_ma'].iloc[-1]) / indicators['slow_ma'].iloc[-1]
            score += np.clip(ma_momentum * 10, -0.3, 0.3)
            weights += 0.3
        
        # MACD momentum (25% weight)
        if not indicators['macd_diff'].empty:
            macd_momentum = indicators['macd_diff'].iloc[-1] / data['close'].iloc[-1]
            score += np.clip(macd_momentum * 20, -0.25, 0.25)
            weights += 0.25
        
        # Price momentum (20% weight)
        if not indicators['momentum'].empty:
            price_momentum = indicators['momentum'].iloc[-1]
            score += np.clip(price_momentum * 2, -0.2, 0.2)
            weights += 0.2
        
        # ADX trend strength (15% weight)
        if not indicators['adx'].empty:
            adx_strength = indicators['adx'].iloc[-1] / 100
            if indicators['adx_pos'].iloc[-1] > indicators['adx_neg'].iloc[-1]:
                score += adx_strength * 0.15
            else:
                score -= adx_strength * 0.15
            weights += 0.15
        
        # RSI momentum (10% weight)
        if not indicators['rsi'].empty:
            rsi_value = indicators['rsi'].iloc[-1]
            rsi_momentum = (rsi_value - 50) / 50
            score += rsi_momentum * 0.1
            weights += 0.1
        
        # Normalize score
        if weights > 0:
            score = score / weights
        
        return np.clip(score, -1.0, 1.0)
    
    def _determine_market_regime(
        self,
        data: pd.DataFrame,
        indicators: Dict
    ) -> str:
        """
        Determine current market regime.
        
        Returns
        -------
        str
            Market regime: 'bullish', 'bearish', 'ranging', 'volatile'
        """
        # Calculate volatility
        returns = data['close'].pct_change()
        volatility = returns.std() * np.sqrt(252)
        
        # Check trend
        current_price = data['close'].iloc[-1]
        trend_ma = indicators['trend_ma'].iloc[-1] if not indicators['trend_ma'].empty else current_price
        
        # ADX for trend strength
        adx_value = indicators['adx'].iloc[-1] if not indicators['adx'].empty else 20
        
        # Determine regime
        if volatility > self.config.volatility_threshold:
            return 'volatile'
        elif adx_value < 20:
            return 'ranging'
        elif current_price > trend_ma * 1.02:
            return 'bullish'
        elif current_price < trend_ma * 0.98:
            return 'bearish'
        else:
            return 'neutral'
    
    def _check_long_entry(
        self,
        data: pd.DataFrame,
        indicators: Dict,
        momentum_score: float,
        regime: str
    ) -> bool:
        """Check conditions for long entry."""
        # Basic conditions
        if momentum_score < self.config.momentum_threshold:
            return False
        
        if regime in ['bearish', 'volatile'] and self.config.use_regime_filter:
            return False
        
        current_price = data['close'].iloc[-1]
        
        # Moving average crossover
        ma_cross = (indicators['fast_ma'].iloc[-1] > indicators['slow_ma'].iloc[-1] and
                   indicators['fast_ma'].iloc[-2] <= indicators['slow_ma'].iloc[-2])
        
        # MACD signal
        macd_signal = indicators['macd_diff'].iloc[-1] > 0
        
        # Breakout detection
        breakout = current_price > indicators['high_breakout'].iloc[-2]
        
        # Volume confirmation
        volume_confirm = indicators['volume_ratio'].iloc[-1] > self.config.breakout_volume_factor
        
        # ADX strength
        adx_confirm = indicators['adx'].iloc[-1] > self.config.adx_threshold
        
        # Combined conditions
        return (ma_cross or breakout) and macd_signal and volume_confirm and adx_confirm
    
    def _check_short_entry(
        self,
        data: pd.DataFrame,
        indicators: Dict,
        momentum_score: float,
        regime: str
    ) -> bool:
        """Check conditions for short entry."""
        # Basic conditions
        if momentum_score > -self.config.momentum_threshold:
            return False
        
        if regime in ['bullish', 'volatile'] and self.config.use_regime_filter:
            return False
        
        current_price = data['close'].iloc[-1]
        
        # Moving average crossover (bearish)
        ma_cross = (indicators['fast_ma'].iloc[-1] < indicators['slow_ma'].iloc[-1] and
                   indicators['fast_ma'].iloc[-2] >= indicators['slow_ma'].iloc[-2])
        
        # MACD signal (bearish)
        macd_signal = indicators['macd_diff'].iloc[-1] < 0
        
        # Breakdown detection
        breakdown = current_price < indicators['low_breakout'].iloc[-2]
        
        # Volume confirmation
        volume_confirm = indicators['volume_ratio'].iloc[-1] > self.config.breakout_volume_factor
        
        # ADX strength
        adx_confirm = indicators['adx'].iloc[-1] > self.config.adx_threshold
        
        # Combined conditions
        return (ma_cross or breakdown) and macd_signal and volume_confirm and adx_confirm
    
    def _generate_long_signal(
        self,
        data: pd.DataFrame,
        indicators: Dict,
        symbol: str
    ) -> Signal:
        """Generate long entry signal."""
        current_price = data['close'].iloc[-1]
        atr = indicators['atr'].iloc[-1]
        
        # Calculate stops
        stop_loss = current_price - (atr * self.config.atr_multiplier) if self.config.use_atr_stops else None
        take_profit = current_price + (atr * self.config.atr_multiplier * 2)
        
        # Initialize trailing stop
        self.trailing_stops[symbol] = current_price * (1 - self.config.trailing_stop_pct)
        
        return Signal(
            timestamp=data.index[-1],
            symbol=symbol,
            signal_type=SignalType.BUY,
            strength=self.momentum_scores[symbol],
            stop_loss=stop_loss,
            take_profit=take_profit,
            confidence=self._calculate_confidence(indicators),
            metadata={
                'momentum_score': self.momentum_scores[symbol],
                'regime': self.regime_states[symbol],
                'adx': indicators['adx'].iloc[-1],
                'volatility': indicators['returns'].std(),
                'volume_ratio': indicators['volume_ratio'].iloc[-1],
                'entry_type': 'momentum_breakout'
            }
        )
    
    def _generate_short_signal(
        self,
        data: pd.DataFrame,
        indicators: Dict,
        symbol: str
    ) -> Signal:
        """Generate short entry signal."""
        current_price = data['close'].iloc[-1]
        atr = indicators['atr'].iloc[-1]
        
        # Calculate stops (inverted for short)
        stop_loss = current_price + (atr * self.config.atr_multiplier) if self.config.use_atr_stops else None
        take_profit = current_price - (atr * self.config.atr_multiplier * 2)
        
        # Initialize trailing stop (for shorts)
        self.trailing_stops[symbol] = current_price * (1 + self.config.trailing_stop_pct)
        
        return Signal(
            timestamp=data.index[-1],
            symbol=symbol,
            signal_type=SignalType.SELL,
            strength=abs(self.momentum_scores[symbol]),
            stop_loss=stop_loss,
            take_profit=take_profit,
            confidence=self._calculate_confidence(indicators),
            metadata={
                'momentum_score': self.momentum_scores[symbol],
                'regime': self.regime_states[symbol],
                'adx': indicators['adx'].iloc[-1],
                'volatility': indicators['returns'].std(),
                'volume_ratio': indicators['volume_ratio'].iloc[-1],
                'entry_type': 'momentum_breakdown'
            }
        )
    
    def _check_exit_conditions(
        self,
        position: Position,
        data: pd.DataFrame,
        indicators: Dict,
        momentum_score: float
    ) -> Optional[Signal]:
        """Check exit conditions for existing position."""
        symbol = position.symbol
        current_price = data['close'].iloc[-1]
        
        # Check trailing stop
        if symbol in self.trailing_stops:
            if position.side == position.side.LONG:
                if current_price <= self.trailing_stops[symbol]:
                    return Signal(
                        timestamp=data.index[-1],
                        symbol=symbol,
                        signal_type=SignalType.CLOSE_BUY,
                        strength=1.0,
                        metadata={'exit_reason': 'trailing_stop'}
                    )
            else:  # SHORT
                if current_price >= self.trailing_stops[symbol]:
                    return Signal(
                        timestamp=data.index[-1],
                        symbol=symbol,
                        signal_type=SignalType.CLOSE_SELL,
                        strength=1.0,
                        metadata={'exit_reason': 'trailing_stop'}
                    )
        
        # Check momentum reversal
        if position.side == position.side.LONG and momentum_score < -0.1:
            return Signal(
                timestamp=data.index[-1],
                symbol=symbol,
                signal_type=SignalType.CLOSE_BUY,
                strength=abs(momentum_score),
                metadata={'exit_reason': 'momentum_reversal'}
            )
        elif position.side == position.side.SHORT and momentum_score > 0.1:
            return Signal(
                timestamp=data.index[-1],
                symbol=symbol,
                signal_type=SignalType.CLOSE_SELL,
                strength=momentum_score,
                metadata={'exit_reason': 'momentum_reversal'}
            )
        
        # Check max holding period
        if hasattr(position, 'entry_time'):
            holding_days = (datetime.now() - position.entry_time).days
            if holding_days >= self.config.max_position_age:
                signal_type = SignalType.CLOSE_BUY if position.side == position.side.LONG else SignalType.CLOSE_SELL
                return Signal(
                    timestamp=data.index[-1],
                    symbol=symbol,
                    signal_type=signal_type,
                    strength=0.5,
                    metadata={'exit_reason': 'max_holding_period'}
                )
        
        return None
    
    def _check_scaling_conditions(
        self,
        position: Position,
        data: pd.DataFrame,
        indicators: Dict,
        momentum_score: float
    ) -> Optional[Signal]:
        """Check conditions for scaling in/out of position."""
        if not self.config.scale_in and not self.config.scale_out:
            return None
        
        current_price = data['close'].iloc[-1]
        
        # Scale in on continued momentum
        if self.config.scale_in and momentum_score > 0.5:
            if position.side == position.side.LONG and current_price > position.entry_price * 1.02:
                return Signal(
                    timestamp=data.index[-1],
                    symbol=position.symbol,
                    signal_type=SignalType.BUY,
                    strength=momentum_score * 0.5,  # Smaller position
                    metadata={'action': 'scale_in', 'momentum_score': momentum_score}
                )
        
        # Scale out on weakening momentum
        if self.config.scale_out and abs(momentum_score) < 0.2:
            signal_type = SignalType.CLOSE_BUY if position.side == position.side.LONG else SignalType.CLOSE_SELL
            return Signal(
                timestamp=data.index[-1],
                symbol=position.symbol,
                signal_type=signal_type,
                strength=0.5,  # Partial exit
                quantity=position.quantity * 0.5,  # Exit half
                metadata={'action': 'scale_out', 'momentum_score': momentum_score}
            )
        
        return None
    
    def _update_trailing_stop(self, symbol: str, current_price: float) -> None:
        """Update trailing stop for position."""
        if symbol not in self.positions or symbol not in self.trailing_stops:
            return
        
        position = self.positions[symbol]
        
        if position.side == position.side.LONG:
            # For long positions, only move stop up
            new_stop = current_price * (1 - self.config.trailing_stop_pct)
            self.trailing_stops[symbol] = max(self.trailing_stops[symbol], new_stop)
        else:  # SHORT
            # For short positions, only move stop down
            new_stop = current_price * (1 + self.config.trailing_stop_pct)
            self.trailing_stops[symbol] = min(self.trailing_stops[symbol], new_stop)
    
    def _update_relative_strength(self, data: Dict[str, pd.DataFrame]) -> None:
        """Update relative strength calculations."""
        if self.config.rs_benchmark not in data:
            return
        
        benchmark_data = data[self.config.rs_benchmark]
        benchmark_returns = benchmark_data['close'].pct_change(self.config.rs_period).iloc[-1]
        
        for symbol in self.config.symbols:
            if symbol in data:
                symbol_returns = data[symbol]['close'].pct_change(self.config.rs_period).iloc[-1]
                relative_strength = symbol_returns - benchmark_returns
                
                # Store in metadata for later use
                if symbol not in self.indicators:
                    self.indicators[symbol] = {}
                self.indicators[symbol]['relative_strength'] = relative_strength
    
    def _calculate_confidence(self, indicators: Dict) -> float:
        """Calculate confidence score for signal."""
        confidence = 0.5  # Base confidence
        
        # ADX strength adds confidence
        if 'adx' in indicators and not indicators['adx'].empty:
            adx_value = indicators['adx'].iloc[-1]
            if adx_value > 40:
                confidence += 0.2
            elif adx_value > 25:
                confidence += 0.1
        
        # MACD agreement
        if 'macd_diff' in indicators and not indicators['macd_diff'].empty:
            if abs(indicators['macd_diff'].iloc[-1]) > 0:
                confidence += 0.1
        
        # Volume confirmation
        if 'volume_ratio' in indicators and not indicators['volume_ratio'].empty:
            if indicators['volume_ratio'].iloc[-1] > 1.5:
                confidence += 0.1
        
        # RSI not extreme
        if 'rsi' in indicators and not indicators['rsi'].empty:
            rsi = indicators['rsi'].iloc[-1]
            if 30 < rsi < 70:
                confidence += 0.1
        
        return min(confidence, 1.0)
    
    def _apply_portfolio_filters(
        self,
        signals: List[Signal],
        data: Dict[str, pd.DataFrame]
    ) -> List[Signal]:
        """Apply portfolio-level risk filters."""
        filtered = []
        
        for signal in signals:
            # Check correlation with existing positions
            if signal.signal_type in [SignalType.BUY, SignalType.SELL]:
                correlations = self._calculate_correlations(signal.symbol, data)
                
                # Skip if too correlated with existing positions
                max_corr = max(correlations.values()) if correlations else 0
                if max_corr > self.config.max_correlation:
                    logger.info(f"Skipping {signal.symbol} due to high correlation: {max_corr:.2f}")
                    continue
            
            filtered.append(signal)
        
        return filtered
    
    def _calculate_correlations(
        self,
        symbol: str,
        data: Dict[str, pd.DataFrame]
    ) -> Dict[str, float]:
        """Calculate correlations with existing positions."""
        correlations = {}
        
        if symbol not in data:
            return correlations
        
        symbol_returns = data[symbol]['close'].pct_change().dropna()
        
        for pos_symbol in self.positions.keys():
            if pos_symbol in data and pos_symbol != symbol:
                pos_returns = data[pos_symbol]['close'].pct_change().dropna()
                
                # Align series
                aligned = pd.concat([symbol_returns, pos_returns], axis=1).dropna()
                if len(aligned) > 20:
                    corr = aligned.iloc[:, 0].corr(aligned.iloc[:, 1])
                    correlations[pos_symbol] = corr
        
        return correlations
    
    def _rank_signals(self, signals: List[Signal]) -> List[Signal]:
        """Rank signals by momentum strength."""
        # Separate entry and exit signals
        entry_signals = [s for s in signals if s.signal_type in [SignalType.BUY, SignalType.SELL]]
        exit_signals = [s for s in signals if s.signal_type in [SignalType.CLOSE_BUY, SignalType.CLOSE_SELL]]
        
        # Sort entry signals by strength
        entry_signals.sort(key=lambda x: abs(x.strength), reverse=True)
        
        # Combine with exit signals (exits always have priority)
        return exit_signals + entry_signals
