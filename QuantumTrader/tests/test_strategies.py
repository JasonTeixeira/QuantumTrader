"""
Professional Testing Suite for Trading Strategies

Comprehensive unit and integration tests ensuring strategy reliability,
performance validation, and edge case handling.

Author: QuantumTrader Team
Date: 2024
"""

import asyncio
import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from typing import Dict, List, Any

from src.strategies.mean_reversion import MeanReversionStrategy, MeanReversionConfig
from src.strategies.momentum import MomentumStrategy, MomentumConfig
from src.engine.base_strategy import Signal, SignalType
from src.data.models import MarketData
from src.risk.risk_manager import RiskManager, RiskLimits


class TestMeanReversionStrategy:
    """Comprehensive tests for Mean Reversion Strategy."""
    
    @pytest.fixture
    def strategy_config(self):
        """Create test strategy configuration."""
        return MeanReversionConfig(
            name="test_mean_reversion",
            symbols=["AAPL", "GOOGL"],
            lookback_period=20,
            entry_z_score=2.0,
            exit_z_score=0.5,
            use_dynamic_thresholds=True,
            max_positions=5
        )
    
    @pytest.fixture
    def strategy(self, strategy_config):
        """Create strategy instance."""
        return MeanReversionStrategy(strategy_config)
    
    @pytest.fixture
    def sample_data(self):
        """Generate sample market data."""
        dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
        data = {}
        
        for symbol in ["AAPL", "GOOGL"]:
            np.random.seed(42 if symbol == "AAPL" else 43)
            prices = 100 + np.cumsum(np.random.randn(100) * 2)
            
            data[symbol] = pd.DataFrame({
                'open': prices + np.random.randn(100) * 0.5,
                'high': prices + abs(np.random.randn(100)),
                'low': prices - abs(np.random.randn(100)),
                'close': prices,
                'volume': np.random.randint(1000000, 10000000, 100)
            }, index=dates)
        
        return data
    
    def test_initialization(self, strategy, strategy_config):
        """Test strategy initialization."""
        assert strategy.config == strategy_config
        assert strategy.name == "test_mean_reversion"
        assert len(strategy.symbols) == 2
        assert strategy.lookback_period == 20
        assert strategy.entry_z_score == 2.0
    
    def test_calculate_bollinger_bands(self, strategy, sample_data):
        """Test Bollinger Bands calculation."""
        symbol = "AAPL"
        bb_upper, bb_middle, bb_lower = strategy._calculate_bollinger_bands(
            sample_data[symbol], 20, 2
        )
        
        assert len(bb_upper) == len(sample_data[symbol])
        assert len(bb_middle) == len(sample_data[symbol])
        assert len(bb_lower) == len(sample_data[symbol])
        
        # Check mathematical relationships
        assert all(bb_upper >= bb_middle)
        assert all(bb_middle >= bb_lower)
        
        # Check NaN handling
        assert bb_upper[19:].notna().all()  # After lookback period
    
    def test_calculate_rsi(self, strategy, sample_data):
        """Test RSI calculation."""
        symbol = "AAPL"
        rsi = strategy._calculate_rsi(sample_data[symbol], 14)
        
        assert len(rsi) == len(sample_data[symbol])
        assert all(0 <= x <= 100 for x in rsi.dropna())
        
        # Test extreme cases
        # All up moves should give RSI close to 100
        up_data = sample_data[symbol].copy()
        up_data['close'] = range(len(up_data))
        rsi_up = strategy._calculate_rsi(up_data, 14)
        assert rsi_up.iloc[-1] > 70  # Should be overbought
    
    def test_calculate_z_score(self, strategy, sample_data):
        """Test Z-score calculation."""
        symbol = "AAPL"
        z_score = strategy._calculate_z_score(sample_data[symbol], 20)
        
        assert len(z_score) == len(sample_data[symbol])
        
        # Check statistical properties
        valid_z = z_score[19:]  # After lookback
        assert abs(valid_z.mean()) < 0.5  # Should be centered around 0
        assert 0.5 < valid_z.std() < 2  # Reasonable standard deviation
    
    def test_signal_generation(self, strategy, sample_data):
        """Test signal generation logic."""
        strategy.initialize()
        
        # Process data
        for symbol, data in sample_data.items():
            strategy.market_data[symbol] = data
        
        # Generate signals
        signals = []
        for i in range(50, len(sample_data["AAPL"])):
            current_data = {
                symbol: data.iloc[:i+1]
                for symbol, data in sample_data.items()
            }
            signal = strategy.generate_signals(current_data)
            if signal:
                signals.extend(signal)
        
        # Validate signals
        assert len(signals) > 0
        
        for signal in signals:
            assert isinstance(signal, Signal)
            assert signal.symbol in ["AAPL", "GOOGL"]
            assert signal.signal_type in [SignalType.BUY, SignalType.SELL, SignalType.HOLD]
            assert 0 <= signal.strength <= 1
            assert signal.metadata is not None
    
    def test_pairs_trading_cointegration(self, strategy):
        """Test pairs trading cointegration check."""
        # Create cointegrated series
        np.random.seed(42)
        series1 = pd.Series(np.cumsum(np.random.randn(100)) + 100)
        series2 = series1 * 1.5 + np.random.randn(100) * 2 + 50
        
        is_coint, p_value = strategy._check_cointegration(series1, series2)
        
        assert isinstance(is_coint, bool)
        assert 0 <= p_value <= 1
    
    def test_position_sizing(self, strategy):
        """Test position sizing calculations."""
        signal = Signal(
            symbol="AAPL",
            signal_type=SignalType.BUY,
            strength=0.8,
            price=150.0,
            timestamp=datetime.now(),
            metadata={'confidence': 0.9}
        )
        
        position_size = strategy._calculate_position_size(
            signal, 
            portfolio_value=100000,
            current_positions=2
        )
        
        assert position_size > 0
        assert position_size <= 100000 * 0.2  # Max 20% per position
    
    def test_risk_limits(self, strategy, sample_data):
        """Test risk limit enforcement."""
        strategy.initialize()
        strategy.max_positions = 2
        
        # Generate many signals
        signals = []
        for i in range(10):
            signals.append(Signal(
                symbol=f"TEST{i}",
                signal_type=SignalType.BUY,
                strength=0.9,
                price=100.0,
                timestamp=datetime.now()
            ))
        
        # Apply risk limits
        filtered_signals = strategy._apply_risk_limits(signals, current_positions=1)
        
        # Should respect max positions
        assert len(filtered_signals) <= 1  # 2 max - 1 current = 1 allowed
    
    @pytest.mark.asyncio
    async def test_concurrent_signal_generation(self, strategy, sample_data):
        """Test thread-safe concurrent signal generation."""
        strategy.initialize()
        strategy.market_data = sample_data
        
        async def generate_signals_async():
            return strategy.generate_signals(sample_data)
        
        # Run multiple concurrent signal generations
        tasks = [generate_signals_async() for _ in range(10)]
        results = await asyncio.gather(*tasks)
        
        # All should complete without errors
        assert len(results) == 10
    
    def test_edge_cases(self, strategy):
        """Test edge cases and error handling."""
        # Empty data
        empty_data = pd.DataFrame()
        result = strategy._calculate_bollinger_bands(empty_data, 20, 2)
        assert all(r.empty for r in result)
        
        # Insufficient data
        small_data = pd.DataFrame({'close': [100, 101]})
        z_score = strategy._calculate_z_score(small_data, 20)
        assert len(z_score) == 2
        assert z_score.isna().all()
        
        # Single value
        single_data = pd.DataFrame({'close': [100]})
        rsi = strategy._calculate_rsi(single_data, 14)
        assert len(rsi) == 1


class TestMomentumStrategy:
    """Comprehensive tests for Momentum Strategy."""
    
    @pytest.fixture
    def strategy_config(self):
        """Create test strategy configuration."""
        return MomentumConfig(
            name="test_momentum",
            symbols=["SPY", "QQQ"],
            fast_ma_period=10,
            slow_ma_period=30,
            rsi_period=14,
            rsi_overbought=70,
            rsi_oversold=30,
            volume_factor=1.5,
            use_volume_confirmation=True
        )
    
    @pytest.fixture
    def strategy(self, strategy_config):
        """Create strategy instance."""
        return MomentumStrategy(strategy_config)
    
    @pytest.fixture
    def trending_data(self):
        """Generate trending market data."""
        dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
        
        # Create uptrend
        trend = np.linspace(100, 150, 100)
        noise = np.random.randn(100) * 2
        prices = trend + noise
        
        return pd.DataFrame({
            'open': prices - 1,
            'high': prices + 2,
            'low': prices - 2,
            'close': prices,
            'volume': np.random.randint(1000000, 5000000, 100)
        }, index=dates)
    
    def test_moving_average_crossover(self, strategy, trending_data):
        """Test moving average crossover detection."""
        fast_ma = trending_data['close'].rolling(10).mean()
        slow_ma = trending_data['close'].rolling(30).mean()
        
        # Detect crossovers
        crossovers = strategy._detect_ma_crossover(fast_ma, slow_ma)
        
        assert 'golden_cross' in crossovers
        assert 'death_cross' in crossovers
        
        # In uptrend, should have golden crosses
        assert crossovers['golden_cross'].any()
    
    def test_macd_calculation(self, strategy, trending_data):
        """Test MACD calculation."""
        macd_line, signal_line, histogram = strategy._calculate_macd(
            trending_data, 12, 26, 9
        )
        
        assert len(macd_line) == len(trending_data)
        assert len(signal_line) == len(trending_data)
        assert len(histogram) == len(trending_data)
        
        # Check mathematical relationship
        assert np.allclose(
            histogram.dropna(), 
            (macd_line - signal_line).dropna(),
            rtol=1e-10
        )
    
    def test_adx_calculation(self, strategy, trending_data):
        """Test ADX trend strength calculation."""
        adx = strategy._calculate_adx(trending_data, 14)
        
        assert len(adx) == len(trending_data)
        assert all(0 <= x <= 100 for x in adx.dropna())
        
        # Strong trend should have high ADX
        assert adx.iloc[-1] > 25  # Indicates trending market
    
    def test_breakout_detection(self, strategy, trending_data):
        """Test breakout detection logic."""
        breakouts = strategy._detect_breakouts(trending_data, 20, 2.0)
        
        assert 'resistance_break' in breakouts
        assert 'support_break' in breakouts
        
        # In uptrend, should have resistance breaks
        assert breakouts['resistance_break'].any()
    
    def test_volume_confirmation(self, strategy, trending_data):
        """Test volume confirmation logic."""
        # Create volume spike
        trending_data.loc[trending_data.index[50], 'volume'] *= 3
        
        volume_confirmed = strategy._check_volume_confirmation(
            trending_data, 
            trending_data.index[50],
            1.5
        )
        
        assert volume_confirmed is True
        
        # Normal volume should not confirm
        normal_confirmed = strategy._check_volume_confirmation(
            trending_data,
            trending_data.index[30],
            1.5
        )
        
        assert normal_confirmed is False
    
    def test_trailing_stop_calculation(self, strategy):
        """Test trailing stop loss calculation."""
        entry_price = 100.0
        current_price = 110.0
        atr = 2.0
        
        stop_loss = strategy._calculate_trailing_stop(
            entry_price, current_price, atr, multiplier=2.0
        )
        
        assert stop_loss < current_price
        assert stop_loss > entry_price  # Should trail up in profit
        
        # Test for short position
        stop_loss_short = strategy._calculate_trailing_stop(
            entry_price, 90.0, atr, multiplier=2.0, is_long=False
        )
        
        assert stop_loss_short > 90.0  # Stop above current for short
    
    def test_relative_strength_ranking(self, strategy):
        """Test relative strength ranking."""
        # Create performance data
        returns = {
            'AAPL': 0.15,
            'GOOGL': 0.10,
            'MSFT': 0.20,
            'AMZN': 0.05
        }
        
        rankings = strategy._rank_by_relative_strength(returns)
        
        assert rankings[0] == 'MSFT'  # Highest return
        assert rankings[-1] == 'AMZN'  # Lowest return
        assert len(rankings) == 4
    
    def test_signal_filtering(self, strategy):
        """Test signal filtering and prioritization."""
        signals = [
            Signal('A', SignalType.BUY, 0.9, 100, datetime.now()),
            Signal('B', SignalType.BUY, 0.5, 100, datetime.now()),
            Signal('C', SignalType.BUY, 0.7, 100, datetime.now()),
            Signal('D', SignalType.SELL, 0.8, 100, datetime.now()),
        ]
        
        filtered = strategy._filter_signals_by_strength(signals, min_strength=0.6)
        
        assert len(filtered) == 3  # Only signals >= 0.6
        assert all(s.strength >= 0.6 for s in filtered)


class TestIntegration:
    """Integration tests for complete system."""
    
    @pytest.fixture
    def full_system(self):
        """Create complete system setup."""
        return {
            'mean_reversion': MeanReversionStrategy(MeanReversionConfig(
                name="mr_test",
                symbols=["AAPL", "GOOGL"]
            )),
            'momentum': MomentumStrategy(MomentumConfig(
                name="mom_test",
                symbols=["SPY", "QQQ"]
            )),
            'risk_manager': RiskManager()
        }
    
    def test_strategy_risk_integration(self, full_system):
        """Test strategy and risk manager integration."""
        strategy = full_system['mean_reversion']
        risk_manager = full_system['risk_manager']
        
        # Generate signal
        signal = Signal(
            symbol="AAPL",
            signal_type=SignalType.BUY,
            strength=0.8,
            price=150.0,
            timestamp=datetime.now()
        )
        
        # Check risk limits
        is_allowed, violations = risk_manager.check_risk_limits(
            signal=signal,
            positions={},
            portfolio_value=100000
        )
        
        assert isinstance(is_allowed, bool)
        assert isinstance(violations, list)
    
    @pytest.mark.asyncio
    async def test_concurrent_strategy_execution(self, full_system):
        """Test concurrent execution of multiple strategies."""
        strategies = [full_system['mean_reversion'], full_system['momentum']]
        
        async def run_strategy(strategy):
            strategy.initialize()
            return strategy.name
        
        results = await asyncio.gather(*[run_strategy(s) for s in strategies])
        
        assert len(results) == 2
        assert 'mr_test' in results
        assert 'mom_test' in results
    
    def test_end_to_end_signal_flow(self, full_system):
        """Test complete signal generation to risk check flow."""
        strategy = full_system['momentum']
        risk_manager = full_system['risk_manager']
        
        # Create market data
        dates = pd.date_range(start='2024-01-01', periods=50, freq='D')
        data = pd.DataFrame({
            'open': np.random.randn(50) * 2 + 100,
            'high': np.random.randn(50) * 2 + 102,
            'low': np.random.randn(50) * 2 + 98,
            'close': np.random.randn(50) * 2 + 100,
            'volume': np.random.randint(1000000, 5000000, 50)
        }, index=dates)
        
        strategy.initialize()
        strategy.market_data = {'SPY': data, 'QQQ': data}
        
        # Generate signals
        signals = strategy.generate_signals(strategy.market_data)
        
        if signals:
            # Check each signal with risk manager
            for signal in signals:
                is_allowed, _ = risk_manager.check_risk_limits(
                    signal=signal,
                    positions={},
                    portfolio_value=100000
                )
                assert isinstance(is_allowed, bool)


class TestPerformanceAndStress:
    """Performance and stress tests."""
    
    def test_large_data_processing(self):
        """Test with large datasets."""
        # Create large dataset
        dates = pd.date_range(start='2020-01-01', periods=10000, freq='1min')
        large_data = pd.DataFrame({
            'close': np.random.randn(10000) * 2 + 100,
            'volume': np.random.randint(1000, 10000, 10000)
        }, index=dates)
        
        strategy = MeanReversionStrategy(MeanReversionConfig(
            name="stress_test",
            symbols=["TEST"]
        ))
        
        # Should handle large data without error
        z_score = strategy._calculate_z_score(large_data, 20)
        assert len(z_score) == len(large_data)
    
    def test_memory_efficiency(self):
        """Test memory usage with multiple strategies."""
        strategies = []
        
        for i in range(100):
            config = MomentumConfig(
                name=f"strategy_{i}",
                symbols=[f"SYMBOL_{i}"]
            )
            strategies.append(MomentumStrategy(config))
        
        # Should not cause memory issues
        assert len(strategies) == 100
        
        # Clean up
        for strategy in strategies:
            strategy.cleanup()
    
    @pytest.mark.benchmark
    def test_signal_generation_performance(self, benchmark):
        """Benchmark signal generation performance."""
        strategy = MeanReversionStrategy(MeanReversionConfig(
            name="perf_test",
            symbols=["AAPL"]
        ))
        
        data = pd.DataFrame({
            'close': np.random.randn(100) * 2 + 100,
            'volume': np.random.randint(1000000, 5000000, 100)
        })
        
        strategy.initialize()
        strategy.market_data = {'AAPL': data}
        
        # Benchmark signal generation
        result = benchmark(strategy.generate_signals, strategy.market_data)
        
        # Should complete quickly
        assert benchmark.stats['mean'] < 0.1  # Less than 100ms


# Fixtures for pytest
@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def mock_market_data():
    """Create mock market data."""
    dates = pd.date_range(start='2024-01-01', periods=30, freq='D')
    return pd.DataFrame({
        'open': np.random.randn(30) * 2 + 100,
        'high': np.random.randn(30) * 2 + 102,
        'low': np.random.randn(30) * 2 + 98,
        'close': np.random.randn(30) * 2 + 100,
        'volume': np.random.randint(1000000, 5000000, 30)
    }, index=dates)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--cov=src", "--cov-report=html"])
