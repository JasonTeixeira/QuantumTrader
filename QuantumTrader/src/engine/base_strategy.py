"""
Base Strategy Abstract Class for QuantumTrader

This module provides the abstract base class for all trading strategies.
All strategies must inherit from this class and implement the required methods.

Author: QuantumTrader Team
Date: 2024
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, validator


class SignalType(Enum):
    """Trading signal types."""
    
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"
    CLOSE = "CLOSE"
    CLOSE_BUY = "CLOSE_BUY"
    CLOSE_SELL = "CLOSE_SELL"


class OrderType(Enum):
    """Order types."""
    
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"
    TRAILING_STOP = "TRAILING_STOP"


class PositionSide(Enum):
    """Position sides."""
    
    LONG = "LONG"
    SHORT = "SHORT"
    FLAT = "FLAT"


@dataclass
class Signal:
    """Trading signal data structure."""
    
    timestamp: datetime
    symbol: str
    signal_type: SignalType
    strength: float  # Signal strength between -1 and 1
    quantity: Optional[float] = None
    price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    confidence: float = 0.0  # Confidence level between 0 and 1
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate signal data."""
        if not -1 <= self.strength <= 1:
            raise ValueError(f"Signal strength must be between -1 and 1, got {self.strength}")
        if not 0 <= self.confidence <= 1:
            raise ValueError(f"Confidence must be between 0 and 1, got {self.confidence}")


@dataclass
class Position:
    """Position data structure."""
    
    symbol: str
    side: PositionSide
    quantity: float
    entry_price: float
    entry_time: datetime
    current_price: float
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    commission: float = 0.0
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def value(self) -> float:
        """Get current position value."""
        return self.quantity * self.current_price
    
    @property
    def pnl_percentage(self) -> float:
        """Get PnL percentage."""
        if self.entry_price == 0:
            return 0.0
        return ((self.current_price - self.entry_price) / self.entry_price) * 100


class StrategyConfig(BaseModel):
    """Base configuration for strategies."""
    
    name: str = Field(..., description="Strategy name")
    version: str = Field("1.0.0", description="Strategy version")
    symbols: List[str] = Field(..., description="Trading symbols")
    timeframe: str = Field("1h", description="Trading timeframe")
    lookback_period: int = Field(100, description="Historical data lookback period")
    max_positions: int = Field(5, description="Maximum concurrent positions")
    position_size: float = Field(0.1, description="Position size as fraction of capital")
    stop_loss: Optional[float] = Field(None, description="Default stop loss percentage")
    take_profit: Optional[float] = Field(None, description="Default take profit percentage")
    enable_shorts: bool = Field(False, description="Enable short positions")
    risk_per_trade: float = Field(0.02, description="Risk per trade as fraction of capital")
    
    @validator("position_size")
    def validate_position_size(cls, v):
        """Validate position size."""
        if not 0 < v <= 1:
            raise ValueError("Position size must be between 0 and 1")
        return v
    
    @validator("risk_per_trade")
    def validate_risk_per_trade(cls, v):
        """Validate risk per trade."""
        if not 0 < v <= 0.1:
            raise ValueError("Risk per trade must be between 0 and 0.1")
        return v


class BaseStrategy(ABC):
    """Abstract base class for trading strategies."""
    
    def __init__(self, config: StrategyConfig):
        """
        Initialize the strategy.
        
        Parameters
        ----------
        config : StrategyConfig
            Strategy configuration
        """
        self.config = config
        self.positions: Dict[str, Position] = {}
        self.signals: List[Signal] = []
        self.performance_metrics: Dict[str, float] = {}
        self.is_initialized = False
        self._logger = None
        
    def initialize(self, **kwargs) -> None:
        """
        Initialize strategy components.
        
        This method should be called before running the strategy.
        Override this method to add custom initialization logic.
        """
        self.is_initialized = True
        self._setup_indicators()
        self._setup_risk_management()
        
    @abstractmethod
    def _setup_indicators(self) -> None:
        """Set up technical indicators used by the strategy."""
        pass
    
    @abstractmethod
    def _setup_risk_management(self) -> None:
        """Set up risk management parameters."""
        pass
    
    @abstractmethod
    def calculate_signals(
        self,
        market_data: pd.DataFrame,
        **kwargs
    ) -> List[Signal]:
        """
        Calculate trading signals based on market data.
        
        Parameters
        ----------
        market_data : pd.DataFrame
            Market data with columns: timestamp, open, high, low, close, volume
        **kwargs
            Additional parameters
            
        Returns
        -------
        List[Signal]
            List of trading signals
        """
        pass
    
    @abstractmethod
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
        pass
    
    @abstractmethod
    def calculate_position_size(
        self,
        signal: Signal,
        account_balance: float,
        current_price: float
    ) -> float:
        """
        Calculate position size for a signal.
        
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
        pass
    
    def update_positions(
        self,
        positions: Dict[str, Position],
        market_data: Dict[str, pd.DataFrame]
    ) -> None:
        """
        Update current positions with latest market data.
        
        Parameters
        ----------
        positions : Dict[str, Position]
            Current positions
        market_data : Dict[str, pd.DataFrame]
            Latest market data
        """
        for symbol, position in positions.items():
            if symbol in market_data:
                current_price = market_data[symbol]["close"].iloc[-1]
                position.current_price = current_price
                
                # Calculate unrealized PnL
                if position.side == PositionSide.LONG:
                    position.unrealized_pnl = (
                        (current_price - position.entry_price) * position.quantity
                    )
                elif position.side == PositionSide.SHORT:
                    position.unrealized_pnl = (
                        (position.entry_price - current_price) * position.quantity
                    )
    
    def check_stop_conditions(self, position: Position) -> Optional[Signal]:
        """
        Check if stop loss or take profit conditions are met.
        
        Parameters
        ----------
        position : Position
            Position to check
            
        Returns
        -------
        Optional[Signal]
            Close signal if conditions are met
        """
        signal = None
        
        if position.stop_loss is not None:
            if position.side == PositionSide.LONG and position.current_price <= position.stop_loss:
                signal = Signal(
                    timestamp=datetime.now(),
                    symbol=position.symbol,
                    signal_type=SignalType.CLOSE_BUY,
                    strength=-1.0,
                    quantity=position.quantity,
                    metadata={"reason": "stop_loss"}
                )
            elif position.side == PositionSide.SHORT and position.current_price >= position.stop_loss:
                signal = Signal(
                    timestamp=datetime.now(),
                    symbol=position.symbol,
                    signal_type=SignalType.CLOSE_SELL,
                    strength=-1.0,
                    quantity=position.quantity,
                    metadata={"reason": "stop_loss"}
                )
        
        if position.take_profit is not None and signal is None:
            if position.side == PositionSide.LONG and position.current_price >= position.take_profit:
                signal = Signal(
                    timestamp=datetime.now(),
                    symbol=position.symbol,
                    signal_type=SignalType.CLOSE_BUY,
                    strength=1.0,
                    quantity=position.quantity,
                    metadata={"reason": "take_profit"}
                )
            elif position.side == PositionSide.SHORT and position.current_price <= position.take_profit:
                signal = Signal(
                    timestamp=datetime.now(),
                    symbol=position.symbol,
                    signal_type=SignalType.CLOSE_SELL,
                    strength=1.0,
                    quantity=position.quantity,
                    metadata={"reason": "take_profit"}
                )
        
        return signal
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """
        Get strategy performance metrics.
        
        Returns
        -------
        Dict[str, float]
            Performance metrics
        """
        total_pnl = sum(p.realized_pnl + p.unrealized_pnl for p in self.positions.values())
        total_commission = sum(p.commission for p in self.positions.values())
        
        self.performance_metrics.update({
            "total_pnl": total_pnl,
            "total_commission": total_commission,
            "net_pnl": total_pnl - total_commission,
            "num_positions": len(self.positions),
            "num_signals": len(self.signals)
        })
        
        return self.performance_metrics
    
    def reset(self) -> None:
        """Reset strategy state."""
        self.positions.clear()
        self.signals.clear()
        self.performance_metrics.clear()
        self.is_initialized = False
    
    def __repr__(self) -> str:
        """String representation of the strategy."""
        return f"{self.__class__.__name__}(name={self.config.name}, version={self.config.version})"
