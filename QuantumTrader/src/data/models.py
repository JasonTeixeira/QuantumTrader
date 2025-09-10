"""
Data Models and Schemas for QuantumTrader

This module defines data models and schemas for market data, orders, and trades.

Author: QuantumTrader Team
Date: 2024
"""

from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, validator
from sqlalchemy import (
    Column, DateTime, Float, Integer, String, Text, 
    Boolean, JSON, Index, ForeignKey, Enum as SQLEnum
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()


# Enums
class MarketDataSource(str, Enum):
    """Market data sources."""
    
    YAHOO = "yahoo"
    ALPHA_VANTAGE = "alpha_vantage"
    POLYGON = "polygon"
    IEX = "iex"
    BINANCE = "binance"
    COINBASE = "coinbase"
    INTERACTIVE_BROKERS = "interactive_brokers"
    QUANDL = "quandl"
    BLOOMBERG = "bloomberg"


class AssetClass(str, Enum):
    """Asset classes."""
    
    EQUITY = "equity"
    FOREX = "forex"
    CRYPTO = "crypto"
    COMMODITY = "commodity"
    FIXED_INCOME = "fixed_income"
    DERIVATIVE = "derivative"
    INDEX = "index"


class Exchange(str, Enum):
    """Exchanges."""
    
    NYSE = "NYSE"
    NASDAQ = "NASDAQ"
    LSE = "LSE"
    TSE = "TSE"
    HKEX = "HKEX"
    SSE = "SSE"
    BINANCE = "BINANCE"
    COINBASE = "COINBASE"
    CME = "CME"
    FOREX = "FOREX"


# Pydantic Models for API
class MarketDataSchema(BaseModel):
    """Market data schema for API."""
    
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    vwap: Optional[float] = None
    bid: Optional[float] = None
    ask: Optional[float] = None
    bid_size: Optional[float] = None
    ask_size: Optional[float] = None
    trades_count: Optional[int] = None
    source: MarketDataSource
    
    @validator("volume")
    def validate_volume(cls, v):
        """Validate volume is non-negative."""
        if v < 0:
            raise ValueError("Volume must be non-negative")
        return v
    
    class Config:
        use_enum_values = True
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class OrderBookLevel(BaseModel):
    """Order book level."""
    
    price: float
    size: float
    orders_count: Optional[int] = None
    
    @validator("size")
    def validate_size(cls, v):
        """Validate size is positive."""
        if v <= 0:
            raise ValueError("Size must be positive")
        return v


class OrderBookSnapshot(BaseModel):
    """Order book snapshot."""
    
    symbol: str
    timestamp: datetime
    bids: List[OrderBookLevel]
    asks: List[OrderBookLevel]
    sequence_number: Optional[int] = None
    
    @property
    def best_bid(self) -> Optional[float]:
        """Get best bid price."""
        return self.bids[0].price if self.bids else None
    
    @property
    def best_ask(self) -> Optional[float]:
        """Get best ask price."""
        return self.asks[0].price if self.asks else None
    
    @property
    def spread(self) -> Optional[float]:
        """Get bid-ask spread."""
        if self.best_bid and self.best_ask:
            return self.best_ask - self.best_bid
        return None
    
    @property
    def mid_price(self) -> Optional[float]:
        """Get mid price."""
        if self.best_bid and self.best_ask:
            return (self.best_bid + self.best_ask) / 2
        return None


class TradeData(BaseModel):
    """Trade execution data."""
    
    trade_id: str
    symbol: str
    timestamp: datetime
    price: float
    size: float
    side: str  # "buy" or "sell"
    order_id: Optional[str] = None
    venue: Optional[str] = None
    fee: float = 0.0
    
    @validator("side")
    def validate_side(cls, v):
        """Validate trade side."""
        if v not in ["buy", "sell"]:
            raise ValueError("Side must be 'buy' or 'sell'")
        return v


# SQLAlchemy Models for Database
class Symbol(Base):
    """Symbol database model."""
    
    __tablename__ = "symbols"
    
    id = Column(Integer, primary_key=True)
    symbol = Column(String(50), unique=True, nullable=False, index=True)
    name = Column(String(200))
    asset_class = Column(SQLEnum(AssetClass), nullable=False)
    exchange = Column(SQLEnum(Exchange))
    currency = Column(String(10))
    sector = Column(String(100))
    industry = Column(String(100))
    market_cap = Column(Float)
    is_active = Column(Boolean, default=True)
    metadata = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    market_data = relationship("MarketData", back_populates="symbol_rel", cascade="all, delete-orphan")
    
    __table_args__ = (
        Index("idx_symbol_asset_class", "asset_class"),
        Index("idx_symbol_exchange", "exchange"),
    )


class MarketData(Base):
    """Market data database model."""
    
    __tablename__ = "market_data"
    
    id = Column(Integer, primary_key=True)
    symbol_id = Column(Integer, ForeignKey("symbols.id"), nullable=False)
    timestamp = Column(DateTime, nullable=False, index=True)
    open = Column(Float, nullable=False)
    high = Column(Float, nullable=False)
    low = Column(Float, nullable=False)
    close = Column(Float, nullable=False)
    volume = Column(Float, nullable=False)
    vwap = Column(Float)
    bid = Column(Float)
    ask = Column(Float)
    bid_size = Column(Float)
    ask_size = Column(Float)
    trades_count = Column(Integer)
    source = Column(SQLEnum(MarketDataSource), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    symbol_rel = relationship("Symbol", back_populates="market_data")
    
    __table_args__ = (
        Index("idx_market_data_symbol_timestamp", "symbol_id", "timestamp"),
        Index("idx_market_data_source", "source"),
    )


class Trade(Base):
    """Trade execution database model."""
    
    __tablename__ = "trades"
    
    id = Column(Integer, primary_key=True)
    trade_id = Column(String(100), unique=True, nullable=False, index=True)
    symbol_id = Column(Integer, ForeignKey("symbols.id"), nullable=False)
    strategy_id = Column(String(100), index=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    side = Column(String(10), nullable=False)
    price = Column(Float, nullable=False)
    size = Column(Float, nullable=False)
    value = Column(Float, nullable=False)
    fee = Column(Float, default=0.0)
    slippage = Column(Float, default=0.0)
    order_id = Column(String(100), index=True)
    venue = Column(String(50))
    metadata = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    __table_args__ = (
        Index("idx_trade_timestamp", "timestamp"),
        Index("idx_trade_strategy", "strategy_id"),
    )


class Position(Base):
    """Position database model."""
    
    __tablename__ = "positions"
    
    id = Column(Integer, primary_key=True)
    position_id = Column(String(100), unique=True, nullable=False, index=True)
    symbol_id = Column(Integer, ForeignKey("symbols.id"), nullable=False)
    strategy_id = Column(String(100), index=True)
    side = Column(String(10), nullable=False)
    quantity = Column(Float, nullable=False)
    entry_price = Column(Float, nullable=False)
    entry_time = Column(DateTime, nullable=False)
    exit_price = Column(Float)
    exit_time = Column(DateTime)
    realized_pnl = Column(Float, default=0.0)
    unrealized_pnl = Column(Float, default=0.0)
    commission = Column(Float, default=0.0)
    is_open = Column(Boolean, default=True, index=True)
    stop_loss = Column(Float)
    take_profit = Column(Float)
    metadata = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    __table_args__ = (
        Index("idx_position_strategy", "strategy_id"),
        Index("idx_position_is_open", "is_open"),
    )


# Data transformation utilities
def dataframe_to_market_data(
    df: pd.DataFrame,
    symbol: str,
    source: MarketDataSource
) -> List[MarketDataSchema]:
    """
    Convert DataFrame to list of MarketDataSchema objects.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with OHLCV data
    symbol : str
        Symbol name
    source : MarketDataSource
        Data source
        
    Returns
    -------
    List[MarketDataSchema]
        List of market data objects
    """
    market_data = []
    
    for idx, row in df.iterrows():
        data = MarketDataSchema(
            symbol=symbol,
            timestamp=idx if isinstance(idx, datetime) else datetime.fromisoformat(str(idx)),
            open=float(row["open"]),
            high=float(row["high"]),
            low=float(row["low"]),
            close=float(row["close"]),
            volume=float(row["volume"]),
            vwap=float(row["vwap"]) if "vwap" in row else None,
            bid=float(row["bid"]) if "bid" in row else None,
            ask=float(row["ask"]) if "ask" in row else None,
            bid_size=float(row["bid_size"]) if "bid_size" in row else None,
            ask_size=float(row["ask_size"]) if "ask_size" in row else None,
            trades_count=int(row["trades_count"]) if "trades_count" in row else None,
            source=source
        )
        market_data.append(data)
    
    return market_data


def market_data_to_dataframe(
    market_data: List[MarketDataSchema]
) -> pd.DataFrame:
    """
    Convert list of MarketDataSchema objects to DataFrame.
    
    Parameters
    ----------
    market_data : List[MarketDataSchema]
        List of market data objects
        
    Returns
    -------
    pd.DataFrame
        DataFrame with OHLCV data
    """
    if not market_data:
        return pd.DataFrame()
    
    data = []
    for md in market_data:
        data.append({
            "timestamp": md.timestamp,
            "open": md.open,
            "high": md.high,
            "low": md.low,
            "close": md.close,
            "volume": md.volume,
            "vwap": md.vwap,
            "bid": md.bid,
            "ask": md.ask,
            "bid_size": md.bid_size,
            "ask_size": md.ask_size,
            "trades_count": md.trades_count
        })
    
    df = pd.DataFrame(data)
    df.set_index("timestamp", inplace=True)
    df.sort_index(inplace=True)
    
    return df
