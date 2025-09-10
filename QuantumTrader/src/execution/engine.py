"""
Professional Execution Engine for QuantumTrader

Institutional-grade order management system with smart order routing,
execution algorithms, and multi-broker support.

Author: QuantumTrader Team
Date: 2024
"""

import asyncio
import hashlib
import time
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sortedcontainers import SortedDict

from ..utils.logging import get_logger

logger = get_logger(__name__)


# Enums for order types and statuses
class OrderType(Enum):
    """Order types."""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"
    TRAILING_STOP = "TRAILING_STOP"
    ICEBERG = "ICEBERG"
    TWAP = "TWAP"
    VWAP = "VWAP"
    POV = "POV"  # Percentage of Volume
    IS = "IS"  # Implementation Shortfall
    MOC = "MOC"  # Market on Close
    MOO = "MOO"  # Market on Open
    PEG = "PEG"  # Pegged order


class OrderSide(Enum):
    """Order side."""
    BUY = "BUY"
    SELL = "SELL"


class OrderStatus(Enum):
    """Order status."""
    PENDING = "PENDING"
    SUBMITTED = "SUBMITTED"
    PARTIAL = "PARTIAL"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"


class TimeInForce(Enum):
    """Time in force."""
    DAY = "DAY"
    GTC = "GTC"  # Good Till Cancelled
    IOC = "IOC"  # Immediate or Cancel
    FOK = "FOK"  # Fill or Kill
    GTD = "GTD"  # Good Till Date
    ATO = "ATO"  # At the Open
    ATC = "ATC"  # At the Close


class ExecutionAlgo(Enum):
    """Execution algorithms."""
    AGGRESSIVE = "AGGRESSIVE"
    PASSIVE = "PASSIVE"
    TWAP = "TWAP"
    VWAP = "VWAP"
    POV = "POV"
    IS = "IS"
    ARRIVAL_PRICE = "ARRIVAL_PRICE"
    DARK_POOL = "DARK_POOL"
    SMART = "SMART"


@dataclass
class Order:
    """Order representation."""
    
    order_id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: TimeInForce = TimeInForce.DAY
    status: OrderStatus = OrderStatus.PENDING
    
    # Execution details
    filled_quantity: float = 0.0
    avg_fill_price: float = 0.0
    commission: float = 0.0
    slippage: float = 0.0
    
    # Timestamps
    created_at: datetime = field(default_factory=datetime.now)
    submitted_at: Optional[datetime] = None
    filled_at: Optional[datetime] = None
    cancelled_at: Optional[datetime] = None
    
    # Advanced features
    parent_order_id: Optional[str] = None
    child_orders: List[str] = field(default_factory=list)
    execution_algo: Optional[ExecutionAlgo] = None
    algo_params: Dict[str, Any] = field(default_factory=dict)
    
    # Risk controls
    max_slippage: Optional[float] = None
    max_participation: Optional[float] = None
    urgency: float = 0.5  # 0 to 1
    
    # Metadata
    strategy_id: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_complete(self) -> bool:
        """Check if order is complete."""
        return self.status in [
            OrderStatus.FILLED,
            OrderStatus.CANCELLED,
            OrderStatus.REJECTED,
            OrderStatus.EXPIRED
        ]
    
    @property
    def remaining_quantity(self) -> float:
        """Get remaining quantity."""
        return self.quantity - self.filled_quantity
    
    @property
    def fill_ratio(self) -> float:
        """Get fill ratio."""
        return self.filled_quantity / self.quantity if self.quantity > 0 else 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "order_id": self.order_id,
            "symbol": self.symbol,
            "side": self.side.value,
            "order_type": self.order_type.value,
            "quantity": self.quantity,
            "price": self.price,
            "status": self.status.value,
            "filled_quantity": self.filled_quantity,
            "avg_fill_price": self.avg_fill_price,
            "created_at": self.created_at.isoformat()
        }


@dataclass
class Fill:
    """Order fill representation."""
    
    fill_id: str
    order_id: str
    symbol: str
    side: OrderSide
    quantity: float
    price: float
    commission: float
    timestamp: datetime
    venue: str
    liquidity_flag: str  # MAKER/TAKER/PASSIVE/AGGRESSIVE
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "fill_id": self.fill_id,
            "order_id": self.order_id,
            "symbol": self.symbol,
            "quantity": self.quantity,
            "price": self.price,
            "commission": self.commission,
            "timestamp": self.timestamp.isoformat()
        }


class BrokerConnector(ABC):
    """Abstract base class for broker connections."""
    
    @abstractmethod
    async def connect(self) -> bool:
        """Connect to broker."""
        pass
    
    @abstractmethod
    async def disconnect(self) -> None:
        """Disconnect from broker."""
        pass
    
    @abstractmethod
    async def submit_order(self, order: Order) -> bool:
        """Submit order to broker."""
        pass
    
    @abstractmethod
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel order."""
        pass
    
    @abstractmethod
    async def modify_order(self, order_id: str, modifications: Dict) -> bool:
        """Modify order."""
        pass
    
    @abstractmethod
    async def get_order_status(self, order_id: str) -> OrderStatus:
        """Get order status."""
        pass
    
    @abstractmethod
    async def get_positions(self) -> Dict[str, float]:
        """Get current positions."""
        pass
    
    @abstractmethod
    async def get_account_info(self) -> Dict[str, Any]:
        """Get account information."""
        pass


class SimulatedBroker(BrokerConnector):
    """Simulated broker for testing."""
    
    def __init__(self, latency_ms: float = 10, fill_ratio: float = 0.95):
        self.latency_ms = latency_ms
        self.fill_ratio = fill_ratio
        self.orders: Dict[str, Order] = {}
        self.positions: Dict[str, float] = defaultdict(float)
        self.cash = 1000000.0
        self.connected = False
    
    async def connect(self) -> bool:
        """Connect to broker."""
        await asyncio.sleep(self.latency_ms / 1000)
        self.connected = True
        logger.info("Connected to simulated broker")
        return True
    
    async def disconnect(self) -> None:
        """Disconnect from broker."""
        self.connected = False
        logger.info("Disconnected from simulated broker")
    
    async def submit_order(self, order: Order) -> bool:
        """Submit order."""
        await asyncio.sleep(self.latency_ms / 1000)
        
        if not self.connected:
            return False
        
        # Simulate order processing
        order.status = OrderStatus.SUBMITTED
        order.submitted_at = datetime.now()
        self.orders[order.order_id] = order
        
        # Simulate fill
        if np.random.random() < self.fill_ratio:
            fill_quantity = order.quantity * np.random.uniform(0.8, 1.0)
            fill_price = order.price or 100.0  # Simulated price
            
            order.filled_quantity = min(fill_quantity, order.quantity)
            order.avg_fill_price = fill_price
            order.status = OrderStatus.FILLED if order.filled_quantity == order.quantity else OrderStatus.PARTIAL
            order.filled_at = datetime.now()
            
            # Update positions
            multiplier = 1 if order.side == OrderSide.BUY else -1
            self.positions[order.symbol] += multiplier * order.filled_quantity
            
        return True
    
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel order."""
        await asyncio.sleep(self.latency_ms / 1000)
        
        if order_id in self.orders:
            order = self.orders[order_id]
            if not order.is_complete:
                order.status = OrderStatus.CANCELLED
                order.cancelled_at = datetime.now()
                return True
        
        return False
    
    async def modify_order(self, order_id: str, modifications: Dict) -> bool:
        """Modify order."""
        await asyncio.sleep(self.latency_ms / 1000)
        
        if order_id in self.orders:
            order = self.orders[order_id]
            for key, value in modifications.items():
                if hasattr(order, key):
                    setattr(order, key, value)
            return True
        
        return False
    
    async def get_order_status(self, order_id: str) -> OrderStatus:
        """Get order status."""
        if order_id in self.orders:
            return self.orders[order_id].status
        return OrderStatus.REJECTED
    
    async def get_positions(self) -> Dict[str, float]:
        """Get positions."""
        return dict(self.positions)
    
    async def get_account_info(self) -> Dict[str, Any]:
        """Get account info."""
        total_value = self.cash
        for symbol, quantity in self.positions.items():
            total_value += quantity * 100  # Simulated price
        
        return {
            "cash": self.cash,
            "positions": dict(self.positions),
            "total_value": total_value,
            "buying_power": self.cash * 2  # 2x leverage
        }


class OrderRouter:
    """Smart order router for optimal execution venue selection."""
    
    def __init__(self):
        self.venues: Dict[str, BrokerConnector] = {}
        self.venue_metrics: Dict[str, Dict] = defaultdict(lambda: {
            "fill_rate": 0.95,
            "avg_latency": 10,
            "avg_spread": 0.01,
            "liquidity_score": 0.8,
            "fee_rate": 0.001
        })
        self.routing_rules: List[Callable] = []
    
    def add_venue(self, name: str, connector: BrokerConnector) -> None:
        """Add execution venue."""
        self.venues[name] = connector
        logger.info(f"Added venue: {name}")
    
    def add_routing_rule(self, rule: Callable) -> None:
        """Add routing rule."""
        self.routing_rules.append(rule)
    
    async def route_order(self, order: Order) -> Tuple[str, BrokerConnector]:
        """Route order to optimal venue."""
        # Apply routing rules
        for rule in self.routing_rules:
            venue_name = rule(order, self.venue_metrics)
            if venue_name and venue_name in self.venues:
                return venue_name, self.venues[venue_name]
        
        # Default routing based on metrics
        best_venue = self._select_best_venue(order)
        return best_venue, self.venues[best_venue]
    
    def _select_best_venue(self, order: Order) -> str:
        """Select best venue based on metrics."""
        scores = {}
        
        for venue_name, metrics in self.venue_metrics.items():
            if venue_name not in self.venues:
                continue
            
            # Calculate composite score
            score = (
                metrics["fill_rate"] * 0.3 +
                (1 - metrics["avg_latency"] / 100) * 0.2 +
                (1 - metrics["avg_spread"]) * 0.2 +
                metrics["liquidity_score"] * 0.2 +
                (1 - metrics["fee_rate"]) * 0.1
            )
            
            # Adjust for order characteristics
            if order.urgency > 0.7:
                score *= (1 - metrics["avg_latency"] / 100)
            
            if order.quantity > 10000:  # Large order
                score *= metrics["liquidity_score"]
            
            scores[venue_name] = score
        
        return max(scores, key=scores.get)
    
    def update_venue_metrics(self, venue_name: str, fill: Fill) -> None:
        """Update venue metrics based on fill."""
        if venue_name in self.venue_metrics:
            metrics = self.venue_metrics[venue_name]
            
            # Update with exponential moving average
            alpha = 0.1
            metrics["fill_rate"] = alpha * 1.0 + (1 - alpha) * metrics["fill_rate"]
            # Update other metrics similarly


class ExecutionAlgorithm(ABC):
    """Abstract base class for execution algorithms."""
    
    @abstractmethod
    async def execute(self, order: Order, market_data: pd.DataFrame) -> List[Order]:
        """Execute order and return child orders."""
        pass
    
    @abstractmethod
    def update_params(self, market_conditions: Dict) -> None:
        """Update algorithm parameters based on market conditions."""
        pass


class TWAPAlgorithm(ExecutionAlgorithm):
    """Time-Weighted Average Price algorithm."""
    
    def __init__(self, num_slices: int = 10, randomize: bool = True):
        self.num_slices = num_slices
        self.randomize = randomize
    
    async def execute(self, order: Order, market_data: pd.DataFrame) -> List[Order]:
        """Execute TWAP algorithm."""
        child_orders = []
        
        # Calculate slice size
        base_slice_size = order.quantity / self.num_slices
        
        for i in range(self.num_slices):
            # Randomize size if configured
            if self.randomize:
                slice_size = base_slice_size * np.random.uniform(0.8, 1.2)
            else:
                slice_size = base_slice_size
            
            # Create child order
            child_order = Order(
                order_id=f"{order.order_id}_slice_{i}",
                symbol=order.symbol,
                side=order.side,
                order_type=OrderType.LIMIT,
                quantity=min(slice_size, order.remaining_quantity),
                price=self._calculate_limit_price(order, market_data),
                time_in_force=TimeInForce.IOC,
                parent_order_id=order.order_id,
                execution_algo=ExecutionAlgo.TWAP
            )
            
            child_orders.append(child_order)
            
            # Add delay between slices
            await asyncio.sleep(1)
        
        return child_orders
    
    def _calculate_limit_price(self, order: Order, market_data: pd.DataFrame) -> float:
        """Calculate limit price for slice."""
        if market_data.empty:
            return order.price or 100.0
        
        current_price = market_data['close'].iloc[-1]
        spread = market_data['high'].iloc[-1] - market_data['low'].iloc[-1]
        
        # Adjust for side
        if order.side == OrderSide.BUY:
            return current_price - spread * 0.1
        else:
            return current_price + spread * 0.1
    
    def update_params(self, market_conditions: Dict) -> None:
        """Update parameters."""
        if market_conditions.get("volatility", 0) > 0.3:
            self.num_slices = min(20, self.num_slices + 2)
        else:
            self.num_slices = max(5, self.num_slices - 1)


class VWAPAlgorithm(ExecutionAlgorithm):
    """Volume-Weighted Average Price algorithm."""
    
    def __init__(self, lookback_periods: int = 20, aggressiveness: float = 0.5):
        self.lookback_periods = lookback_periods
        self.aggressiveness = aggressiveness
    
    async def execute(self, order: Order, market_data: pd.DataFrame) -> List[Order]:
        """Execute VWAP algorithm."""
        if market_data.empty or 'volume' not in market_data.columns:
            # Fallback to TWAP
            twap = TWAPAlgorithm()
            return await twap.execute(order, market_data)
        
        # Calculate volume profile
        volume_profile = self._calculate_volume_profile(market_data)
        
        child_orders = []
        remaining_quantity = order.quantity
        
        for time_bucket, volume_pct in volume_profile.items():
            if remaining_quantity <= 0:
                break
            
            # Calculate order size based on volume profile
            slice_size = order.quantity * volume_pct
            slice_size = min(slice_size, remaining_quantity)
            
            # Adjust price based on aggressiveness
            limit_price = self._calculate_vwap_price(
                order, market_data, self.aggressiveness
            )
            
            child_order = Order(
                order_id=f"{order.order_id}_vwap_{time_bucket}",
                symbol=order.symbol,
                side=order.side,
                order_type=OrderType.LIMIT,
                quantity=slice_size,
                price=limit_price,
                time_in_force=TimeInForce.IOC,
                parent_order_id=order.order_id,
                execution_algo=ExecutionAlgo.VWAP
            )
            
            child_orders.append(child_order)
            remaining_quantity -= slice_size
            
            await asyncio.sleep(0.5)
        
        return child_orders
    
    def _calculate_volume_profile(self, market_data: pd.DataFrame) -> Dict[int, float]:
        """Calculate historical volume profile."""
        recent_data = market_data.tail(self.lookback_periods)
        
        # Group by hour of day
        recent_data['hour'] = pd.to_datetime(recent_data.index).hour
        volume_by_hour = recent_data.groupby('hour')['volume'].mean()
        total_volume = volume_by_hour.sum()
        
        # Convert to percentages
        volume_profile = {}
        for hour, volume in volume_by_hour.items():
            volume_profile[hour] = volume / total_volume if total_volume > 0 else 0.1
        
        return volume_profile
    
    def _calculate_vwap_price(self, order: Order, market_data: pd.DataFrame, 
                              aggressiveness: float) -> float:
        """Calculate VWAP-based limit price."""
        # Calculate VWAP
        typical_price = (market_data['high'] + market_data['low'] + market_data['close']) / 3
        vwap = (typical_price * market_data['volume']).sum() / market_data['volume'].sum()
        
        current_price = market_data['close'].iloc[-1]
        
        # Blend VWAP and current price based on aggressiveness
        target_price = vwap * (1 - aggressiveness) + current_price * aggressiveness
        
        # Adjust for side
        spread = market_data['high'].iloc[-1] - market_data['low'].iloc[-1]
        if order.side == OrderSide.BUY:
            return target_price - spread * (1 - aggressiveness) * 0.1
        else:
            return target_price + spread * (1 - aggressiveness) * 0.1
    
    def update_params(self, market_conditions: Dict) -> None:
        """Update parameters."""
        volume_ratio = market_conditions.get("volume_ratio", 1.0)
        
        if volume_ratio > 1.5:  # High volume
            self.aggressiveness = min(0.8, self.aggressiveness + 0.1)
        elif volume_ratio < 0.5:  # Low volume
            self.aggressiveness = max(0.2, self.aggressiveness - 0.1)


class ImplementationShortfallAlgorithm(ExecutionAlgorithm):
    """Implementation Shortfall (Arrival Price) algorithm."""
    
    def __init__(self, risk_aversion: float = 0.5, urgency: float = 0.5):
        self.risk_aversion = risk_aversion
        self.urgency = urgency
        self.arrival_price = None
    
    async def execute(self, order: Order, market_data: pd.DataFrame) -> List[Order]:
        """Execute IS algorithm."""
        if market_data.empty:
            return []
        
        # Capture arrival price
        if self.arrival_price is None:
            self.arrival_price = market_data['close'].iloc[-1]
        
        # Calculate optimal execution trajectory
        trajectory = self._calculate_trajectory(order, market_data)
        
        child_orders = []
        
        for i, (time_point, quantity, aggressiveness) in enumerate(trajectory):
            # Calculate limit price
            limit_price = self._calculate_is_price(
                order, market_data, aggressiveness
            )
            
            child_order = Order(
                order_id=f"{order.order_id}_is_{i}",
                symbol=order.symbol,
                side=order.side,
                order_type=OrderType.LIMIT if aggressiveness < 0.8 else OrderType.MARKET,
                quantity=quantity,
                price=limit_price if aggressiveness < 0.8 else None,
                time_in_force=TimeInForce.IOC,
                parent_order_id=order.order_id,
                execution_algo=ExecutionAlgo.IS
            )
            
            child_orders.append(child_order)
            
            await asyncio.sleep(time_point)
        
        return child_orders
    
    def _calculate_trajectory(self, order: Order, market_data: pd.DataFrame) -> List[Tuple[float, float, float]]:
        """Calculate optimal execution trajectory."""
        # Simplified Almgren-Chriss model
        total_time = 300  # 5 minutes
        num_intervals = 10
        
        # Calculate volatility
        returns = market_data['close'].pct_change().dropna()
        volatility = returns.std() if len(returns) > 0 else 0.01
        
        # Risk-adjusted trading rate
        kappa = np.sqrt(self.risk_aversion / volatility) if volatility > 0 else 1.0
        
        trajectory = []
        remaining = order.quantity
        
        for i in range(num_intervals):
            # Time point
            time_point = (total_time / num_intervals) * (1 - self.urgency * 0.5)
            
            # Quantity based on exponential decay
            decay_rate = kappa * (1 + self.urgency)
            quantity = remaining * (1 - np.exp(-decay_rate)) / num_intervals
            quantity = min(quantity, remaining)
            
            # Aggressiveness increases with urgency
            aggressiveness = self.urgency * (1 + i / num_intervals)
            
            trajectory.append((time_point, quantity, aggressiveness))
            remaining -= quantity
        
        return trajectory
    
    def _calculate_is_price(self, order: Order, market_data: pd.DataFrame, 
                           aggressiveness: float) -> float:
        """Calculate IS-based limit price."""
        current_price = market_data['close'].iloc[-1]
        spread = market_data['high'].iloc[-1] - market_data['low'].iloc[-1]
        
        # Calculate expected slippage
        slippage = spread * aggressiveness * 0.5
        
        if order.side == OrderSide.BUY:
            return current_price + slippage
        else:
            return current_price - slippage
    
    def update_params(self, market_conditions: Dict) -> None:
        """Update parameters."""
        if market_conditions.get("volatility", 0) > 0.3:
            self.risk_aversion = min(0.9, self.risk_aversion + 0.1)
        
        if market_conditions.get("spread", 0) > 0.002:
            self.urgency = max(0.2, self.urgency - 0.1)


class ExecutionEngine:
    """Main execution engine managing orders and algorithms."""
    
    def __init__(self):
        self.orders: Dict[str, Order] = {}
        self.fills: List[Fill] = []
        self.active_orders: SortedDict = SortedDict()  # Sorted by priority
        
        self.brokers: Dict[str, BrokerConnector] = {}
        self.router = OrderRouter()
        self.algorithms: Dict[ExecutionAlgo, ExecutionAlgorithm] = {
            ExecutionAlgo.TWAP: TWAPAlgorithm(),
            ExecutionAlgo.VWAP: VWAPAlgorithm(),
            ExecutionAlgo.IS: ImplementationShortfallAlgorithm()
        }
        
        self.order_queue: asyncio.Queue = asyncio.Queue()
        self.fill_callbacks: List[Callable] = []
        
        # Risk controls
        self.max_order_size: Dict[str, float] = defaultdict(lambda: 10000)
        self.max_daily_volume: Dict[str, float] = defaultdict(lambda: 100000)
        self.daily_volume: Dict[str, float] = defaultdict(float)
        
        # Performance tracking
        self.execution_metrics = {
            "total_orders": 0,
            "filled_orders": 0,
            "cancelled_orders": 0,
            "avg_fill_time": 0,
            "avg_slippage": 0,
            "total_commission": 0
        }
        
        self.running = False
        self.tasks: List[asyncio.Task] = []
    
    async def start(self) -> None:
        """Start execution engine."""
        logger.info("Starting execution engine")
        self.running = True
        
        # Connect to brokers
        for name, broker in self.brokers.items():
            await broker.connect()
            self.router.add_venue(name, broker)
        
        # Start background tasks
        self.tasks.append(asyncio.create_task(self._process_orders()))
        self.tasks.append(asyncio.create_task(self._monitor_orders()))
        self.tasks.append(asyncio.create_task(self._update_metrics()))
    
    async def stop(self) -> None:
        """Stop execution engine."""
        logger.info("Stopping execution engine")
        self.running = False
        
        # Cancel all pending orders
        for order_id, order in self.orders.items():
            if not order.is_complete:
                await self.cancel_order(order_id)
        
        # Stop background tasks
        for task in self.tasks:
            task.cancel()
        
        # Disconnect from brokers
        for broker in self.brokers.values():
            await broker.disconnect()
    
    def add_broker(self, name: str, connector: BrokerConnector) -> None:
        """Add broker connector."""
        self.brokers[name] = connector
        logger.info(f"Added broker: {name}")
    
    async def submit_order(self, order: Order) -> str:
        """Submit order for execution."""
        # Validate order
        if not self._validate_order(order):
            order.status = OrderStatus.REJECTED
            logger.warning(f"Order {order.order_id} rejected by validation")
            return order.order_id
        
        # Generate order ID if not provided
        if not order.order_id:
            order.order_id = self._generate_order_id(order)
        
        # Store order
        self.orders[order.order_id] = order
        self.execution_metrics["total_orders"] += 1
        
        # Queue for processing
        await self.order_queue.put(order)
        
        logger.info(f"Order {order.order_id} submitted: {order.symbol} {order.side.value} {order.quantity}")
        
        return order.order_id
    
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel order."""
        if order_id not in self.orders:
            return False
        
        order = self.orders[order_id]
        
        if order.is_complete:
            return False
        
        # Cancel at broker
        venue, broker = await self.router.route_order(order)
        success = await broker.cancel_order(order_id)
        
        if success:
            order.status = OrderStatus.CANCELLED
            order.cancelled_at = datetime.now()
            self.execution_metrics["cancelled_orders"] += 1
            logger.info(f"Order {order_id} cancelled")
        
        return success
    
    async def modify_order(self, order_id: str, modifications: Dict) -> bool:
        """Modify order."""
        if order_id not in self.orders:
            return False
        
        order = self.orders[order_id]
        
        if order.is_complete:
            return False
        
        # Modify at broker
        venue, broker = await self.router.route_order(order)
        success = await broker.modify_order(order_id, modifications)
        
        if success:
            for key, value in modifications.items():
                if hasattr(order, key):
                    setattr(order, key, value)
            logger.info(f"Order {order_id} modified")
        
        return success
    
    def get_order(self, order_id: str) -> Optional[Order]:
        """Get order by ID."""
        return self.orders.get(order_id)
    
    def get_active_orders(self) -> List[Order]:
        """Get all active orders."""
        return [order for order in self.orders.values() if not order.is_complete]
    
    def get_fills(self, order_id: Optional[str] = None) -> List[Fill]:
        """Get fills, optionally filtered by order ID."""
        if order_id:
            return [fill for fill in self.fills if fill.order_id == order_id]
        return self.fills
    
    def add_fill_callback(self, callback: Callable) -> None:
        """Add fill callback."""
        self.fill_callbacks.append(callback)
    
    async def _process_orders(self) -> None:
        """Process order queue."""
        while self.running:
            try:
                # Get order from queue
                order = await asyncio.wait_for(
                    self.order_queue.get(),
                    timeout=1.0
                )
                
                # Select execution algorithm
                if order.execution_algo:
                    await self._execute_with_algo(order)
                else:
                    await self._execute_direct(order)
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error processing order: {e}")
    
    async def _execute_with_algo(self, order: Order) -> None:
        """Execute order using algorithm."""
        if order.execution_algo not in self.algorithms:
            logger.warning(f"Unknown algorithm: {order.execution_algo}")
            await self._execute_direct(order)
            return
        
        algorithm = self.algorithms[order.execution_algo]
        
        # Get market data (simplified)
        market_data = pd.DataFrame()  # Would fetch actual data
        
        # Generate child orders
        child_orders = await algorithm.execute(order, market_data)
        
        # Submit child orders
        for child_order in child_orders:
            order.child_orders.append(child_order.order_id)
            await self.submit_order(child_order)
    
    async def _execute_direct(self, order: Order) -> None:
        """Execute order directly."""
        # Route to best venue
        venue, broker = await self.router.route_order(order)
        
        # Submit to broker
        success = await broker.submit_order(order)
        
        if success:
            order.status = OrderStatus.SUBMITTED
            order.submitted_at = datetime.now()
            
            # Simulate fill for demo
            await self._simulate_fill(order, venue)
    
    async def _simulate_fill(self, order: Order, venue: str) -> None:
        """Simulate order fill (for demo)."""
        await asyncio.sleep(np.random.uniform(0.1, 1.0))
        
        # Create fill
        fill = Fill(
            fill_id=f"fill_{int(time.time() * 1000000)}",
            order_id=order.order_id,
            symbol=order.symbol,
            side=order.side,
            quantity=order.quantity,
            price=order.price or 100.0,
            commission=order.quantity * 0.001,
            timestamp=datetime.now(),
            venue=venue,
            liquidity_flag="TAKER"
        )
        
        # Update order
        order.filled_quantity = fill.quantity
        order.avg_fill_price = fill.price
        order.commission = fill.commission
        order.status = OrderStatus.FILLED
        order.filled_at = fill.timestamp
        
        # Store fill
        self.fills.append(fill)
        
        # Update metrics
        self.execution_metrics["filled_orders"] += 1
        self.execution_metrics["total_commission"] += fill.commission
        
        # Notify callbacks
        for callback in self.fill_callbacks:
            await callback(fill)
        
        logger.info(f"Order {order.order_id} filled: {fill.quantity} @ {fill.price}")
    
    async def _monitor_orders(self) -> None:
        """Monitor active orders."""
        while self.running:
            try:
                for order in self.get_active_orders():
                    # Check for timeout
                    if order.time_in_force == TimeInForce.DAY:
                        if datetime.now() - order.created_at > timedelta(hours=8):
                            order.status = OrderStatus.EXPIRED
                            logger.info(f"Order {order.order_id} expired")
                    
                    # Update order status from broker
                    if order.status == OrderStatus.SUBMITTED:
                        venue, broker = await self.router.route_order(order)
                        status = await broker.get_order_status(order.order_id)
                        
                        if status != order.status:
                            order.status = status
                            logger.info(f"Order {order.order_id} status updated: {status.value}")
                
                await asyncio.sleep(5)
                
            except Exception as e:
                logger.error(f"Error monitoring orders: {e}")
    
    async def _update_metrics(self) -> None:
        """Update execution metrics."""
        while self.running:
            try:
                # Calculate average fill time
                fill_times = []
                for order in self.orders.values():
                    if order.filled_at and order.submitted_at:
                        fill_time = (order.filled_at - order.submitted_at).total_seconds()
                        fill_times.append(fill_time)
                
                if fill_times:
                    self.execution_metrics["avg_fill_time"] = np.mean(fill_times)
                
                # Calculate average slippage
                slippages = []
                for order in self.orders.values():
                    if order.slippage:
                        slippages.append(abs(order.slippage))
                
                if slippages:
                    self.execution_metrics["avg_slippage"] = np.mean(slippages)
                
                # Update router metrics based on fills
                for fill in self.fills[-100:]:  # Last 100 fills
                    # Would update venue metrics here
                    pass
                
                await asyncio.sleep(30)
                
            except Exception as e:
                logger.error(f"Error updating metrics: {e}")
    
    def _validate_order(self, order: Order) -> bool:
        """Validate order against risk controls."""
        # Check order size
        if order.quantity > self.max_order_size[order.symbol]:
            logger.warning(f"Order size exceeds limit: {order.quantity} > {self.max_order_size[order.symbol]}")
            return False
        
        # Check daily volume
        if self.daily_volume[order.symbol] + order.quantity > self.max_daily_volume[order.symbol]:
            logger.warning(f"Daily volume limit exceeded for {order.symbol}")
            return False
        
        # Check price limits
        if order.order_type == OrderType.LIMIT and order.price:
            if order.price <= 0:
                logger.warning(f"Invalid limit price: {order.price}")
                return False
        
        return True
    
    def _generate_order_id(self, order: Order) -> str:
        """Generate unique order ID."""
        timestamp = int(time.time() * 1000000)
        hash_input = f"{order.symbol}_{order.side.value}_{order.quantity}_{timestamp}"
        order_hash = hashlib.md5(hash_input.encode()).hexdigest()[:8]
        return f"ORD_{timestamp}_{order_hash}"
    
    def get_execution_report(self) -> Dict[str, Any]:
        """Get execution performance report."""
        total_orders = len(self.orders)
        
        if total_orders == 0:
            return {}
        
        filled_orders = [o for o in self.orders.values() if o.status == OrderStatus.FILLED]
        partial_orders = [o for o in self.orders.values() if o.status == OrderStatus.PARTIAL]
        cancelled_orders = [o for o in self.orders.values() if o.status == OrderStatus.CANCELLED]
        
        # Calculate statistics
        fill_rate = len(filled_orders) / total_orders if total_orders > 0 else 0
        
        # Slippage analysis
        slippages = []
        for order in filled_orders:
            if order.price and order.avg_fill_price:
                slippage = (order.avg_fill_price - order.price) / order.price
                if order.side == OrderSide.SELL:
                    slippage = -slippage
                slippages.append(slippage)
        
        avg_slippage = np.mean(slippages) if slippages else 0
        slippage_std = np.std(slippages) if slippages else 0
        
        # Commission analysis
        total_commission = sum(o.commission for o in self.orders.values())
        avg_commission = total_commission / total_orders if total_orders > 0 else 0
        
        # Timing analysis
        fill_times = []
        for order in filled_orders:
            if order.filled_at and order.created_at:
                fill_time = (order.filled_at - order.created_at).total_seconds()
                fill_times.append(fill_time)
        
        avg_fill_time = np.mean(fill_times) if fill_times else 0
        
        return {
            "summary": {
                "total_orders": total_orders,
                "filled_orders": len(filled_orders),
                "partial_orders": len(partial_orders),
                "cancelled_orders": len(cancelled_orders),
                "fill_rate": fill_rate
            },
            "slippage": {
                "average": avg_slippage,
                "std_dev": slippage_std,
                "min": min(slippages) if slippages else 0,
                "max": max(slippages) if slippages else 0
            },
            "commission": {
                "total": total_commission,
                "average": avg_commission
            },
            "timing": {
                "avg_fill_time_seconds": avg_fill_time,
                "min_fill_time": min(fill_times) if fill_times else 0,
                "max_fill_time": max(fill_times) if fill_times else 0
            },
            "metrics": self.execution_metrics
        }


# Example usage
async def main():
    """Example usage of execution engine."""
    # Create engine
    engine = ExecutionEngine()
    
    # Add simulated broker
    broker = SimulatedBroker()
    engine.add_broker("sim", broker)
    
    # Start engine
    await engine.start()
    
    # Create and submit orders
    order1 = Order(
        order_id="",
        symbol="AAPL",
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        quantity=1000,
        price=150.0,
        execution_algo=ExecutionAlgo.VWAP
    )
    
    order_id = await engine.submit_order(order1)
    print(f"Submitted order: {order_id}")
    
    # Wait for execution
    await asyncio.sleep(5)
    
    # Get execution report
    report = engine.get_execution_report()
    print(f"Execution report: {report}")
    
    # Stop engine
    await engine.stop()


if __name__ == "__main__":
    asyncio.run(main())
