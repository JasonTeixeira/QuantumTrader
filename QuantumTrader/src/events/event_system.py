"""
Professional Event-Driven Architecture for QuantumTrader

Enterprise-grade event system with Kafka integration for real-time
signal generation, order management, and system coordination.

Author: QuantumTrader Team
Date: 2024
"""

import asyncio
import json
import pickle
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Union

import aiokafka
from aiokafka import AIOKafkaProducer, AIOKafkaConsumer
from confluent_kafka import Producer, Consumer, KafkaError
from confluent_kafka.admin import AdminClient, NewTopic
import redis
from redis.asyncio import Redis as AsyncRedis

from ..utils.logging import get_logger

logger = get_logger(__name__)


# Event Types
class EventType(Enum):
    """Event type enumeration."""
    # Market Events
    MARKET_DATA = "market_data"
    PRICE_UPDATE = "price_update"
    VOLUME_SPIKE = "volume_spike"
    
    # Trading Events
    SIGNAL_GENERATED = "signal_generated"
    ORDER_SUBMITTED = "order_submitted"
    ORDER_FILLED = "order_filled"
    ORDER_CANCELLED = "order_cancelled"
    ORDER_REJECTED = "order_rejected"
    
    # Position Events
    POSITION_OPENED = "position_opened"
    POSITION_CLOSED = "position_closed"
    POSITION_UPDATED = "position_updated"
    
    # Risk Events
    RISK_LIMIT_BREACH = "risk_limit_breach"
    DRAWDOWN_ALERT = "drawdown_alert"
    EXPOSURE_WARNING = "exposure_warning"
    
    # System Events
    STRATEGY_STARTED = "strategy_started"
    STRATEGY_STOPPED = "strategy_stopped"
    SYSTEM_ERROR = "system_error"
    HEARTBEAT = "heartbeat"


@dataclass
class Event:
    """Base event class."""
    
    event_id: str
    event_type: EventType
    timestamp: datetime
    source: str
    data: Dict[str, Any]
    metadata: Dict[str, Any] = None
    
    def to_json(self) -> str:
        """Convert event to JSON."""
        return json.dumps({
            'event_id': self.event_id,
            'event_type': self.event_type.value,
            'timestamp': self.timestamp.isoformat(),
            'source': self.source,
            'data': self.data,
            'metadata': self.metadata or {}
        })
    
    @classmethod
    def from_json(cls, json_str: str) -> 'Event':
        """Create event from JSON."""
        data = json.loads(json_str)
        return cls(
            event_id=data['event_id'],
            event_type=EventType(data['event_type']),
            timestamp=datetime.fromisoformat(data['timestamp']),
            source=data['source'],
            data=data['data'],
            metadata=data.get('metadata')
        )
    
    def to_bytes(self) -> bytes:
        """Serialize event to bytes."""
        return pickle.dumps(self)
    
    @classmethod
    def from_bytes(cls, data: bytes) -> 'Event':
        """Deserialize event from bytes."""
        return pickle.loads(data)


# Specialized Event Classes
@dataclass
class MarketDataEvent(Event):
    """Market data update event."""
    
    def __init__(self, symbol: str, price: float, volume: int, 
                 bid: float, ask: float, source: str):
        super().__init__(
            event_id=f"md_{symbol}_{int(time.time() * 1000000)}",
            event_type=EventType.MARKET_DATA,
            timestamp=datetime.now(),
            source=source,
            data={
                'symbol': symbol,
                'price': price,
                'volume': volume,
                'bid': bid,
                'ask': ask,
                'spread': ask - bid
            }
        )


@dataclass
class SignalEvent(Event):
    """Trading signal event."""
    
    def __init__(self, strategy_id: str, symbol: str, signal_type: str,
                 strength: float, price: float, metadata: Dict = None):
        super().__init__(
            event_id=f"sig_{strategy_id}_{int(time.time() * 1000000)}",
            event_type=EventType.SIGNAL_GENERATED,
            timestamp=datetime.now(),
            source=strategy_id,
            data={
                'symbol': symbol,
                'signal_type': signal_type,
                'strength': strength,
                'price': price,
                'timestamp': datetime.now().isoformat()
            },
            metadata=metadata
        )


@dataclass
class OrderEvent(Event):
    """Order event."""
    
    def __init__(self, order_id: str, symbol: str, side: str, 
                 quantity: float, price: Optional[float], 
                 status: str, source: str):
        event_type_map = {
            'SUBMITTED': EventType.ORDER_SUBMITTED,
            'FILLED': EventType.ORDER_FILLED,
            'CANCELLED': EventType.ORDER_CANCELLED,
            'REJECTED': EventType.ORDER_REJECTED
        }
        
        super().__init__(
            event_id=f"ord_{order_id}_{int(time.time() * 1000000)}",
            event_type=event_type_map.get(status, EventType.ORDER_SUBMITTED),
            timestamp=datetime.now(),
            source=source,
            data={
                'order_id': order_id,
                'symbol': symbol,
                'side': side,
                'quantity': quantity,
                'price': price,
                'status': status
            }
        )


@dataclass
class RiskEvent(Event):
    """Risk alert event."""
    
    def __init__(self, risk_type: str, severity: str, 
                 message: str, metrics: Dict, source: str):
        event_type_map = {
            'LIMIT_BREACH': EventType.RISK_LIMIT_BREACH,
            'DRAWDOWN': EventType.DRAWDOWN_ALERT,
            'EXPOSURE': EventType.EXPOSURE_WARNING
        }
        
        super().__init__(
            event_id=f"risk_{risk_type}_{int(time.time() * 1000000)}",
            event_type=event_type_map.get(risk_type, EventType.RISK_LIMIT_BREACH),
            timestamp=datetime.now(),
            source=source,
            data={
                'risk_type': risk_type,
                'severity': severity,
                'message': message,
                'metrics': metrics
            }
        )


# Event Handler Interface
class EventHandler(ABC):
    """Abstract event handler interface."""
    
    @abstractmethod
    async def handle(self, event: Event) -> None:
        """Handle event."""
        pass
    
    @abstractmethod
    def can_handle(self, event: Event) -> bool:
        """Check if handler can process event."""
        pass


# Event Bus Implementation
class EventBus:
    """In-memory event bus for local event distribution."""
    
    def __init__(self):
        self.handlers: Dict[EventType, List[EventHandler]] = defaultdict(list)
        self.subscribers: Dict[str, List[Callable]] = defaultdict(list)
        self.event_queue: asyncio.Queue = asyncio.Queue()
        self.running = False
        self._task = None
    
    def register_handler(self, event_type: EventType, handler: EventHandler) -> None:
        """Register event handler."""
        self.handlers[event_type].append(handler)
        logger.debug(f"Registered handler for {event_type.value}")
    
    def subscribe(self, event_type: str, callback: Callable) -> None:
        """Subscribe to event type."""
        self.subscribers[event_type].append(callback)
    
    async def publish(self, event: Event) -> None:
        """Publish event to bus."""
        await self.event_queue.put(event)
        logger.debug(f"Published event: {event.event_id}")
    
    async def start(self) -> None:
        """Start event bus."""
        self.running = True
        self._task = asyncio.create_task(self._process_events())
        logger.info("Event bus started")
    
    async def stop(self) -> None:
        """Stop event bus."""
        self.running = False
        if self._task:
            self._task.cancel()
        logger.info("Event bus stopped")
    
    async def _process_events(self) -> None:
        """Process events from queue."""
        while self.running:
            try:
                event = await asyncio.wait_for(
                    self.event_queue.get(), 
                    timeout=1.0
                )
                
                # Process handlers
                handlers = self.handlers.get(event.event_type, [])
                for handler in handlers:
                    if handler.can_handle(event):
                        asyncio.create_task(handler.handle(event))
                
                # Process subscribers
                callbacks = self.subscribers.get(event.event_type.value, [])
                for callback in callbacks:
                    asyncio.create_task(callback(event))
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error processing event: {e}")


# Kafka Event Stream
class KafkaEventStream:
    """Kafka-based event streaming for distributed systems."""
    
    def __init__(self, 
                 bootstrap_servers: str = 'localhost:9092',
                 group_id: str = 'quantumtrader'):
        self.bootstrap_servers = bootstrap_servers
        self.group_id = group_id
        self.producer = None
        self.consumer = None
        self.admin_client = None
        self.topics: Set[str] = set()
        self.running = False
        self._consumer_task = None
    
    async def initialize(self) -> None:
        """Initialize Kafka connections."""
        # Create producer
        self.producer = AIOKafkaProducer(
            bootstrap_servers=self.bootstrap_servers,
            value_serializer=lambda v: v.to_bytes() if isinstance(v, Event) else pickle.dumps(v),
            compression_type='gzip',
            acks='all',
            max_batch_size=32768,
            linger_ms=10
        )
        await self.producer.start()
        
        # Create consumer
        self.consumer = AIOKafkaConsumer(
            bootstrap_servers=self.bootstrap_servers,
            group_id=self.group_id,
            value_deserializer=lambda v: Event.from_bytes(v) if v else None,
            auto_offset_reset='latest',
            enable_auto_commit=True,
            max_poll_records=500
        )
        
        # Create admin client
        self.admin_client = AdminClient({
            'bootstrap.servers': self.bootstrap_servers
        })
        
        # Create default topics
        await self._create_topics([
            'market_data',
            'signals',
            'orders',
            'positions',
            'risk_alerts',
            'system_events'
        ])
        
        logger.info("Kafka event stream initialized")
    
    async def _create_topics(self, topic_names: List[str]) -> None:
        """Create Kafka topics."""
        topics = []
        for name in topic_names:
            topics.append(NewTopic(
                topic=name,
                num_partitions=3,
                replication_factor=1,
                config={
                    'retention.ms': '86400000',  # 1 day
                    'compression.type': 'gzip',
                    'segment.ms': '3600000'  # 1 hour
                }
            ))
            self.topics.add(name)
        
        # Create topics
        fs = self.admin_client.create_topics(topics)
        
        for topic, f in fs.items():
            try:
                f.result()
                logger.info(f"Topic {topic} created")
            except Exception as e:
                logger.debug(f"Topic {topic} already exists: {e}")
    
    async def publish_event(self, event: Event, topic: Optional[str] = None) -> None:
        """Publish event to Kafka."""
        if not self.producer:
            await self.initialize()
        
        # Determine topic
        if not topic:
            topic = self._get_topic_for_event(event)
        
        # Send to Kafka
        await self.producer.send(
            topic=topic,
            key=event.source.encode('utf-8'),
            value=event
        )
        
        logger.debug(f"Published event {event.event_id} to topic {topic}")
    
    async def subscribe(self, topics: List[str], handler: Callable) -> None:
        """Subscribe to Kafka topics."""
        if not self.consumer:
            await self.initialize()
        
        # Subscribe to topics
        self.consumer.subscribe(topics)
        
        # Start consumer if not running
        if not self.running:
            await self.start_consumer(handler)
    
    async def start_consumer(self, handler: Callable) -> None:
        """Start consuming events."""
        self.running = True
        self._consumer_task = asyncio.create_task(
            self._consume_events(handler)
        )
    
    async def _consume_events(self, handler: Callable) -> None:
        """Consume events from Kafka."""
        await self.consumer.start()
        
        try:
            while self.running:
                try:
                    # Fetch messages
                    messages = await self.consumer.getmany(
                        timeout_ms=1000,
                        max_records=100
                    )
                    
                    for topic_partition, records in messages.items():
                        for record in records:
                            if record.value:
                                asyncio.create_task(handler(record.value))
                    
                except Exception as e:
                    logger.error(f"Error consuming events: {e}")
                    await asyncio.sleep(1)
        
        finally:
            await self.consumer.stop()
    
    def _get_topic_for_event(self, event: Event) -> str:
        """Get Kafka topic for event type."""
        topic_map = {
            EventType.MARKET_DATA: 'market_data',
            EventType.PRICE_UPDATE: 'market_data',
            EventType.SIGNAL_GENERATED: 'signals',
            EventType.ORDER_SUBMITTED: 'orders',
            EventType.ORDER_FILLED: 'orders',
            EventType.POSITION_OPENED: 'positions',
            EventType.POSITION_CLOSED: 'positions',
            EventType.RISK_LIMIT_BREACH: 'risk_alerts',
            EventType.SYSTEM_ERROR: 'system_events'
        }
        
        return topic_map.get(event.event_type, 'system_events')
    
    async def stop(self) -> None:
        """Stop Kafka connections."""
        self.running = False
        
        if self._consumer_task:
            self._consumer_task.cancel()
        
        if self.producer:
            await self.producer.stop()
        
        logger.info("Kafka event stream stopped")


# Redis Event Cache
class RedisEventCache:
    """Redis-based event caching and pub/sub."""
    
    def __init__(self, 
                 host: str = 'localhost',
                 port: int = 6379,
                 db: int = 0):
        self.host = host
        self.port = port
        self.db = db
        self.redis_client = None
        self.pubsub = None
        self.subscriptions: Dict[str, List[Callable]] = defaultdict(list)
        self._subscription_task = None
    
    async def connect(self) -> None:
        """Connect to Redis."""
        self.redis_client = await AsyncRedis.from_url(
            f"redis://{self.host}:{self.port}/{self.db}",
            decode_responses=True
        )
        self.pubsub = self.redis_client.pubsub()
        logger.info("Connected to Redis event cache")
    
    async def cache_event(self, event: Event, ttl: int = 3600) -> None:
        """Cache event in Redis."""
        if not self.redis_client:
            await self.connect()
        
        # Store event
        key = f"event:{event.event_type.value}:{event.event_id}"
        await self.redis_client.setex(
            key,
            ttl,
            event.to_json()
        )
        
        # Add to sorted set for time-based queries
        await self.redis_client.zadd(
            f"events:{event.event_type.value}",
            {event.event_id: event.timestamp.timestamp()}
        )
    
    async def get_recent_events(self, 
                                event_type: EventType,
                                limit: int = 100) -> List[Event]:
        """Get recent events of a type."""
        if not self.redis_client:
            await self.connect()
        
        # Get event IDs from sorted set
        event_ids = await self.redis_client.zrevrange(
            f"events:{event_type.value}",
            0,
            limit - 1
        )
        
        # Fetch events
        events = []
        for event_id in event_ids:
            key = f"event:{event_type.value}:{event_id}"
            data = await self.redis_client.get(key)
            if data:
                events.append(Event.from_json(data))
        
        return events
    
    async def publish_event(self, channel: str, event: Event) -> None:
        """Publish event to Redis channel."""
        if not self.redis_client:
            await self.connect()
        
        await self.redis_client.publish(channel, event.to_json())
    
    async def subscribe(self, channel: str, handler: Callable) -> None:
        """Subscribe to Redis channel."""
        if not self.redis_client:
            await self.connect()
        
        # Subscribe to channel
        await self.pubsub.subscribe(channel)
        self.subscriptions[channel].append(handler)
        
        # Start subscription handler if not running
        if not self._subscription_task:
            self._subscription_task = asyncio.create_task(
                self._handle_subscriptions()
            )
    
    async def _handle_subscriptions(self) -> None:
        """Handle Redis pub/sub messages."""
        async for message in self.pubsub.listen():
            if message['type'] == 'message':
                channel = message['channel']
                handlers = self.subscriptions.get(channel, [])
                
                # Parse event
                try:
                    event = Event.from_json(message['data'])
                    for handler in handlers:
                        asyncio.create_task(handler(event))
                except Exception as e:
                    logger.error(f"Error handling subscription: {e}")
    
    async def close(self) -> None:
        """Close Redis connections."""
        if self._subscription_task:
            self._subscription_task.cancel()
        
        if self.pubsub:
            await self.pubsub.close()
        
        if self.redis_client:
            await self.redis_client.close()
        
        logger.info("Redis event cache closed")


# Event Aggregator
class EventAggregator:
    """Aggregate and analyze events."""
    
    def __init__(self, window_size: int = 60):
        self.window_size = window_size  # seconds
        self.event_buffer: Dict[EventType, deque] = defaultdict(
            lambda: deque(maxlen=1000)
        )
        self.metrics: Dict[str, Any] = {}
    
    def add_event(self, event: Event) -> None:
        """Add event to aggregator."""
        self.event_buffer[event.event_type].append(event)
        self._update_metrics(event.event_type)
    
    def _update_metrics(self, event_type: EventType) -> None:
        """Update metrics for event type."""
        events = list(self.event_buffer[event_type])
        if not events:
            return
        
        # Calculate metrics
        now = datetime.now()
        recent_events = [
            e for e in events 
            if (now - e.timestamp).total_seconds() < self.window_size
        ]
        
        self.metrics[event_type.value] = {
            'count': len(recent_events),
            'rate': len(recent_events) / self.window_size,
            'latest': events[-1].timestamp if events else None
        }
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get aggregated metrics."""
        return self.metrics
    
    def get_event_rate(self, event_type: EventType) -> float:
        """Get event rate per second."""
        metrics = self.metrics.get(event_type.value, {})
        return metrics.get('rate', 0.0)


# Integrated Event System
class EventSystem:
    """Integrated event system combining all components."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # Core components
        self.event_bus = EventBus()
        self.kafka_stream = None
        self.redis_cache = None
        self.aggregator = EventAggregator()
        
        # Configuration
        self.use_kafka = self.config.get('use_kafka', False)
        self.use_redis = self.config.get('use_redis', True)
    
    async def initialize(self) -> None:
        """Initialize event system."""
        # Start event bus
        await self.event_bus.start()
        
        # Initialize Kafka if configured
        if self.use_kafka:
            self.kafka_stream = KafkaEventStream(
                bootstrap_servers=self.config.get('kafka_servers', 'localhost:9092')
            )
            await self.kafka_stream.initialize()
        
        # Initialize Redis if configured
        if self.use_redis:
            self.redis_cache = RedisEventCache(
                host=self.config.get('redis_host', 'localhost'),
                port=self.config.get('redis_port', 6379)
            )
            await self.redis_cache.connect()
        
        logger.info("Event system initialized")
    
    async def publish(self, event: Event) -> None:
        """Publish event to all channels."""
        # Add to aggregator
        self.aggregator.add_event(event)
        
        # Publish to event bus
        await self.event_bus.publish(event)
        
        # Publish to Kafka
        if self.kafka_stream:
            await self.kafka_stream.publish_event(event)
        
        # Cache in Redis
        if self.redis_cache:
            await self.redis_cache.cache_event(event)
            
            # Publish to Redis pub/sub
            channel = f"events:{event.event_type.value}"
            await self.redis_cache.publish_event(channel, event)
    
    def register_handler(self, event_type: EventType, handler: EventHandler) -> None:
        """Register event handler."""
        self.event_bus.register_handler(event_type, handler)
    
    async def subscribe(self, event_types: List[EventType], handler: Callable) -> None:
        """Subscribe to event types."""
        # Subscribe to event bus
        for event_type in event_types:
            self.event_bus.subscribe(event_type.value, handler)
        
        # Subscribe to Kafka topics
        if self.kafka_stream:
            topics = [self.kafka_stream._get_topic_for_event(
                Event("", et, datetime.now(), "", {})
            ) for et in event_types]
            await self.kafka_stream.subscribe(topics, handler)
        
        # Subscribe to Redis channels
        if self.redis_cache:
            for event_type in event_types:
                channel = f"events:{event_type.value}"
                await self.redis_cache.subscribe(channel, handler)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get system metrics."""
        return {
            'aggregator': self.aggregator.get_metrics(),
            'event_bus_queue_size': self.event_bus.event_queue.qsize()
        }
    
    async def shutdown(self) -> None:
        """Shutdown event system."""
        # Stop event bus
        await self.event_bus.stop()
        
        # Stop Kafka
        if self.kafka_stream:
            await self.kafka_stream.stop()
        
        # Close Redis
        if self.redis_cache:
            await self.redis_cache.close()
        
        logger.info("Event system shutdown complete")


# Example usage
async def main():
    """Example event system usage."""
    # Initialize event system
    event_system = EventSystem({
        'use_kafka': True,
        'use_redis': True,
        'kafka_servers': 'localhost:9092',
        'redis_host': 'localhost'
    })
    
    await event_system.initialize()
    
    # Define event handler
    async def handle_signal(event: Event):
        logger.info(f"Received signal: {event.data}")
    
    # Subscribe to signals
    await event_system.subscribe([EventType.SIGNAL_GENERATED], handle_signal)
    
    # Publish test event
    signal_event = SignalEvent(
        strategy_id="test_strategy",
        symbol="AAPL",
        signal_type="BUY",
        strength=0.8,
        price=150.0
    )
    
    await event_system.publish(signal_event)
    
    # Wait a bit
    await asyncio.sleep(2)
    
    # Get metrics
    metrics = event_system.get_metrics()
    logger.info(f"System metrics: {metrics}")
    
    # Shutdown
    await event_system.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
