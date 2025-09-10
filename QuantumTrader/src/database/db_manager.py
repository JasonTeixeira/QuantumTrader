"""
Professional Database Management Layer for QuantumTrader

TimescaleDB integration for high-performance time-series data storage,
with support for market data, trades, and performance metrics.

Author: QuantumTrader Team
Date: 2024
"""

import asyncio
import json
from contextlib import asynccontextmanager, contextmanager
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union

import asyncpg
import pandas as pd
import psycopg2
from psycopg2.extras import RealDictCursor, execute_batch
from sqlalchemy import (
    Boolean, Column, DateTime, Float, ForeignKey, Index, Integer,
    JSON, String, Text, UniqueConstraint, create_engine, text
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker, Session
from sqlalchemy.pool import QueuePool

from ..utils.logging import get_logger

logger = get_logger(__name__)

Base = declarative_base()


# Database Models
class MarketData(Base):
    """Market data time-series model."""
    __tablename__ = 'market_data'
    
    time = Column(DateTime, primary_key=True)
    symbol = Column(String(20), primary_key=True, index=True)
    open = Column(Float, nullable=False)
    high = Column(Float, nullable=False)
    low = Column(Float, nullable=False)
    close = Column(Float, nullable=False)
    volume = Column(Float, nullable=False)
    vwap = Column(Float)
    bid = Column(Float)
    ask = Column(Float)
    bid_size = Column(Integer)
    ask_size = Column(Integer)
    
    __table_args__ = (
        Index('idx_market_data_symbol_time', 'symbol', 'time'),
        {'timescaledb_hypertable': {'time_column_name': 'time'}}
    )


class Trade(Base):
    """Trade execution model."""
    __tablename__ = 'trades'
    
    trade_id = Column(String(50), primary_key=True)
    strategy_id = Column(String(50), nullable=False, index=True)
    symbol = Column(String(20), nullable=False, index=True)
    side = Column(String(10), nullable=False)
    quantity = Column(Float, nullable=False)
    entry_time = Column(DateTime, nullable=False, index=True)
    entry_price = Column(Float, nullable=False)
    exit_time = Column(DateTime, index=True)
    exit_price = Column(Float)
    commission = Column(Float, default=0)
    slippage = Column(Float, default=0)
    pnl = Column(Float)
    pnl_pct = Column(Float)
    status = Column(String(20), default='OPEN')
    metadata = Column(JSON)
    
    __table_args__ = (
        Index('idx_trades_strategy_time', 'strategy_id', 'entry_time'),
        {'timescaledb_hypertable': {'time_column_name': 'entry_time'}}
    )


class Position(Base):
    """Current position model."""
    __tablename__ = 'positions'
    
    position_id = Column(Integer, primary_key=True, autoincrement=True)
    strategy_id = Column(String(50), nullable=False, index=True)
    symbol = Column(String(20), nullable=False, index=True)
    quantity = Column(Float, nullable=False)
    avg_price = Column(Float, nullable=False)
    current_price = Column(Float)
    unrealized_pnl = Column(Float)
    realized_pnl = Column(Float, default=0)
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)
    
    __table_args__ = (
        UniqueConstraint('strategy_id', 'symbol', name='uq_strategy_symbol'),
    )


class StrategyPerformance(Base):
    """Strategy performance metrics model."""
    __tablename__ = 'strategy_performance'
    
    time = Column(DateTime, primary_key=True)
    strategy_id = Column(String(50), primary_key=True, index=True)
    total_return = Column(Float)
    daily_return = Column(Float)
    sharpe_ratio = Column(Float)
    max_drawdown = Column(Float)
    win_rate = Column(Float)
    profit_factor = Column(Float)
    total_trades = Column(Integer)
    portfolio_value = Column(Float)
    metrics = Column(JSON)
    
    __table_args__ = (
        Index('idx_performance_strategy_time', 'strategy_id', 'time'),
        {'timescaledb_hypertable': {'time_column_name': 'time'}}
    )


class Signal(Base):
    """Trading signal model."""
    __tablename__ = 'signals'
    
    signal_id = Column(Integer, primary_key=True, autoincrement=True)
    time = Column(DateTime, nullable=False, index=True)
    strategy_id = Column(String(50), nullable=False, index=True)
    symbol = Column(String(20), nullable=False, index=True)
    signal_type = Column(String(20), nullable=False)
    strength = Column(Float)
    price = Column(Float)
    metadata = Column(JSON)
    executed = Column(Boolean, default=False)
    
    __table_args__ = (
        Index('idx_signals_time', 'time'),
        Index('idx_signals_strategy_symbol', 'strategy_id', 'symbol'),
        {'timescaledb_hypertable': {'time_column_name': 'time'}}
    )


class OrderHistory(Base):
    """Order history model."""
    __tablename__ = 'order_history'
    
    order_id = Column(String(50), primary_key=True)
    time = Column(DateTime, nullable=False, index=True)
    strategy_id = Column(String(50), nullable=False, index=True)
    symbol = Column(String(20), nullable=False, index=True)
    side = Column(String(10), nullable=False)
    order_type = Column(String(20), nullable=False)
    quantity = Column(Float, nullable=False)
    price = Column(Float)
    filled_quantity = Column(Float, default=0)
    avg_fill_price = Column(Float)
    status = Column(String(20), nullable=False)
    venue = Column(String(50))
    metadata = Column(JSON)
    
    __table_args__ = (
        Index('idx_orders_time', 'time'),
        Index('idx_orders_strategy', 'strategy_id'),
        {'timescaledb_hypertable': {'time_column_name': 'time'}}
    )


class DatabaseManager:
    """Professional database management system."""
    
    def __init__(
        self,
        host: str = 'localhost',
        port: int = 5432,
        database: str = 'quantumtrader',
        user: str = 'postgres',
        password: str = 'postgres'
    ):
        self.host = host
        self.port = port
        self.database = database
        self.user = user
        self.password = password
        
        self.connection_string = (
            f"postgresql://{user}:{password}@{host}:{port}/{database}"
        )
        
        # SQLAlchemy engine with connection pooling
        self.engine = create_engine(
            self.connection_string,
            pool_size=20,
            max_overflow=30,
            pool_pre_ping=True,
            pool_recycle=3600,
            poolclass=QueuePool
        )
        
        # Session factory
        self.SessionLocal = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=self.engine
        )
        
        # Async connection pool
        self.async_pool = None
        
    def initialize_database(self) -> None:
        """Initialize database with TimescaleDB extensions."""
        try:
            # Create tables
            Base.metadata.create_all(bind=self.engine)
            
            with self.engine.connect() as conn:
                # Enable TimescaleDB extension
                conn.execute(text("CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;"))
                conn.commit()
                
                # Create hypertables for time-series data
                hypertables = [
                    ('market_data', 'time'),
                    ('trades', 'entry_time'),
                    ('strategy_performance', 'time'),
                    ('signals', 'time'),
                    ('order_history', 'time')
                ]
                
                for table, time_column in hypertables:
                    try:
                        conn.execute(text(
                            f"SELECT create_hypertable('{table}', '{time_column}', "
                            f"if_not_exists => TRUE, chunk_time_interval => INTERVAL '1 day');"
                        ))
                        conn.commit()
                    except Exception as e:
                        logger.warning(f"Hypertable {table} might already exist: {e}")
                
                # Create continuous aggregates for performance
                self._create_continuous_aggregates(conn)
                
                # Create indexes for better query performance
                self._create_indexes(conn)
                
            logger.info("Database initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise
    
    def _create_continuous_aggregates(self, conn) -> None:
        """Create continuous aggregates for faster queries."""
        # 1-hour market data aggregate
        try:
            conn.execute(text("""
                CREATE MATERIALIZED VIEW IF NOT EXISTS market_data_1h
                WITH (timescaledb.continuous) AS
                SELECT 
                    time_bucket('1 hour', time) AS bucket,
                    symbol,
                    FIRST(open, time) AS open,
                    MAX(high) AS high,
                    MIN(low) AS low,
                    LAST(close, time) AS close,
                    SUM(volume) AS volume
                FROM market_data
                GROUP BY bucket, symbol
                WITH NO DATA;
            """))
            
            # Daily performance aggregate
            conn.execute(text("""
                CREATE MATERIALIZED VIEW IF NOT EXISTS daily_performance
                WITH (timescaledb.continuous) AS
                SELECT 
                    time_bucket('1 day', time) AS day,
                    strategy_id,
                    LAST(total_return, time) AS total_return,
                    AVG(daily_return) AS avg_daily_return,
                    LAST(sharpe_ratio, time) AS sharpe_ratio,
                    MIN(max_drawdown) AS max_drawdown,
                    SUM(total_trades) AS total_trades
                FROM strategy_performance
                GROUP BY day, strategy_id
                WITH NO DATA;
            """))
            
            conn.commit()
            
        except Exception as e:
            logger.warning(f"Continuous aggregates might already exist: {e}")
    
    def _create_indexes(self, conn) -> None:
        """Create additional indexes for performance."""
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_market_data_time ON market_data(time DESC);",
            "CREATE INDEX IF NOT EXISTS idx_trades_pnl ON trades(pnl DESC);",
            "CREATE INDEX IF NOT EXISTS idx_positions_unrealized ON positions(unrealized_pnl DESC);",
            "CREATE INDEX IF NOT EXISTS idx_signals_executed ON signals(executed);",
        ]
        
        for index_sql in indexes:
            try:
                conn.execute(text(index_sql))
                conn.commit()
            except Exception as e:
                logger.warning(f"Index might already exist: {e}")
    
    @contextmanager
    def get_session(self) -> Session:
        """Get database session with context manager."""
        session = self.SessionLocal()
        try:
            yield session
        finally:
            session.close()
    
    async def initialize_async_pool(self) -> None:
        """Initialize async connection pool."""
        self.async_pool = await asyncpg.create_pool(
            host=self.host,
            port=self.port,
            database=self.database,
            user=self.user,
            password=self.password,
            min_size=5,
            max_size=20,
            command_timeout=60
        )
        logger.info("Async connection pool initialized")
    
    @asynccontextmanager
    async def get_async_connection(self):
        """Get async database connection."""
        if not self.async_pool:
            await self.initialize_async_pool()
        
        async with self.async_pool.acquire() as connection:
            yield connection
    
    # Market Data Operations
    async def insert_market_data(self, data: pd.DataFrame, symbol: str) -> None:
        """Insert market data efficiently."""
        if data.empty:
            return
        
        # Prepare data for insertion
        records = []
        for idx, row in data.iterrows():
            records.append({
                'time': idx,
                'symbol': symbol,
                'open': row.get('open', row.get('Open', 0)),
                'high': row.get('high', row.get('High', 0)),
                'low': row.get('low', row.get('Low', 0)),
                'close': row.get('close', row.get('Close', 0)),
                'volume': row.get('volume', row.get('Volume', 0)),
                'vwap': row.get('vwap', None),
                'bid': row.get('bid', None),
                'ask': row.get('ask', None)
            })
        
        async with self.get_async_connection() as conn:
            # Use COPY for bulk insert (fastest method)
            await conn.copy_records_to_table(
                'market_data',
                records=[(
                    r['time'], r['symbol'], r['open'], r['high'],
                    r['low'], r['close'], r['volume'], r['vwap'],
                    r['bid'], r['ask'], None, None
                ) for r in records]
            )
        
        logger.info(f"Inserted {len(records)} market data records for {symbol}")
    
    async def get_market_data(
        self,
        symbol: str,
        start_time: datetime,
        end_time: datetime,
        interval: str = '1m'
    ) -> pd.DataFrame:
        """Retrieve market data."""
        async with self.get_async_connection() as conn:
            # Use time_bucket for aggregation if needed
            if interval == '1m':
                query = """
                    SELECT time, open, high, low, close, volume
                    FROM market_data
                    WHERE symbol = $1 AND time >= $2 AND time <= $3
                    ORDER BY time
                """
            else:
                # Convert interval to PostgreSQL interval
                pg_interval = {
                    '5m': '5 minutes',
                    '15m': '15 minutes',
                    '1h': '1 hour',
                    '1d': '1 day'
                }.get(interval, '1 hour')
                
                query = f"""
                    SELECT 
                        time_bucket('{pg_interval}', time) AS time,
                        FIRST(open, time) AS open,
                        MAX(high) AS high,
                        MIN(low) AS low,
                        LAST(close, time) AS close,
                        SUM(volume) AS volume
                    FROM market_data
                    WHERE symbol = $1 AND time >= $2 AND time <= $3
                    GROUP BY time_bucket('{pg_interval}', time)
                    ORDER BY time
                """
            
            rows = await conn.fetch(query, symbol, start_time, end_time)
            
            if rows:
                df = pd.DataFrame(rows)
                df.set_index('time', inplace=True)
                return df
            
            return pd.DataFrame()
    
    # Trade Operations
    def insert_trade(self, trade: Dict[str, Any]) -> None:
        """Insert trade record."""
        with self.get_session() as session:
            trade_obj = Trade(**trade)
            session.add(trade_obj)
            session.commit()
            logger.info(f"Trade {trade['trade_id']} inserted")
    
    def update_trade(self, trade_id: str, updates: Dict[str, Any]) -> None:
        """Update trade record."""
        with self.get_session() as session:
            trade = session.query(Trade).filter_by(trade_id=trade_id).first()
            if trade:
                for key, value in updates.items():
                    setattr(trade, key, value)
                
                # Calculate P&L if closing trade
                if 'exit_price' in updates and trade.exit_price:
                    if trade.side == 'BUY':
                        trade.pnl = (trade.exit_price - trade.entry_price) * trade.quantity
                    else:
                        trade.pnl = (trade.entry_price - trade.exit_price) * trade.quantity
                    
                    trade.pnl_pct = trade.pnl / (trade.entry_price * trade.quantity)
                    trade.status = 'CLOSED'
                
                session.commit()
                logger.info(f"Trade {trade_id} updated")
    
    def get_trades(
        self,
        strategy_id: Optional[str] = None,
        symbol: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        status: Optional[str] = None
    ) -> pd.DataFrame:
        """Get trades with filters."""
        with self.get_session() as session:
            query = session.query(Trade)
            
            if strategy_id:
                query = query.filter(Trade.strategy_id == strategy_id)
            if symbol:
                query = query.filter(Trade.symbol == symbol)
            if start_time:
                query = query.filter(Trade.entry_time >= start_time)
            if end_time:
                query = query.filter(Trade.entry_time <= end_time)
            if status:
                query = query.filter(Trade.status == status)
            
            trades = query.all()
            
            if trades:
                data = [
                    {
                        'trade_id': t.trade_id,
                        'strategy_id': t.strategy_id,
                        'symbol': t.symbol,
                        'side': t.side,
                        'quantity': t.quantity,
                        'entry_time': t.entry_time,
                        'entry_price': t.entry_price,
                        'exit_time': t.exit_time,
                        'exit_price': t.exit_price,
                        'pnl': t.pnl,
                        'pnl_pct': t.pnl_pct,
                        'status': t.status
                    }
                    for t in trades
                ]
                return pd.DataFrame(data)
            
            return pd.DataFrame()
    
    # Position Operations
    def update_position(
        self,
        strategy_id: str,
        symbol: str,
        quantity: float,
        price: float
    ) -> None:
        """Update or create position."""
        with self.get_session() as session:
            position = session.query(Position).filter_by(
                strategy_id=strategy_id,
                symbol=symbol
            ).first()
            
            if position:
                # Update existing position
                total_value = position.quantity * position.avg_price + quantity * price
                position.quantity += quantity
                
                if position.quantity != 0:
                    position.avg_price = total_value / position.quantity
                else:
                    # Position closed
                    session.delete(position)
                    session.commit()
                    return
            else:
                # Create new position
                position = Position(
                    strategy_id=strategy_id,
                    symbol=symbol,
                    quantity=quantity,
                    avg_price=price,
                    current_price=price
                )
                session.add(position)
            
            position.updated_at = datetime.now()
            session.commit()
    
    def get_positions(
        self,
        strategy_id: Optional[str] = None
    ) -> pd.DataFrame:
        """Get current positions."""
        with self.get_session() as session:
            query = session.query(Position)
            
            if strategy_id:
                query = query.filter(Position.strategy_id == strategy_id)
            
            positions = query.all()
            
            if positions:
                data = [
                    {
                        'strategy_id': p.strategy_id,
                        'symbol': p.symbol,
                        'quantity': p.quantity,
                        'avg_price': p.avg_price,
                        'current_price': p.current_price,
                        'unrealized_pnl': p.unrealized_pnl,
                        'realized_pnl': p.realized_pnl
                    }
                    for p in positions
                ]
                return pd.DataFrame(data)
            
            return pd.DataFrame()
    
    async def update_position_prices(self, current_prices: Dict[str, float]) -> None:
        """Update position prices and calculate unrealized P&L."""
        async with self.get_async_connection() as conn:
            for symbol, price in current_prices.items():
                await conn.execute("""
                    UPDATE positions
                    SET current_price = $1,
                        unrealized_pnl = (quantity * ($1 - avg_price)),
                        updated_at = NOW()
                    WHERE symbol = $2
                """, price, symbol)
    
    # Performance Operations
    async def insert_performance_metrics(
        self,
        strategy_id: str,
        metrics: Dict[str, Any]
    ) -> None:
        """Insert performance metrics."""
        async with self.get_async_connection() as conn:
            await conn.execute("""
                INSERT INTO strategy_performance 
                (time, strategy_id, total_return, daily_return, sharpe_ratio,
                 max_drawdown, win_rate, profit_factor, total_trades, 
                 portfolio_value, metrics)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                ON CONFLICT (time, strategy_id) DO UPDATE
                SET total_return = EXCLUDED.total_return,
                    daily_return = EXCLUDED.daily_return,
                    sharpe_ratio = EXCLUDED.sharpe_ratio,
                    max_drawdown = EXCLUDED.max_drawdown,
                    win_rate = EXCLUDED.win_rate,
                    profit_factor = EXCLUDED.profit_factor,
                    total_trades = EXCLUDED.total_trades,
                    portfolio_value = EXCLUDED.portfolio_value,
                    metrics = EXCLUDED.metrics
            """,
                datetime.now(),
                strategy_id,
                metrics.get('total_return', 0),
                metrics.get('daily_return', 0),
                metrics.get('sharpe_ratio', 0),
                metrics.get('max_drawdown', 0),
                metrics.get('win_rate', 0),
                metrics.get('profit_factor', 0),
                metrics.get('total_trades', 0),
                metrics.get('portfolio_value', 0),
                json.dumps(metrics)
            )
    
    async def get_performance_history(
        self,
        strategy_id: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> pd.DataFrame:
        """Get performance history."""
        async with self.get_async_connection() as conn:
            query = """
                SELECT * FROM strategy_performance
                WHERE strategy_id = $1
            """
            params = [strategy_id]
            
            if start_time:
                query += f" AND time >= ${len(params) + 1}"
                params.append(start_time)
            
            if end_time:
                query += f" AND time <= ${len(params) + 1}"
                params.append(end_time)
            
            query += " ORDER BY time"
            
            rows = await conn.fetch(query, *params)
            
            if rows:
                df = pd.DataFrame(rows)
                df.set_index('time', inplace=True)
                return df
            
            return pd.DataFrame()
    
    # Signal Operations
    async def insert_signal(self, signal: Dict[str, Any]) -> None:
        """Insert trading signal."""
        async with self.get_async_connection() as conn:
            await conn.execute("""
                INSERT INTO signals 
                (time, strategy_id, symbol, signal_type, strength, price, metadata, executed)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            """,
                signal['time'],
                signal['strategy_id'],
                signal['symbol'],
                signal['signal_type'],
                signal.get('strength', 0),
                signal.get('price', 0),
                json.dumps(signal.get('metadata', {})),
                signal.get('executed', False)
            )
    
    async def get_latest_signals(
        self,
        strategy_id: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict]:
        """Get latest signals."""
        async with self.get_async_connection() as conn:
            query = """
                SELECT * FROM signals
            """
            params = []
            
            if strategy_id:
                query += " WHERE strategy_id = $1"
                params.append(strategy_id)
            
            query += f" ORDER BY time DESC LIMIT {limit}"
            
            rows = await conn.fetch(query, *params)
            
            return [dict(row) for row in rows]
    
    # Cleanup Operations
    async def cleanup_old_data(self, days_to_keep: int = 30) -> None:
        """Clean up old data to manage storage."""
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        
        async with self.get_async_connection() as conn:
            # Clean up old market data (keep aggregates)
            await conn.execute("""
                DELETE FROM market_data
                WHERE time < $1
            """, cutoff_date)
            
            # Archive old trades
            await conn.execute("""
                INSERT INTO trades_archive
                SELECT * FROM trades
                WHERE entry_time < $1 AND status = 'CLOSED'
            """, cutoff_date)
            
            await conn.execute("""
                DELETE FROM trades
                WHERE entry_time < $1 AND status = 'CLOSED'
            """, cutoff_date)
            
            logger.info(f"Cleaned up data older than {days_to_keep} days")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get database statistics."""
        with self.engine.connect() as conn:
            stats = {}
            
            # Table sizes
            result = conn.execute(text("""
                SELECT 
                    schemaname,
                    tablename,
                    pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) AS size
                FROM pg_tables
                WHERE schemaname = 'public'
                ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC;
            """))
            
            stats['table_sizes'] = [dict(row) for row in result]
            
            # Hypertable stats
            result = conn.execute(text("""
                SELECT 
                    hypertable_name,
                    num_chunks,
                    compressed_total_bytes,
                    uncompressed_total_bytes
                FROM timescaledb_information.hypertables
                WHERE hypertable_schema = 'public';
            """))
            
            stats['hypertables'] = [dict(row) for row in result]
            
            # Connection stats
            result = conn.execute(text("""
                SELECT 
                    count(*) as total_connections,
                    count(*) FILTER (WHERE state = 'active') as active_connections,
                    count(*) FILTER (WHERE state = 'idle') as idle_connections
                FROM pg_stat_activity
                WHERE datname = current_database();
            """))
            
            stats['connections'] = dict(result.fetchone())
            
            return stats
    
    def close(self) -> None:
        """Close database connections."""
        self.engine.dispose()
        if self.async_pool:
            asyncio.get_event_loop().run_until_complete(self.async_pool.close())
        logger.info("Database connections closed")


# Example usage
async def main():
    """Example database operations."""
    # Initialize database manager
    db = DatabaseManager()
    
    # Initialize database
    db.initialize_database()
    
    # Initialize async pool
    await db.initialize_async_pool()
    
    # Insert sample market data
    dates = pd.date_range(start='2024-01-01', end='2024-01-02', freq='1min')
    sample_data = pd.DataFrame({
        'open': np.random.randn(len(dates)) * 0.01 + 100,
        'high': np.random.randn(len(dates)) * 0.01 + 101,
        'low': np.random.randn(len(dates)) * 0.01 + 99,
        'close': np.random.randn(len(dates)) * 0.01 + 100,
        'volume': np.random.randint(1000, 10000, len(dates))
    }, index=dates)
    
    await db.insert_market_data(sample_data, 'AAPL')
    
    # Retrieve market data
    data = await db.get_market_data(
        'AAPL',
        datetime(2024, 1, 1),
        datetime(2024, 1, 2)
    )
    print(f"Retrieved {len(data)} market data records")
    
    # Insert performance metrics
    await db.insert_performance_metrics('strategy_001', {
        'total_return': 0.15,
        'sharpe_ratio': 1.5,
        'max_drawdown': -0.08,
        'win_rate': 0.65
    })
    
    # Get statistics
    stats = db.get_statistics()
    print(f"Database statistics: {stats}")
    
    # Close connections
    db.close()


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
