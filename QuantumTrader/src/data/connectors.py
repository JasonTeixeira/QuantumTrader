"""
Market Data Connectors for QuantumTrader

Production-grade data connectors with caching, rate limiting, and error recovery.

Author: QuantumTrader Team
Date: 2024
"""

import asyncio
import hashlib
import json
import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from enum import Enum
from functools import lru_cache, wraps
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import aiohttp
import numpy as np
import pandas as pd
import requests
from diskcache import Cache
from ratelimit import limits, sleep_and_retry
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

import yfinance as yf
from alpha_vantage.timeseries import TimeSeries
from polygon import RESTClient as PolygonClient

from ..utils.logging import get_logger
from .models import MarketDataSource, MarketDataSchema

logger = get_logger()


class DataQuality(Enum):
    """Data quality levels."""
    
    RAW = "raw"
    CLEANED = "cleaned"
    ADJUSTED = "adjusted"
    VALIDATED = "validated"


class CacheConfig:
    """Cache configuration."""
    
    def __init__(
        self,
        cache_dir: Path = Path("cache/market_data"),
        max_size: int = 10 * 1024 * 1024 * 1024,  # 10GB
        ttl: int = 3600,  # 1 hour
        enable_compression: bool = True
    ):
        self.cache_dir = cache_dir
        self.max_size = max_size
        self.ttl = ttl
        self.enable_compression = enable_compression
        
        # Create cache directory
        cache_dir.mkdir(parents=True, exist_ok=True)


def retry_on_failure(max_retries: int = 3, delay: float = 1.0, backoff: float = 2.0):
    """Decorator for retrying failed API calls."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    wait_time = delay * (backoff ** attempt)
                    logger.warning(
                        f"Attempt {attempt + 1} failed for {func.__name__}: {e}. "
                        f"Retrying in {wait_time:.1f} seconds..."
                    )
                    time.sleep(wait_time)
            
            logger.error(f"All {max_retries} attempts failed for {func.__name__}")
            raise last_exception
        
        return wrapper
    return decorator


class BaseDataConnector(ABC):
    """Abstract base class for data connectors."""
    
    def __init__(self, cache_config: Optional[CacheConfig] = None):
        """
        Initialize data connector.
        
        Parameters
        ----------
        cache_config : Optional[CacheConfig]
            Cache configuration
        """
        self.cache_config = cache_config or CacheConfig()
        self.cache = Cache(
            str(self.cache_config.cache_dir),
            size_limit=self.cache_config.max_size
        )
        
        # Setup session with retry logic
        self.session = requests.Session()
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
    
    @abstractmethod
    def fetch_historical_data(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        interval: str = "1d"
    ) -> pd.DataFrame:
        """
        Fetch historical market data.
        
        Parameters
        ----------
        symbol : str
            Ticker symbol
        start_date : datetime
            Start date
        end_date : datetime
            End date
        interval : str
            Data interval (1m, 5m, 15m, 30m, 1h, 1d, 1wk, 1mo)
            
        Returns
        -------
        pd.DataFrame
            OHLCV data
        """
        pass
    
    @abstractmethod
    def fetch_real_time_data(self, symbol: str) -> Dict[str, Any]:
        """
        Fetch real-time market data.
        
        Parameters
        ----------
        symbol : str
            Ticker symbol
            
        Returns
        -------
        Dict[str, Any]
            Real-time data
        """
        pass
    
    def _get_cache_key(self, *args) -> str:
        """Generate cache key from arguments."""
        key_data = json.dumps(args, sort_keys=True, default=str)
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def _cache_get(self, key: str) -> Optional[Any]:
        """Get data from cache."""
        try:
            return self.cache.get(key)
        except Exception as e:
            logger.warning(f"Cache get error: {e}")
            return None
    
    def _cache_set(self, key: str, value: Any, expire: Optional[int] = None) -> None:
        """Set data in cache."""
        try:
            expire = expire or self.cache_config.ttl
            self.cache.set(key, value, expire=expire)
        except Exception as e:
            logger.warning(f"Cache set error: {e}")
    
    def validate_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate and clean market data.
        
        Parameters
        ----------
        df : pd.DataFrame
            Raw market data
            
        Returns
        -------
        pd.DataFrame
            Validated data
        """
        if df.empty:
            return df
        
        # Remove duplicates
        df = df[~df.index.duplicated(keep='first')]
        
        # Sort by index
        df = df.sort_index()
        
        # Check for required columns
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = set(required_columns) - set(df.columns)
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Validate OHLC relationships
        invalid_ohlc = (
            (df['high'] < df['low']) |
            (df['high'] < df['open']) |
            (df['high'] < df['close']) |
            (df['low'] > df['open']) |
            (df['low'] > df['close'])
        )
        
        if invalid_ohlc.any():
            logger.warning(f"Found {invalid_ohlc.sum()} rows with invalid OHLC relationships")
            df = df[~invalid_ohlc]
        
        # Handle missing values
        df = df.fillna(method='ffill').fillna(method='bfill')
        
        # Remove outliers (prices more than 10 std from mean)
        for col in ['open', 'high', 'low', 'close']:
            mean = df[col].mean()
            std = df[col].std()
            outliers = np.abs(df[col] - mean) > 10 * std
            if outliers.any():
                logger.warning(f"Removing {outliers.sum()} outliers from {col}")
                df.loc[outliers, col] = np.nan
                df[col] = df[col].fillna(method='ffill').fillna(method='bfill')
        
        # Ensure positive volumes
        df['volume'] = df['volume'].abs()
        
        return df


class YahooFinanceConnector(BaseDataConnector):
    """Yahoo Finance data connector."""
    
    def __init__(self, cache_config: Optional[CacheConfig] = None):
        """Initialize Yahoo Finance connector."""
        super().__init__(cache_config)
        self.source = MarketDataSource.YAHOO
        logger.info("Initialized Yahoo Finance connector")
    
    @retry_on_failure(max_retries=3)
    def fetch_historical_data(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        interval: str = "1d"
    ) -> pd.DataFrame:
        """Fetch historical data from Yahoo Finance."""
        # Check cache
        cache_key = self._get_cache_key(
            "yahoo_historical",
            symbol,
            start_date,
            end_date,
            interval
        )
        cached_data = self._cache_get(cache_key)
        if cached_data is not None:
            logger.info(f"Cache hit for {symbol}")
            return cached_data
        
        logger.info(f"Fetching {symbol} from Yahoo Finance: {start_date} to {end_date}")
        
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(
                start=start_date,
                end=end_date,
                interval=interval,
                auto_adjust=False,
                actions=True
            )
            
            if df.empty:
                raise ValueError(f"No data returned for {symbol}")
            
            # Rename columns to standard format
            df.columns = df.columns.str.lower()
            df.index.name = 'timestamp'
            
            # Add adjusted close if dividends/splits exist
            if 'dividends' in df.columns and 'stock splits' in df.columns:
                df['adj_close'] = df['close'].copy()
                
                # Apply adjustments
                cumulative_adjustment = 1.0
                for i in range(len(df) - 1, -1, -1):
                    if df['stock splits'].iloc[i] != 0:
                        cumulative_adjustment *= df['stock splits'].iloc[i]
                    df.loc[df.index[i], 'adj_close'] *= cumulative_adjustment
                    if df['dividends'].iloc[i] != 0:
                        dividend_adjustment = 1 - (df['dividends'].iloc[i] / df['close'].iloc[i])
                        cumulative_adjustment *= dividend_adjustment
            
            # Validate data
            df = self.validate_data(df)
            
            # Cache the result
            self._cache_set(cache_key, df)
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            raise
    
    @retry_on_failure(max_retries=3)
    def fetch_real_time_data(self, symbol: str) -> Dict[str, Any]:
        """Fetch real-time data from Yahoo Finance."""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            return {
                'symbol': symbol,
                'price': info.get('regularMarketPrice', info.get('currentPrice')),
                'bid': info.get('bid'),
                'ask': info.get('ask'),
                'bid_size': info.get('bidSize'),
                'ask_size': info.get('askSize'),
                'volume': info.get('regularMarketVolume'),
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Error fetching real-time data for {symbol}: {e}")
            raise
    
    def fetch_options_chain(self, symbol: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Fetch options chain data."""
        try:
            ticker = yf.Ticker(symbol)
            
            # Get expiration dates
            expirations = ticker.options
            
            if not expirations:
                return pd.DataFrame(), pd.DataFrame()
            
            # Get options for nearest expiration
            opt = ticker.option_chain(expirations[0])
            
            return opt.calls, opt.puts
            
        except Exception as e:
            logger.error(f"Error fetching options for {symbol}: {e}")
            return pd.DataFrame(), pd.DataFrame()


class AlphaVantageConnector(BaseDataConnector):
    """Alpha Vantage data connector."""
    
    def __init__(
        self,
        api_key: str,
        cache_config: Optional[CacheConfig] = None
    ):
        """
        Initialize Alpha Vantage connector.
        
        Parameters
        ----------
        api_key : str
            Alpha Vantage API key
        cache_config : Optional[CacheConfig]
            Cache configuration
        """
        super().__init__(cache_config)
        self.api_key = api_key
        self.ts = TimeSeries(key=api_key, output_format='pandas')
        self.source = MarketDataSource.ALPHA_VANTAGE
        self.rate_limit = 5  # calls per minute for free tier
        logger.info("Initialized Alpha Vantage connector")
    
    @sleep_and_retry
    @limits(calls=5, period=60)  # Rate limiting
    @retry_on_failure(max_retries=3)
    def fetch_historical_data(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        interval: str = "1d"
    ) -> pd.DataFrame:
        """Fetch historical data from Alpha Vantage."""
        # Check cache
        cache_key = self._get_cache_key(
            "alpha_vantage_historical",
            symbol,
            start_date,
            end_date,
            interval
        )
        cached_data = self._cache_get(cache_key)
        if cached_data is not None:
            logger.info(f"Cache hit for {symbol}")
            return cached_data
        
        logger.info(f"Fetching {symbol} from Alpha Vantage: {start_date} to {end_date}")
        
        try:
            # Map intervals
            if interval == "1d":
                data, meta_data = self.ts.get_daily_adjusted(symbol, outputsize='full')
            elif interval in ["1m", "5m", "15m", "30m", "60m"]:
                data, meta_data = self.ts.get_intraday(
                    symbol,
                    interval=interval.replace("m", "min"),
                    outputsize='full'
                )
            else:
                raise ValueError(f"Unsupported interval: {interval}")
            
            # Process data
            df = data.copy()
            df.index = pd.to_datetime(df.index)
            df.index.name = 'timestamp'
            
            # Rename columns
            column_mapping = {
                '1. open': 'open',
                '2. high': 'high',
                '3. low': 'low',
                '4. close': 'close',
                '5. adjusted close': 'adj_close',
                '5. volume': 'volume',
                '6. volume': 'volume',
                '7. dividend amount': 'dividend',
                '8. split coefficient': 'split'
            }
            
            df.rename(columns=column_mapping, inplace=True)
            
            # Filter by date range
            df = df[(df.index >= start_date) & (df.index <= end_date)]
            
            # Validate data
            df = self.validate_data(df)
            
            # Cache the result
            self._cache_set(cache_key, df)
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            raise
    
    @sleep_and_retry
    @limits(calls=5, period=60)
    def fetch_real_time_data(self, symbol: str) -> Dict[str, Any]:
        """Fetch real-time data from Alpha Vantage."""
        try:
            url = f"https://www.alphavantage.co/query"
            params = {
                'function': 'GLOBAL_QUOTE',
                'symbol': symbol,
                'apikey': self.api_key
            }
            
            response = self.session.get(url, params=params)
            response.raise_for_status()
            
            data = response.json()
            
            if 'Global Quote' not in data:
                raise ValueError(f"No real-time data for {symbol}")
            
            quote = data['Global Quote']
            
            return {
                'symbol': symbol,
                'price': float(quote.get('05. price', 0)),
                'open': float(quote.get('02. open', 0)),
                'high': float(quote.get('03. high', 0)),
                'low': float(quote.get('04. low', 0)),
                'volume': int(quote.get('06. volume', 0)),
                'previous_close': float(quote.get('08. previous close', 0)),
                'change': float(quote.get('09. change', 0)),
                'change_percent': quote.get('10. change percent', '0%'),
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Error fetching real-time data for {symbol}: {e}")
            raise


class PolygonConnector(BaseDataConnector):
    """Polygon.io data connector."""
    
    def __init__(
        self,
        api_key: str,
        cache_config: Optional[CacheConfig] = None
    ):
        """
        Initialize Polygon connector.
        
        Parameters
        ----------
        api_key : str
            Polygon API key
        cache_config : Optional[CacheConfig]
            Cache configuration
        """
        super().__init__(cache_config)
        self.api_key = api_key
        self.client = PolygonClient(api_key)
        self.source = MarketDataSource.POLYGON
        logger.info("Initialized Polygon connector")
    
    @retry_on_failure(max_retries=3)
    def fetch_historical_data(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        interval: str = "1d"
    ) -> pd.DataFrame:
        """Fetch historical data from Polygon."""
        # Check cache
        cache_key = self._get_cache_key(
            "polygon_historical",
            symbol,
            start_date,
            end_date,
            interval
        )
        cached_data = self._cache_get(cache_key)
        if cached_data is not None:
            logger.info(f"Cache hit for {symbol}")
            return cached_data
        
        logger.info(f"Fetching {symbol} from Polygon: {start_date} to {end_date}")
        
        try:
            # Map intervals to Polygon format
            interval_mapping = {
                "1m": (1, "minute"),
                "5m": (5, "minute"),
                "15m": (15, "minute"),
                "30m": (30, "minute"),
                "1h": (1, "hour"),
                "1d": (1, "day"),
                "1wk": (1, "week"),
                "1mo": (1, "month")
            }
            
            if interval not in interval_mapping:
                raise ValueError(f"Unsupported interval: {interval}")
            
            multiplier, timespan = interval_mapping[interval]
            
            # Fetch aggregates
            aggs = self.client.get_aggs(
                ticker=symbol,
                multiplier=multiplier,
                timespan=timespan,
                from_=start_date.strftime("%Y-%m-%d"),
                to=end_date.strftime("%Y-%m-%d"),
                limit=50000
            )
            
            if not aggs:
                raise ValueError(f"No data returned for {symbol}")
            
            # Convert to DataFrame
            data = []
            for agg in aggs:
                data.append({
                    'timestamp': pd.to_datetime(agg.timestamp, unit='ms'),
                    'open': agg.open,
                    'high': agg.high,
                    'low': agg.low,
                    'close': agg.close,
                    'volume': agg.volume,
                    'vwap': agg.vwap if hasattr(agg, 'vwap') else None,
                    'trades': agg.transactions if hasattr(agg, 'transactions') else None
                })
            
            df = pd.DataFrame(data)
            df.set_index('timestamp', inplace=True)
            df.sort_index(inplace=True)
            
            # Validate data
            df = self.validate_data(df)
            
            # Cache the result
            self._cache_set(cache_key, df)
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            raise
    
    def fetch_real_time_data(self, symbol: str) -> Dict[str, Any]:
        """Fetch real-time data from Polygon."""
        try:
            # Get last trade
            trade = self.client.get_last_trade(symbol)
            
            # Get last quote
            quote = self.client.get_last_quote(symbol)
            
            return {
                'symbol': symbol,
                'price': trade.price if trade else None,
                'size': trade.size if trade else None,
                'bid': quote.bid_price if quote else None,
                'ask': quote.ask_price if quote else None,
                'bid_size': quote.bid_size if quote else None,
                'ask_size': quote.ask_size if quote else None,
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Error fetching real-time data for {symbol}: {e}")
            raise
    
    async def stream_real_time_data(
        self,
        symbols: List[str],
        callback: callable
    ) -> None:
        """
        Stream real-time data using WebSocket.
        
        Parameters
        ----------
        symbols : List[str]
            List of symbols to stream
        callback : callable
            Callback function for data processing
        """
        # This would implement WebSocket streaming
        # Simplified for demonstration
        pass


class DataConnectorFactory:
    """Factory for creating data connectors."""
    
    _connectors = {}
    
    @classmethod
    def register_connector(
        cls,
        source: MarketDataSource,
        connector_class: type,
        **kwargs
    ) -> None:
        """Register a data connector."""
        cls._connectors[source] = (connector_class, kwargs)
    
    @classmethod
    def create_connector(
        cls,
        source: MarketDataSource,
        **override_kwargs
    ) -> BaseDataConnector:
        """Create a data connector instance."""
        if source not in cls._connectors:
            raise ValueError(f"Unknown data source: {source}")
        
        connector_class, default_kwargs = cls._connectors[source]
        kwargs = {**default_kwargs, **override_kwargs}
        
        return connector_class(**kwargs)


class MultiSourceDataConnector:
    """Connector that aggregates data from multiple sources."""
    
    def __init__(
        self,
        primary_source: MarketDataSource = MarketDataSource.YAHOO,
        fallback_sources: Optional[List[MarketDataSource]] = None,
        cache_config: Optional[CacheConfig] = None
    ):
        """
        Initialize multi-source connector.
        
        Parameters
        ----------
        primary_source : MarketDataSource
            Primary data source
        fallback_sources : Optional[List[MarketDataSource]]
            Fallback sources in priority order
        cache_config : Optional[CacheConfig]
            Cache configuration
        """
        self.primary_source = primary_source
        self.fallback_sources = fallback_sources or []
        self.cache_config = cache_config or CacheConfig()
        
        # Initialize connectors
        self.connectors = {}
        for source in [primary_source] + self.fallback_sources:
            try:
                self.connectors[source] = DataConnectorFactory.create_connector(
                    source,
                    cache_config=self.cache_config
                )
            except Exception as e:
                logger.warning(f"Failed to initialize {source}: {e}")
    
    def fetch_historical_data(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        interval: str = "1d"
    ) -> pd.DataFrame:
        """
        Fetch historical data with fallback support.
        
        Parameters
        ----------
        symbol : str
            Ticker symbol
        start_date : datetime
            Start date
        end_date : datetime
            End date
        interval : str
            Data interval
            
        Returns
        -------
        pd.DataFrame
            OHLCV data
        """
        sources_to_try = [self.primary_source] + self.fallback_sources
        
        for source in sources_to_try:
            if source not in self.connectors:
                continue
            
            try:
                logger.info(f"Trying {source} for {symbol}")
                data = self.connectors[source].fetch_historical_data(
                    symbol,
                    start_date,
                    end_date,
                    interval
                )
                
                if not data.empty:
                    logger.info(f"Successfully fetched {len(data)} rows from {source}")
                    return data
                    
            except Exception as e:
                logger.warning(f"Failed to fetch from {source}: {e}")
                continue
        
        raise ValueError(f"Failed to fetch data for {symbol} from any source")
    
    async def fetch_multiple_symbols_async(
        self,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime,
        interval: str = "1d"
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch data for multiple symbols asynchronously.
        
        Parameters
        ----------
        symbols : List[str]
            List of symbols
        start_date : datetime
            Start date
        end_date : datetime
            End date
        interval : str
            Data interval
            
        Returns
        -------
        Dict[str, pd.DataFrame]
            Dictionary of symbol to data
        """
        async def fetch_symbol(symbol):
            try:
                return symbol, self.fetch_historical_data(
                    symbol,
                    start_date,
                    end_date,
                    interval
                )
            except Exception as e:
                logger.error(f"Failed to fetch {symbol}: {e}")
                return symbol, pd.DataFrame()
        
        tasks = [fetch_symbol(symbol) for symbol in symbols]
        results = await asyncio.gather(*tasks)
        
        return dict(results)


# Register default connectors
DataConnectorFactory.register_connector(
    MarketDataSource.YAHOO,
    YahooFinanceConnector
)
