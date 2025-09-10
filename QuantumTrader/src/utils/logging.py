"""
Logging and Monitoring Framework for QuantumTrader

This module provides a comprehensive logging and monitoring framework
with support for structured logging, metrics collection, and distributed tracing.

Author: QuantumTrader Team
Date: 2024
"""

import json
import logging
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Union

import structlog
from loguru import logger
from prometheus_client import Counter, Histogram, Gauge, Summary, CollectorRegistry
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor


# Metrics Registry
metrics_registry = CollectorRegistry()

# Prometheus Metrics
TRADE_COUNTER = Counter(
    'trades_total',
    'Total number of trades executed',
    ['strategy', 'symbol', 'side'],
    registry=metrics_registry
)

SIGNAL_COUNTER = Counter(
    'signals_total',
    'Total number of signals generated',
    ['strategy', 'symbol', 'signal_type'],
    registry=metrics_registry
)

PNL_GAUGE = Gauge(
    'pnl_current',
    'Current PnL',
    ['strategy', 'symbol'],
    registry=metrics_registry
)

POSITION_GAUGE = Gauge(
    'positions_open',
    'Number of open positions',
    ['strategy'],
    registry=metrics_registry
)

LATENCY_HISTOGRAM = Histogram(
    'operation_latency_seconds',
    'Operation latency in seconds',
    ['operation', 'strategy'],
    registry=metrics_registry
)

ERROR_COUNTER = Counter(
    'errors_total',
    'Total number of errors',
    ['component', 'error_type'],
    registry=metrics_registry
)

API_REQUEST_DURATION = Summary(
    'api_request_duration_seconds',
    'API request duration in seconds',
    ['endpoint', 'method'],
    registry=metrics_registry
)


class LogLevel:
    """Log levels."""
    
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class LoggerConfig:
    """Logger configuration."""
    
    def __init__(
        self,
        name: str = "QuantumTrader",
        level: str = LogLevel.INFO,
        log_dir: Optional[Path] = None,
        enable_console: bool = True,
        enable_file: bool = True,
        enable_json: bool = True,
        enable_metrics: bool = True,
        enable_tracing: bool = False,
        max_file_size: str = "100 MB",
        retention: str = "30 days",
        compression: str = "zip"
    ):
        """
        Initialize logger configuration.
        
        Parameters
        ----------
        name : str
            Logger name
        level : str
            Log level
        log_dir : Optional[Path]
            Log directory path
        enable_console : bool
            Enable console logging
        enable_file : bool
            Enable file logging
        enable_json : bool
            Enable JSON structured logging
        enable_metrics : bool
            Enable metrics collection
        enable_tracing : bool
            Enable distributed tracing
        max_file_size : str
            Maximum log file size
        retention : str
            Log retention period
        compression : str
            Log compression format
        """
        self.name = name
        self.level = level
        self.log_dir = log_dir or Path("logs")
        self.enable_console = enable_console
        self.enable_file = enable_file
        self.enable_json = enable_json
        self.enable_metrics = enable_metrics
        self.enable_tracing = enable_tracing
        self.max_file_size = max_file_size
        self.retention = retention
        self.compression = compression


class TradingLogger:
    """Enhanced logging for trading systems."""
    
    def __init__(self, config: LoggerConfig):
        """
        Initialize trading logger.
        
        Parameters
        ----------
        config : LoggerConfig
            Logger configuration
        """
        self.config = config
        self._setup_logger()
        self._setup_structured_logging()
        
        if config.enable_tracing:
            self._setup_tracing()
    
    def _setup_logger(self):
        """Set up loguru logger."""
        # Remove default handler
        logger.remove()
        
        # Create log directory
        self.config.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Console handler
        if self.config.enable_console:
            logger.add(
                sys.stdout,
                level=self.config.level,
                format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
                       "<level>{level: <8}</level> | "
                       "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
                       "<level>{message}</level>",
                colorize=True
            )
        
        # File handler with rotation
        if self.config.enable_file:
            log_file = self.config.log_dir / f"{self.config.name}.log"
            logger.add(
                log_file,
                level=self.config.level,
                format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} | {message}",
                rotation=self.config.max_file_size,
                retention=self.config.retention,
                compression=self.config.compression,
                enqueue=True
            )
        
        # JSON handler for structured logging
        if self.config.enable_json:
            json_log_file = self.config.log_dir / f"{self.config.name}.json"
            logger.add(
                json_log_file,
                level=self.config.level,
                format=self._json_formatter,
                rotation=self.config.max_file_size,
                retention=self.config.retention,
                compression=self.config.compression,
                serialize=True,
                enqueue=True
            )
        
        # Error file handler
        error_log_file = self.config.log_dir / f"{self.config.name}_errors.log"
        logger.add(
            error_log_file,
            level=LogLevel.ERROR,
            format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} | {message}",
            rotation="50 MB",
            retention="60 days",
            compression="zip",
            backtrace=True,
            diagnose=True,
            enqueue=True
        )
        
        self.logger = logger
    
    def _json_formatter(self, record):
        """Format log record as JSON."""
        log_data = {
            "timestamp": record["time"].isoformat(),
            "level": record["level"].name,
            "logger": record["name"],
            "function": record["function"],
            "line": record["line"],
            "message": record["message"],
            "module": record["module"],
            "process": record["process"].id,
            "thread": record["thread"].id,
        }
        
        # Add extra fields
        if record["extra"]:
            log_data["extra"] = record["extra"]
        
        # Add exception info if present
        if record["exception"]:
            log_data["exception"] = {
                "type": record["exception"].type.__name__,
                "value": str(record["exception"].value),
                "traceback": "".join(traceback.format_tb(record["exception"].traceback))
            }
        
        return json.dumps(log_data)
    
    def _setup_structured_logging(self):
        """Set up structured logging with structlog."""
        structlog.configure(
            processors=[
                structlog.stdlib.filter_by_level,
                structlog.stdlib.add_logger_name,
                structlog.stdlib.add_log_level,
                structlog.stdlib.PositionalArgumentsFormatter(),
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                structlog.processors.UnicodeDecoder(),
                structlog.processors.JSONRenderer()
            ],
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            cache_logger_on_first_use=True,
        )
    
    def _setup_tracing(self):
        """Set up distributed tracing with OpenTelemetry."""
        resource = Resource.create({
            "service.name": self.config.name,
            "service.version": "1.0.0"
        })
        
        provider = TracerProvider(resource=resource)
        processor = BatchSpanProcessor(OTLPSpanExporter())
        provider.add_span_processor(processor)
        
        trace.set_tracer_provider(provider)
        self.tracer = trace.get_tracer(self.config.name)
    
    def log_trade(
        self,
        strategy: str,
        symbol: str,
        side: str,
        price: float,
        quantity: float,
        **kwargs
    ):
        """
        Log trade execution.
        
        Parameters
        ----------
        strategy : str
            Strategy name
        symbol : str
            Trading symbol
        side : str
            Trade side (buy/sell)
        price : float
            Execution price
        quantity : float
            Trade quantity
        **kwargs
            Additional trade information
        """
        self.logger.info(
            f"Trade executed: {side} {quantity} {symbol} @ {price}",
            strategy=strategy,
            symbol=symbol,
            side=side,
            price=price,
            quantity=quantity,
            **kwargs
        )
        
        if self.config.enable_metrics:
            TRADE_COUNTER.labels(strategy=strategy, symbol=symbol, side=side).inc()
    
    def log_signal(
        self,
        strategy: str,
        symbol: str,
        signal_type: str,
        strength: float,
        **kwargs
    ):
        """
        Log signal generation.
        
        Parameters
        ----------
        strategy : str
            Strategy name
        symbol : str
            Trading symbol
        signal_type : str
            Signal type
        strength : float
            Signal strength
        **kwargs
            Additional signal information
        """
        self.logger.info(
            f"Signal generated: {signal_type} for {symbol} (strength: {strength})",
            strategy=strategy,
            symbol=symbol,
            signal_type=signal_type,
            strength=strength,
            **kwargs
        )
        
        if self.config.enable_metrics:
            SIGNAL_COUNTER.labels(
                strategy=strategy,
                symbol=symbol,
                signal_type=signal_type
            ).inc()
    
    def log_error(
        self,
        component: str,
        error_type: str,
        message: str,
        **kwargs
    ):
        """
        Log error with metrics.
        
        Parameters
        ----------
        component : str
            Component name
        error_type : str
            Error type
        message : str
            Error message
        **kwargs
            Additional error information
        """
        self.logger.error(
            message,
            component=component,
            error_type=error_type,
            **kwargs
        )
        
        if self.config.enable_metrics:
            ERROR_COUNTER.labels(component=component, error_type=error_type).inc()
    
    def log_performance(
        self,
        strategy: str,
        metrics: Dict[str, float]
    ):
        """
        Log performance metrics.
        
        Parameters
        ----------
        strategy : str
            Strategy name
        metrics : Dict[str, float]
            Performance metrics
        """
        self.logger.info(
            f"Performance metrics for {strategy}",
            strategy=strategy,
            **metrics
        )
        
        if self.config.enable_metrics:
            for symbol_pnl in metrics.get("symbol_pnl", {}).items():
                symbol, pnl = symbol_pnl
                PNL_GAUGE.labels(strategy=strategy, symbol=symbol).set(pnl)
            
            if "open_positions" in metrics:
                POSITION_GAUGE.labels(strategy=strategy).set(metrics["open_positions"])
    
    def create_context_logger(self, **context):
        """
        Create a logger with context.
        
        Parameters
        ----------
        **context
            Context variables
            
        Returns
        -------
        Logger
            Logger with context
        """
        return self.logger.bind(**context)
    
    def measure_latency(self, operation: str, strategy: str):
        """
        Create a context manager for measuring operation latency.
        
        Parameters
        ----------
        operation : str
            Operation name
        strategy : str
            Strategy name
            
        Returns
        -------
        ContextManager
            Latency measurement context manager
        """
        if self.config.enable_metrics:
            return LATENCY_HISTOGRAM.labels(
                operation=operation,
                strategy=strategy
            ).time()
        else:
            from contextlib import nullcontext
            return nullcontext()


# Global logger instance
_global_logger: Optional[TradingLogger] = None


def setup_logger(config: Optional[LoggerConfig] = None) -> TradingLogger:
    """
    Set up global logger.
    
    Parameters
    ----------
    config : Optional[LoggerConfig]
        Logger configuration
        
    Returns
    -------
    TradingLogger
        Logger instance
    """
    global _global_logger
    
    if config is None:
        config = LoggerConfig()
    
    _global_logger = TradingLogger(config)
    return _global_logger


def get_logger() -> TradingLogger:
    """
    Get global logger instance.
    
    Returns
    -------
    TradingLogger
        Logger instance
    """
    global _global_logger
    
    if _global_logger is None:
        _global_logger = setup_logger()
    
    return _global_logger
