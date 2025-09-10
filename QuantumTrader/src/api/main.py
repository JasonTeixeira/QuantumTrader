"""
FastAPI Application for QuantumTrader

Production-grade REST API for strategy management, monitoring, and control.

Author: QuantumTrader Team
Date: 2024
"""

import asyncio
import json
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union

import pandas as pd
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, status, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field, validator
from redis import Redis
import uvicorn

from ..backtesting.engine import BacktestingEngine, BacktestConfig, BacktestResult
from ..strategies.mean_reversion import MeanReversionStrategy, MeanReversionConfig
from ..strategies.momentum import MomentumStrategy, MomentumConfig
from ..risk.risk_manager import RiskManager, RiskLimits, PortfolioRiskMetrics
from ..data.connectors import MultiSourceDataConnector, MarketDataSource
from ..utils.logging import get_logger

logger = get_logger()

# Security
security = HTTPBearer()

# Redis for caching and pub/sub
redis_client = Redis(host='localhost', port=6379, decode_responses=True)

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)

manager = ConnectionManager()

# Pydantic models for API
class StrategyCreateRequest(BaseModel):
    """Strategy creation request."""
    
    strategy_type: str = Field(..., description="Strategy type: mean_reversion, momentum, pairs")
    symbols: List[str] = Field(..., description="Trading symbols")
    config: Dict[str, Any] = Field(default_factory=dict, description="Strategy configuration")
    
    @validator('strategy_type')
    def validate_strategy_type(cls, v):
        valid_types = ['mean_reversion', 'momentum', 'pairs']
        if v not in valid_types:
            raise ValueError(f"Strategy type must be one of {valid_types}")
        return v


class BacktestRequest(BaseModel):
    """Backtest request."""
    
    strategy_id: str = Field(..., description="Strategy ID")
    start_date: datetime = Field(..., description="Backtest start date")
    end_date: datetime = Field(..., description="Backtest end date")
    initial_capital: float = Field(100000, description="Initial capital")
    mode: str = Field("event_driven", description="Backtest mode")


class PositionRequest(BaseModel):
    """Position management request."""
    
    symbol: str = Field(..., description="Symbol")
    side: str = Field(..., description="Position side: long, short")
    quantity: float = Field(..., description="Position quantity")
    entry_price: Optional[float] = Field(None, description="Entry price")


class RiskLimitUpdate(BaseModel):
    """Risk limit update request."""
    
    max_position_size: Optional[float] = Field(None, description="Max position size")
    max_portfolio_var: Optional[float] = Field(None, description="Max portfolio VaR")
    max_drawdown: Optional[float] = Field(None, description="Max drawdown")
    max_leverage: Optional[float] = Field(None, description="Max leverage")


class MarketDataRequest(BaseModel):
    """Market data request."""
    
    symbols: List[str] = Field(..., description="Symbols to fetch")
    start_date: datetime = Field(..., description="Start date")
    end_date: datetime = Field(..., description="End date")
    interval: str = Field("1d", description="Data interval")
    source: str = Field("yahoo", description="Data source")


class PerformanceResponse(BaseModel):
    """Performance metrics response."""
    
    total_return: float
    annual_return: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    var_95: float
    trades_count: int


# Application state
class AppState:
    def __init__(self):
        self.strategies: Dict[str, Any] = {}
        self.backtests: Dict[str, BacktestResult] = {}
        self.positions: Dict[str, Any] = {}
        self.risk_manager = RiskManager()
        self.data_connector = MultiSourceDataConnector()
        self.active_engines: Dict[str, BacktestingEngine] = {}
        
app_state = AppState()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("Starting QuantumTrader API")
    
    # Initialize background tasks
    asyncio.create_task(monitor_positions())
    asyncio.create_task(calculate_risk_metrics())
    
    yield
    
    # Shutdown
    logger.info("Shutting down QuantumTrader API")
    
    # Clean up resources
    for engine in app_state.active_engines.values():
        # Cleanup engine resources
        pass

# Create FastAPI app
app = FastAPI(
    title="QuantumTrader API",
    description="Professional Algorithmic Trading Platform",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Authentication dependency
async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify authentication token."""
    token = credentials.credentials
    
    # Implement proper token verification
    # For demo, just check if token exists
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials"
        )
    
    return token

# Health check endpoints
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now(),
        "version": "1.0.0"
    }

@app.get("/ready")
async def readiness_check():
    """Readiness check endpoint."""
    try:
        # Check Redis connection
        redis_client.ping()
        
        return {
            "status": "ready",
            "services": {
                "redis": "connected",
                "risk_manager": "active",
                "data_connector": "ready"
            }
        }
    except Exception as e:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"status": "not_ready", "error": str(e)}
        )

# Strategy management endpoints
@app.post("/api/v1/strategies", response_model=Dict[str, Any])
async def create_strategy(
    request: StrategyCreateRequest,
    token: str = Depends(verify_token)
):
    """Create a new trading strategy."""
    try:
        strategy_id = f"strategy_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Create strategy based on type
        if request.strategy_type == "mean_reversion":
            config = MeanReversionConfig(
                name=strategy_id,
                symbols=request.symbols,
                **request.config
            )
            strategy = MeanReversionStrategy(config)
        elif request.strategy_type == "momentum":
            config = MomentumConfig(
                name=strategy_id,
                symbols=request.symbols,
                **request.config
            )
            strategy = MomentumStrategy(config)
        else:
            raise ValueError(f"Unknown strategy type: {request.strategy_type}")
        
        # Initialize strategy
        strategy.initialize()
        
        # Store strategy
        app_state.strategies[strategy_id] = {
            "id": strategy_id,
            "type": request.strategy_type,
            "strategy": strategy,
            "config": request.dict(),
            "created_at": datetime.now(),
            "status": "active"
        }
        
        # Cache in Redis
        redis_client.set(
            f"strategy:{strategy_id}",
            json.dumps(request.dict(), default=str),
            ex=3600
        )
        
        logger.info(f"Created strategy {strategy_id}")
        
        return {
            "strategy_id": strategy_id,
            "status": "created",
            "message": f"Strategy {strategy_id} created successfully"
        }
        
    except Exception as e:
        logger.error(f"Failed to create strategy: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )

@app.get("/api/v1/strategies")
async def list_strategies(token: str = Depends(verify_token)):
    """List all strategies."""
    strategies = []
    
    for strategy_id, strategy_data in app_state.strategies.items():
        strategies.append({
            "id": strategy_id,
            "type": strategy_data["type"],
            "symbols": strategy_data["config"]["symbols"],
            "status": strategy_data["status"],
            "created_at": strategy_data["created_at"]
        })
    
    return {"strategies": strategies}

@app.get("/api/v1/strategies/{strategy_id}")
async def get_strategy(strategy_id: str, token: str = Depends(verify_token)):
    """Get strategy details."""
    if strategy_id not in app_state.strategies:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Strategy {strategy_id} not found"
        )
    
    strategy_data = app_state.strategies[strategy_id]
    
    return {
        "id": strategy_id,
        "type": strategy_data["type"],
        "config": strategy_data["config"],
        "status": strategy_data["status"],
        "created_at": strategy_data["created_at"],
        "performance": strategy_data["strategy"].get_performance_metrics()
    }

@app.delete("/api/v1/strategies/{strategy_id}")
async def delete_strategy(strategy_id: str, token: str = Depends(verify_token)):
    """Delete a strategy."""
    if strategy_id not in app_state.strategies:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Strategy {strategy_id} not found"
        )
    
    # Clean up strategy
    del app_state.strategies[strategy_id]
    redis_client.delete(f"strategy:{strategy_id}")
    
    return {"message": f"Strategy {strategy_id} deleted"}

# Backtesting endpoints
@app.post("/api/v1/backtest")
async def run_backtest(
    request: BacktestRequest,
    background_tasks: BackgroundTasks,
    token: str = Depends(verify_token)
):
    """Run a backtest."""
    try:
        if request.strategy_id not in app_state.strategies:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Strategy {request.strategy_id} not found"
            )
        
        backtest_id = f"backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Run backtest in background
        background_tasks.add_task(
            execute_backtest,
            backtest_id,
            request.strategy_id,
            request.dict()
        )
        
        return {
            "backtest_id": backtest_id,
            "status": "started",
            "message": "Backtest started in background"
        }
        
    except Exception as e:
        logger.error(f"Failed to start backtest: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )

async def execute_backtest(backtest_id: str, strategy_id: str, config: Dict):
    """Execute backtest in background."""
    try:
        logger.info(f"Starting backtest {backtest_id}")
        
        # Get strategy
        strategy = app_state.strategies[strategy_id]["strategy"]
        
        # Create backtest engine
        backtest_config = BacktestConfig(
            initial_capital=config["initial_capital"],
            mode=config.get("mode", "event_driven")
        )
        engine = BacktestingEngine(backtest_config)
        
        # Store active engine
        app_state.active_engines[backtest_id] = engine
        
        # Fetch market data
        symbols = app_state.strategies[strategy_id]["config"]["symbols"]
        for symbol in symbols:
            data = await fetch_market_data(
                symbol,
                config["start_date"],
                config["end_date"]
            )
            engine.add_data(symbol, data)
        
        # Run backtest
        result = engine.run(
            strategy,
            start_date=config["start_date"],
            end_date=config["end_date"]
        )
        
        # Store result
        app_state.backtests[backtest_id] = result
        
        # Cache result summary
        redis_client.set(
            f"backtest:{backtest_id}",
            json.dumps({
                "total_return": result.total_return,
                "sharpe_ratio": result.sharpe_ratio,
                "max_drawdown": result.max_drawdown,
                "status": "completed"
            }),
            ex=7200
        )
        
        # Clean up
        del app_state.active_engines[backtest_id]
        
        logger.info(f"Backtest {backtest_id} completed")
        
    except Exception as e:
        logger.error(f"Backtest {backtest_id} failed: {e}")
        redis_client.set(
            f"backtest:{backtest_id}",
            json.dumps({"status": "failed", "error": str(e)}),
            ex=3600
        )

@app.get("/api/v1/backtest/{backtest_id}")
async def get_backtest_result(backtest_id: str, token: str = Depends(verify_token)):
    """Get backtest results."""
    # Check cache first
    cached = redis_client.get(f"backtest:{backtest_id}")
    if cached:
        result = json.loads(cached)
        if result.get("status") == "failed":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=result.get("error")
            )
        return result
    
    # Check in-memory storage
    if backtest_id in app_state.backtests:
        result = app_state.backtests[backtest_id]
        
        return PerformanceResponse(
            total_return=result.total_return,
            annual_return=result.annual_return,
            sharpe_ratio=result.sharpe_ratio,
            sortino_ratio=result.sortino_ratio,
            max_drawdown=result.max_drawdown,
            win_rate=result.win_rate,
            profit_factor=result.profit_factor,
            var_95=result.var_95,
            trades_count=result.total_trades
        )
    
    # Check if still running
    if backtest_id in app_state.active_engines:
        return {"status": "running", "message": "Backtest is still running"}
    
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Backtest {backtest_id} not found"
    )

# Position management endpoints
@app.get("/api/v1/positions")
async def get_positions(token: str = Depends(verify_token)):
    """Get all positions."""
    positions = []
    
    for symbol, position in app_state.positions.items():
        positions.append({
            "symbol": symbol,
            "side": position.get("side"),
            "quantity": position.get("quantity"),
            "entry_price": position.get("entry_price"),
            "current_price": position.get("current_price"),
            "unrealized_pnl": position.get("unrealized_pnl"),
            "realized_pnl": position.get("realized_pnl")
        })
    
    return {"positions": positions}

@app.post("/api/v1/positions")
async def create_position(
    request: PositionRequest,
    token: str = Depends(verify_token)
):
    """Create or update a position."""
    try:
        # Validate with risk manager
        is_allowed, violations = app_state.risk_manager.check_risk_limits(
            signal=None,  # Create a signal from request
            positions=app_state.positions,
            portfolio_value=100000  # Get actual portfolio value
        )
        
        if not is_allowed:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Risk limits violated: {violations}"
            )
        
        # Create/update position
        app_state.positions[request.symbol] = {
            "symbol": request.symbol,
            "side": request.side,
            "quantity": request.quantity,
            "entry_price": request.entry_price,
            "entry_time": datetime.now(),
            "current_price": request.entry_price,
            "unrealized_pnl": 0,
            "realized_pnl": 0
        }
        
        return {"message": f"Position created for {request.symbol}"}
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )

@app.delete("/api/v1/positions/{symbol}")
async def close_position(symbol: str, token: str = Depends(verify_token)):
    """Close a position."""
    if symbol not in app_state.positions:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Position for {symbol} not found"
        )
    
    position = app_state.positions[symbol]
    
    # Calculate final P&L
    # This would integrate with execution engine
    
    del app_state.positions[symbol]
    
    return {"message": f"Position for {symbol} closed"}

# Risk management endpoints
@app.get("/api/v1/risk/metrics")
async def get_risk_metrics(token: str = Depends(verify_token)):
    """Get current risk metrics."""
    metrics = app_state.risk_manager.calculate_portfolio_metrics(
        app_state.positions,
        {}  # Current market prices
    )
    
    return {
        "var_95": metrics.portfolio_var_95,
        "var_99": metrics.portfolio_var_99,
        "cvar_95": metrics.portfolio_cvar_95,
        "sharpe_ratio": metrics.sharpe_ratio,
        "sortino_ratio": metrics.sortino_ratio,
        "gross_exposure": metrics.gross_exposure,
        "net_exposure": metrics.net_exposure,
        "leverage": metrics.leverage,
        "concentration": metrics.concentration,
        "avg_correlation": metrics.avg_correlation,
        "max_correlation": metrics.max_correlation
    }

@app.put("/api/v1/risk/limits")
async def update_risk_limits(
    request: RiskLimitUpdate,
    token: str = Depends(verify_token)
):
    """Update risk limits."""
    try:
        if request.max_position_size is not None:
            app_state.risk_manager.risk_limits.max_position_size = request.max_position_size
        
        if request.max_portfolio_var is not None:
            app_state.risk_manager.risk_limits.max_portfolio_var = request.max_portfolio_var
        
        if request.max_drawdown is not None:
            app_state.risk_manager.risk_limits.max_drawdown = request.max_drawdown
        
        if request.max_leverage is not None:
            app_state.risk_manager.risk_limits.max_portfolio_leverage = request.max_leverage
        
        return {"message": "Risk limits updated successfully"}
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )

@app.post("/api/v1/risk/stress-test")
async def run_stress_test(token: str = Depends(verify_token)):
    """Run portfolio stress test."""
    results = app_state.risk_manager.stress_test(app_state.positions)
    
    return {"stress_test_results": results}

# Market data endpoints
@app.post("/api/v1/market-data")
async def fetch_market_data_endpoint(
    request: MarketDataRequest,
    token: str = Depends(verify_token)
):
    """Fetch market data."""
    try:
        data_results = {}
        
        for symbol in request.symbols:
            data = await fetch_market_data(
                symbol,
                request.start_date,
                request.end_date,
                request.interval,
                request.source
            )
            data_results[symbol] = {
                "rows": len(data),
                "start": data.index[0].isoformat() if len(data) > 0 else None,
                "end": data.index[-1].isoformat() if len(data) > 0 else None
            }
        
        return {
            "status": "success",
            "data": data_results
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )

async def fetch_market_data(
    symbol: str,
    start_date: datetime,
    end_date: datetime,
    interval: str = "1d",
    source: str = "yahoo"
) -> pd.DataFrame:
    """Fetch market data from connector."""
    try:
        # Map source
        source_map = {
            "yahoo": MarketDataSource.YAHOO,
            "alpha_vantage": MarketDataSource.ALPHA_VANTAGE,
            "polygon": MarketDataSource.POLYGON
        }
        
        data_source = source_map.get(source, MarketDataSource.YAHOO)
        
        # Use data connector
        return app_state.data_connector.fetch_historical_data(
            symbol,
            start_date,
            end_date,
            interval
        )
        
    except Exception as e:
        logger.error(f"Failed to fetch data for {symbol}: {e}")
        raise

# WebSocket endpoints for real-time updates
@app.websocket("/ws/positions")
async def websocket_positions(websocket: WebSocket):
    """WebSocket for real-time position updates."""
    await manager.connect(websocket)
    
    try:
        while True:
            # Send position updates
            positions_data = json.dumps({
                "type": "positions",
                "data": app_state.positions,
                "timestamp": datetime.now().isoformat()
            })
            await manager.send_personal_message(positions_data, websocket)
            
            # Wait for 1 second
            await asyncio.sleep(1)
            
    except WebSocketDisconnect:
        manager.disconnect(websocket)

@app.websocket("/ws/signals")
async def websocket_signals(websocket: WebSocket):
    """WebSocket for real-time trading signals."""
    await manager.connect(websocket)
    
    try:
        while True:
            # This would connect to actual signal generation
            await asyncio.sleep(1)
            
    except WebSocketDisconnect:
        manager.disconnect(websocket)

# Background tasks
async def monitor_positions():
    """Monitor positions for risk and P&L."""
    while True:
        try:
            # Update position P&L
            for symbol, position in app_state.positions.items():
                # Fetch current price
                # Update unrealized P&L
                pass
            
            await asyncio.sleep(5)
            
        except Exception as e:
            logger.error(f"Position monitoring error: {e}")

async def calculate_risk_metrics():
    """Calculate risk metrics periodically."""
    while True:
        try:
            if app_state.positions:
                metrics = app_state.risk_manager.calculate_portfolio_metrics(
                    app_state.positions,
                    {}  # Current prices
                )
                
                # Store metrics
                redis_client.set(
                    "risk:current_metrics",
                    json.dumps({
                        "var_95": metrics.portfolio_var_95,
                        "sharpe_ratio": metrics.sharpe_ratio,
                        "leverage": metrics.leverage,
                        "timestamp": datetime.now().isoformat()
                    }),
                    ex=60
                )
            
            await asyncio.sleep(30)
            
        except Exception as e:
            logger.error(f"Risk calculation error: {e}")

# Export rate endpoints
@app.get("/api/v1/export/trades/{backtest_id}")
async def export_trades(
    backtest_id: str,
    format: str = "csv",
    token: str = Depends(verify_token)
):
    """Export trade history."""
    if backtest_id not in app_state.backtests:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Backtest {backtest_id} not found"
        )
    
    result = app_state.backtests[backtest_id]
    
    if format == "csv":
        # Convert trades to CSV
        output = result.trades.to_csv(index=False)
        
        return StreamingResponse(
            iter([output]),
            media_type="text/csv",
            headers={
                "Content-Disposition": f"attachment; filename=trades_{backtest_id}.csv"
            }
        )
    else:
        # Return as JSON
        return result.trades.to_dict(orient="records")

@app.get("/api/v1/export/performance/{backtest_id}")
async def export_performance(
    backtest_id: str,
    token: str = Depends(verify_token)
):
    """Export performance report."""
    if backtest_id not in app_state.backtests:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Backtest {backtest_id} not found"
        )
    
    result = app_state.backtests[backtest_id]
    
    report = {
        "backtest_id": backtest_id,
        "performance": {
            "total_return": result.total_return,
            "annual_return": result.annual_return,
            "sharpe_ratio": result.sharpe_ratio,
            "sortino_ratio": result.sortino_ratio,
            "calmar_ratio": result.calmar_ratio,
            "max_drawdown": result.max_drawdown
        },
        "trades": {
            "total": result.total_trades,
            "winning": result.winning_trades,
            "losing": result.losing_trades,
            "win_rate": result.win_rate,
            "profit_factor": result.profit_factor,
            "expectancy": result.expectancy
        },
        "risk": {
            "var_95": result.var_95,
            "cvar_95": result.cvar_95,
            "downside_deviation": result.downside_deviation
        }
    }
    
    return report

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
