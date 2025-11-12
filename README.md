# QuantumTrader

An algorithmic trading system I built to explore momentum and mean-reversion strategies. Event-driven architecture with backtesting engine, risk management, and real-time execution.

```
┌─────────────────────────────────────────────────────────────────┐
│  WHAT THIS IS                                                   │
├─────────────────────────────────────────────────────────────────┤
│  Trading Engine           Event-driven, strategy-agnostic       │
│  Backtesting              Transaction costs, slippage modeling  │
│  Risk Management          Position sizing, stop losses          │
│  Execution Engine         Order routing, fill simulation        │
│  Data Pipeline            Multi-source market data              │
│  11,000+ Lines Python     Real implementation, not a demo       │
└─────────────────────────────────────────────────────────────────┘
```

## Why I Built This

Wanted to understand how algorithmic trading systems actually work—not just backtesting but real execution logic, risk management, and event handling. Built to answer: can I create a system that doesn't blow up an account?

Focus areas:
- Event-driven architecture (no lookahead bias)
- Proper backtesting (transaction costs, slippage)
- Risk management (position sizing, stop losses)
- Strategy isolation (easy to add new strategies)

Used Python because the quant libraries are Python-native. Event-driven because it's the only honest way to backtest. PostgreSQL for trade history, Redis for caching.

## Quick Start

```bash
git clone https://github.com/JasonTeixeira/QuantumTrader.git
cd QuantumTrader

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install -r requirements-base.txt

# Start infrastructure (PostgreSQL, Redis)
docker-compose up -d

# Initialize database
python scripts/init_databases.py

# Run backtest
python -m QuantumTrader.src.cli.quantum_cli backtest \
  --strategy momentum \
  --start 2023-01-01 \
  --end 2024-01-01
```

## Architecture

Event-driven system with clean separation:

```
   Market Data Sources
        │
        v
   Data Pipeline              Load, validate, normalize
   ├── Polygon API
   ├── Alpaca API
   └── Yahoo Finance
        │
        v
   Event Queue                Event-driven processing
   ├── Market data events
   ├── Signal events
   ├── Order events
   └── Fill events
        │
        v
   Strategy Engine            Generate signals
   ├── Momentum strategy
   └── Mean reversion
        │
        v
   Risk Manager               Position sizing, limits
        │
        v
   Execution Engine           Order routing, fills
        │
        v
   Database (PostgreSQL)      Trade history, analytics
```

## Project Structure

```
QuantumTrader/
├── src/
│   ├── engine/
│   │   └── base_strategy.py     # Base strategy class
│   ├── strategies/
│   │   ├── momentum.py          # Momentum strategy
│   │   └── mean_reversion.py    # Mean reversion
│   ├── backtesting/
│   │   └── engine.py            # Backtest engine
│   ├── execution/
│   │   └── engine.py            # Order execution
│   ├── risk/
│   │   └── risk_manager.py      # Risk management
│   ├── data/
│   │   ├── connectors.py        # Data source connectors
│   │   └── models.py            # Data models
│   ├── events/
│   │   └── event_system.py      # Event queue
│   ├── database/
│   │   └── db_manager.py        # Database operations
│   ├── analytics/
│   │   └── performance.py       # Performance metrics
│   ├── api/
│   │   └── main.py              # FastAPI server
│   └── cli/
│       └── quantum_cli.py       # CLI interface
├── tests/
│   └── test_strategies.py       # Strategy tests
├── docker-compose.yml           # Infrastructure
└── requirements-base.txt        # Dependencies
```

## Strategies

### Momentum Strategy

Simple momentum: buy when price crosses above 20-day SMA, sell when crosses below.

```python
from QuantumTrader.src.strategies.momentum import MomentumStrategy

strategy = MomentumStrategy(
    lookback_period=20,
    threshold=2.0  # Standard deviations
)
```

### Mean Reversion Strategy

Bollinger Bands mean reversion: buy oversold, sell overbought.

```python
from QuantumTrader.src.strategies.mean_reversion import MeanReversionStrategy

strategy = MeanReversionStrategy(
    window=20,
    num_std=2.0
)
```

## Backtesting

Backtest engine includes:
- Transaction costs (0.1% per trade)
- Slippage modeling
- Position sizing
- Stop losses
- Portfolio-level risk limits

```python
from QuantumTrader.src.backtesting.engine import BacktestEngine
from QuantumTrader.src.strategies.momentum import MomentumStrategy

# Initialize backtest
backtest = BacktestEngine(
    initial_capital=100000,
    commission=0.001,  # 0.1%
    slippage=0.0005    # 0.05%
)

# Add strategy
strategy = MomentumStrategy(lookback_period=20)
backtest.add_strategy(strategy)

# Run backtest
results = backtest.run(
    symbols=['AAPL', 'GOOGL', 'MSFT'],
    start_date='2023-01-01',
    end_date='2024-01-01'
)

print(results.summary())
```

## Risk Management

Position sizing based on portfolio risk:
- Max 10% per position
- Max 50% total exposure
- Stop loss at 2% per position
- Portfolio stop at 10% drawdown

```python
from QuantumTrader.src.risk.risk_manager import RiskManager

risk_mgr = RiskManager(
    max_position_size=0.10,  # 10% per position
    max_exposure=0.50,        # 50% total
    stop_loss_pct=0.02,       # 2% stop loss
    portfolio_stop=0.10       # 10% portfolio stop
)
```

## What Was Hard

- **Event-driven complexity:** Getting event ordering right is tricky. Signal → Order → Fill with proper timestamps.
- **Backtesting honesty:** Easy to leak lookahead bias. Had to be strict about using only past data.
- **Transaction costs:** Theoretical Sharpe of 2.5 dropped to 1.8 after costs. Optimization needed.
- **Data quality:** Bad ticks, missing data, corporate actions. Validation layer was critical.
- **Risk management:** Position sizing math is harder than it looks. Got it wrong twice before getting it right.

## Testing

```bash
# Unit tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=QuantumTrader.src --cov-report=html

# Specific test
pytest tests/test_strategies.py -k test_momentum
```

## Tech Stack

```
┌─────────────────────────────────────────────────────────────────┐
│  Component           Technology                                 │
├─────────────────────────────────────────────────────────────────┤
│  Language            Python 3.10+                               │
│  Data Processing     Pandas, NumPy                             │
│  Database            PostgreSQL (trade history)                │
│  Caching             Redis (market data)                       │
│  API                 FastAPI (monitoring, control)             │
│  Testing             Pytest                                    │
│  Deployment          Docker + docker-compose                   │
└─────────────────────────────────────────────────────────────────┘
```

## API

FastAPI server for monitoring and control:

```bash
# Start API server
python -m QuantumTrader.src.api.main

# Endpoints
GET  /status                  # System status
GET  /positions               # Current positions
GET  /performance             # Performance metrics
POST /strategy/start          # Start strategy
POST /strategy/stop           # Stop strategy
```

## Deployment

```bash
# Docker
docker-compose up -d

# Manual
python -m QuantumTrader.src.cli.quantum_cli run \
  --strategy momentum \
  --symbols AAPL,GOOGL,MSFT \
  --mode live
```

## What I'd Do Differently

- **More data sources:** Only 3 sources. Would add more for redundancy.
- **Better order routing:** Simple fill simulation. Would add realistic slippage models.
- **Machine learning:** Strategies are rule-based. Would experiment with ML signal generation.
- **Multi-timeframe:** Daily only. Would add intraday strategies.
- **Production monitoring:** Basic logging. Would add Prometheus/Grafana.

## Roadmap

- [ ] Machine learning strategies (RF, XGBoost)
- [ ] Intraday trading (1min, 5min bars)
- [ ] Options strategies
- [ ] Portfolio optimization
- [ ] Production monitoring (Prometheus, Grafana)
- [ ] Cloud deployment (AWS, GCP)

## License

MIT

## Note on Other Projects

This repo originally planned 6 sub-projects (RiskLab, AlphaForge, etc.). Currently only QuantumTrader is implemented. The others are placeholders for future work.
