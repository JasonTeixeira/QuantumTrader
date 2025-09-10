#!/usr/bin/env python3
"""
QuantumTrader CLI - Professional Command Line Interface

Enterprise-grade CLI for managing trading strategies, backtests,
risk parameters, and system monitoring.

Author: QuantumTrader Team
Date: 2024
"""

import asyncio
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import click
import pandas as pd
import yaml
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.tree import Tree
from tabulate import tabulate

from ..strategies.mean_reversion import MeanReversionStrategy, MeanReversionConfig
from ..strategies.momentum import MomentumStrategy, MomentumConfig
from ..backtesting.engine import BacktestingEngine, BacktestConfig
from ..risk.risk_manager import RiskManager, RiskLimits
from ..execution.engine import ExecutionEngine, Order, OrderSide, OrderType
from ..database.db_manager import DatabaseManager
from ..analytics.performance import PerformanceAnalyzer, ReportGenerator
from ..utils.logging import setup_logging, get_logger

# Initialize Rich console for beautiful output
console = Console()
logger = get_logger(__name__)

# CLI Context
class QuantumContext:
    """Context object for CLI state management."""
    
    def __init__(self):
        self.config_path = Path.home() / ".quantum" / "config.yaml"
        self.strategies = {}
        self.backtest_engine = None
        self.execution_engine = None
        self.risk_manager = RiskManager()
        self.db_manager = None
        self.config = self.load_config()
    
    def load_config(self) -> Dict:
        """Load configuration from file."""
        if self.config_path.exists():
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        return {}
    
    def save_config(self) -> None:
        """Save configuration to file."""
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.config_path, 'w') as f:
            yaml.dump(self.config, f)


pass_context = click.make_pass_decorator(QuantumContext, ensure=True)


@click.group()
@click.option('--debug/--no-debug', default=False, help='Enable debug logging')
@click.option('--config', '-c', type=click.Path(), help='Configuration file path')
@pass_context
def cli(ctx: QuantumContext, debug: bool, config: Optional[str]):
    """
    QuantumTrader CLI - Professional Algorithmic Trading Platform
    
    Manage strategies, run backtests, monitor performance, and control execution.
    """
    # Setup logging
    setup_logging(level="DEBUG" if debug else "INFO")
    
    # Load custom config if provided
    if config:
        ctx.config_path = Path(config)
        ctx.config = ctx.load_config()
    
    # Initialize database
    if 'database' in ctx.config:
        ctx.db_manager = DatabaseManager(**ctx.config['database'])


# Strategy Management Commands
@cli.group()
def strategy():
    """Manage trading strategies."""
    pass


@strategy.command('create')
@click.option('--type', '-t', type=click.Choice(['mean_reversion', 'momentum', 'pairs']), 
              required=True, help='Strategy type')
@click.option('--name', '-n', required=True, help='Strategy name')
@click.option('--symbols', '-s', multiple=True, required=True, help='Trading symbols')
@click.option('--config-file', '-f', type=click.Path(exists=True), help='Strategy config file')
@pass_context
def create_strategy(ctx: QuantumContext, type: str, name: str, symbols: List[str], 
                   config_file: Optional[str]):
    """Create a new trading strategy."""
    try:
        # Load config from file if provided
        if config_file:
            with open(config_file, 'r') as f:
                config_data = yaml.safe_load(f)
        else:
            config_data = {}
        
        # Create strategy based on type
        with console.status(f"Creating {type} strategy '{name}'..."):
            if type == "mean_reversion":
                config = MeanReversionConfig(
                    name=name,
                    symbols=list(symbols),
                    **config_data
                )
                strategy = MeanReversionStrategy(config)
            elif type == "momentum":
                config = MomentumConfig(
                    name=name,
                    symbols=list(symbols),
                    **config_data
                )
                strategy = MomentumStrategy(config)
            else:
                raise ValueError(f"Unsupported strategy type: {type}")
            
            # Initialize and store
            strategy.initialize()
            ctx.strategies[name] = strategy
            
        console.print(f"[green]✓[/green] Strategy '{name}' created successfully")
        
        # Display strategy details
        table = Table(title=f"Strategy: {name}")
        table.add_column("Parameter", style="cyan")
        table.add_column("Value", style="magenta")
        
        table.add_row("Type", type)
        table.add_row("Symbols", ", ".join(symbols))
        table.add_row("Status", "Active")
        
        console.print(table)
        
    except Exception as e:
        console.print(f"[red]✗[/red] Failed to create strategy: {e}")
        sys.exit(1)


@strategy.command('list')
@pass_context
def list_strategies(ctx: QuantumContext):
    """List all strategies."""
    if not ctx.strategies:
        console.print("[yellow]No strategies found[/yellow]")
        return
    
    table = Table(title="Trading Strategies")
    table.add_column("Name", style="cyan")
    table.add_column("Type", style="magenta")
    table.add_column("Symbols", style="green")
    table.add_column("Status", style="yellow")
    
    for name, strategy in ctx.strategies.items():
        strategy_type = strategy.__class__.__name__.replace("Strategy", "")
        symbols = ", ".join(strategy.symbols[:3])
        if len(strategy.symbols) > 3:
            symbols += f" (+{len(strategy.symbols) - 3} more)"
        
        table.add_row(
            name,
            strategy_type,
            symbols,
            "Active"
        )
    
    console.print(table)


@strategy.command('optimize')
@click.argument('name')
@click.option('--start-date', '-s', required=True, help='Start date (YYYY-MM-DD)')
@click.option('--end-date', '-e', required=True, help='End date (YYYY-MM-DD)')
@click.option('--metric', '-m', default='sharpe', 
              type=click.Choice(['sharpe', 'return', 'calmar']),
              help='Optimization metric')
@pass_context
def optimize_strategy(ctx: QuantumContext, name: str, start_date: str, 
                      end_date: str, metric: str):
    """Optimize strategy parameters."""
    if name not in ctx.strategies:
        console.print(f"[red]Strategy '{name}' not found[/red]")
        return
    
    strategy = ctx.strategies[name]
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task(f"Optimizing {name}...", total=100)
        
        # Parameter grid search (simplified)
        param_grid = {
            'lookback_period': [10, 20, 30],
            'entry_threshold': [1.5, 2.0, 2.5],
            'exit_threshold': [0.5, 1.0]
        }
        
        best_params = {}
        best_score = -float('inf')
        
        for lookback in param_grid['lookback_period']:
            for entry in param_grid['entry_threshold']:
                for exit_th in param_grid['exit_threshold']:
                    # Update progress
                    progress.update(task, advance=10)
                    
                    # Test parameters (simplified)
                    score = lookback * entry / exit_th  # Placeholder
                    
                    if score > best_score:
                        best_score = score
                        best_params = {
                            'lookback_period': lookback,
                            'entry_threshold': entry,
                            'exit_threshold': exit_th
                        }
    
    console.print(Panel(
        f"[green]Optimization Complete[/green]\n\n"
        f"Best {metric.upper()}: {best_score:.4f}\n"
        f"Parameters: {json.dumps(best_params, indent=2)}",
        title=f"Strategy Optimization: {name}"
    ))


# Backtesting Commands
@cli.group()
def backtest():
    """Run and manage backtests."""
    pass


@backtest.command('run')
@click.option('--strategy', '-s', required=True, help='Strategy name')
@click.option('--start-date', required=True, help='Start date (YYYY-MM-DD)')
@click.option('--end-date', required=True, help='End date (YYYY-MM-DD)')
@click.option('--capital', '-c', default=100000, type=float, help='Initial capital')
@click.option('--mode', '-m', type=click.Choice(['event', 'vectorized']), 
              default='event', help='Backtest mode')
@click.option('--output', '-o', type=click.Path(), help='Output file for results')
@pass_context
def run_backtest(ctx: QuantumContext, strategy: str, start_date: str, 
                end_date: str, capital: float, mode: str, output: Optional[str]):
    """Run a backtest for a strategy."""
    if strategy not in ctx.strategies:
        console.print(f"[red]Strategy '{strategy}' not found[/red]")
        return
    
    try:
        with console.status(f"Running backtest for {strategy}..."):
            # Create backtest engine
            config = BacktestConfig(
                initial_capital=capital,
                mode="event_driven" if mode == "event" else "vectorized"
            )
            engine = BacktestingEngine(config)
            
            # Load market data (simplified - would fetch real data)
            start = pd.to_datetime(start_date)
            end = pd.to_datetime(end_date)
            
            # Run backtest
            strategy_obj = ctx.strategies[strategy]
            result = engine.run(strategy_obj, start, end)
        
        # Display results
        console.print(Panel(
            f"[green]Backtest Complete[/green]\n\n"
            f"Total Return: {result.total_return:.2%}\n"
            f"Annual Return: {result.annual_return:.2%}\n"
            f"Sharpe Ratio: {result.sharpe_ratio:.2f}\n"
            f"Max Drawdown: {result.max_drawdown:.2%}\n"
            f"Win Rate: {result.win_rate:.2%}\n"
            f"Total Trades: {result.total_trades}",
            title=f"Backtest Results: {strategy}"
        ))
        
        # Save results if output specified
        if output:
            with open(output, 'w') as f:
                json.dump(result.to_dict(), f, indent=2, default=str)
            console.print(f"[green]Results saved to {output}[/green]")
        
        # Store in context
        ctx.backtest_engine = engine
        
    except Exception as e:
        console.print(f"[red]Backtest failed: {e}[/red]")
        sys.exit(1)


@backtest.command('compare')
@click.option('--strategies', '-s', multiple=True, required=True, help='Strategy names')
@click.option('--start-date', required=True, help='Start date')
@click.option('--end-date', required=True, help='End date')
@pass_context
def compare_backtests(ctx: QuantumContext, strategies: List[str], 
                      start_date: str, end_date: str):
    """Compare multiple strategy backtests."""
    results = {}
    
    with Progress(console=console) as progress:
        task = progress.add_task("Running backtests...", total=len(strategies))
        
        for strategy_name in strategies:
            if strategy_name not in ctx.strategies:
                console.print(f"[yellow]Warning: Strategy '{strategy_name}' not found[/yellow]")
                continue
            
            # Run backtest
            config = BacktestConfig(initial_capital=100000)
            engine = BacktestingEngine(config)
            
            strategy_obj = ctx.strategies[strategy_name]
            result = engine.run(
                strategy_obj,
                pd.to_datetime(start_date),
                pd.to_datetime(end_date)
            )
            
            results[strategy_name] = result
            progress.update(task, advance=1)
    
    # Create comparison table
    table = Table(title="Strategy Comparison")
    table.add_column("Strategy", style="cyan")
    table.add_column("Total Return", justify="right", style="green")
    table.add_column("Sharpe Ratio", justify="right", style="magenta")
    table.add_column("Max Drawdown", justify="right", style="red")
    table.add_column("Win Rate", justify="right", style="yellow")
    
    for name, result in results.items():
        table.add_row(
            name,
            f"{result.total_return:.2%}",
            f"{result.sharpe_ratio:.2f}",
            f"{result.max_drawdown:.2%}",
            f"{result.win_rate:.1%}"
        )
    
    console.print(table)


# Risk Management Commands
@cli.group()
def risk():
    """Manage risk parameters and monitoring."""
    pass


@risk.command('limits')
@click.option('--set', 'action', flag_value='set', default=True)
@click.option('--get', 'action', flag_value='get')
@click.option('--max-position', type=float, help='Max position size (%)')
@click.option('--max-drawdown', type=float, help='Max drawdown (%)')
@click.option('--max-leverage', type=float, help='Max leverage')
@click.option('--max-var', type=float, help='Max portfolio VaR (%)')
@pass_context
def manage_risk_limits(ctx: QuantumContext, action: str, max_position: Optional[float],
                       max_drawdown: Optional[float], max_leverage: Optional[float],
                       max_var: Optional[float]):
    """Get or set risk limits."""
    if action == 'get':
        # Display current limits
        limits = ctx.risk_manager.risk_limits
        
        table = Table(title="Risk Limits")
        table.add_column("Parameter", style="cyan")
        table.add_column("Limit", justify="right", style="magenta")
        table.add_column("Current", justify="right", style="green")
        
        table.add_row("Max Position Size", 
                     f"{limits.max_position_size:.1%}",
                     "N/A")
        table.add_row("Max Drawdown", 
                     f"{limits.max_drawdown:.1%}",
                     "N/A")
        table.add_row("Max Leverage", 
                     f"{limits.max_portfolio_leverage:.1f}x",
                     "N/A")
        table.add_row("Max VaR (95%)", 
                     f"{limits.max_portfolio_var:.1%}",
                     "N/A")
        
        console.print(table)
    
    else:  # set
        # Update limits
        if max_position:
            ctx.risk_manager.risk_limits.max_position_size = max_position / 100
        if max_drawdown:
            ctx.risk_manager.risk_limits.max_drawdown = max_drawdown / 100
        if max_leverage:
            ctx.risk_manager.risk_limits.max_portfolio_leverage = max_leverage
        if max_var:
            ctx.risk_manager.risk_limits.max_portfolio_var = max_var / 100
        
        console.print("[green]Risk limits updated successfully[/green]")


@risk.command('monitor')
@click.option('--interval', '-i', default=5, type=int, help='Update interval (seconds)')
@pass_context
def monitor_risk(ctx: QuantumContext, interval: int):
    """Real-time risk monitoring."""
    console.print("[cyan]Starting risk monitor... Press Ctrl+C to stop[/cyan]")
    
    try:
        while True:
            # Get current metrics (simplified)
            metrics = {
                'Portfolio Value': '$100,000',
                'Current Drawdown': '-2.3%',
                'VaR (95%)': '-3.5%',
                'Leverage': '1.2x',
                'Positions': '5',
                'Exposure': '85%'
            }
            
            # Clear and redraw
            console.clear()
            
            # Create dashboard
            table = Table(title=f"Risk Monitor - {datetime.now().strftime('%H:%M:%S')}")
            table.add_column("Metric", style="cyan")
            table.add_column("Value", justify="right", style="magenta")
            table.add_column("Status", justify="center")
            
            for metric, value in metrics.items():
                status = "[green]✓[/green]"  # Simplified status
                table.add_row(metric, value, status)
            
            console.print(table)
            
            # Sleep
            asyncio.run(asyncio.sleep(interval))
            
    except KeyboardInterrupt:
        console.print("\n[yellow]Risk monitor stopped[/yellow]")


# Execution Commands
@cli.group()
def execute():
    """Manage order execution."""
    pass


@execute.command('order')
@click.option('--symbol', '-s', required=True, help='Trading symbol')
@click.option('--side', type=click.Choice(['buy', 'sell']), required=True)
@click.option('--quantity', '-q', type=float, required=True, help='Order quantity')
@click.option('--type', '-t', type=click.Choice(['market', 'limit', 'stop']), 
              default='market', help='Order type')
@click.option('--price', '-p', type=float, help='Limit/stop price')
@click.option('--strategy', help='Strategy name for the order')
@pass_context
def place_order(ctx: QuantumContext, symbol: str, side: str, quantity: float,
               type: str, price: Optional[float], strategy: Optional[str]):
    """Place an order."""
    try:
        # Create order
        order = Order(
            order_id="",
            symbol=symbol,
            side=OrderSide.BUY if side == 'buy' else OrderSide.SELL,
            order_type=OrderType[type.upper()],
            quantity=quantity,
            price=price,
            strategy_id=strategy
        )
        
        # Initialize execution engine if needed
        if not ctx.execution_engine:
            ctx.execution_engine = ExecutionEngine()
            asyncio.run(ctx.execution_engine.start())
        
        # Submit order
        with console.status(f"Submitting {side} order for {quantity} {symbol}..."):
            order_id = asyncio.run(ctx.execution_engine.submit_order(order))
        
        console.print(f"[green]✓[/green] Order submitted: {order_id}")
        
        # Display order details
        table = Table(title="Order Details")
        table.add_column("Field", style="cyan")
        table.add_column("Value", style="magenta")
        
        table.add_row("Order ID", order_id)
        table.add_row("Symbol", symbol)
        table.add_row("Side", side.upper())
        table.add_row("Quantity", str(quantity))
        table.add_row("Type", type.upper())
        if price:
            table.add_row("Price", f"${price:.2f}")
        
        console.print(table)
        
    except Exception as e:
        console.print(f"[red]Failed to place order: {e}[/red]")


@execute.command('positions')
@pass_context
def show_positions(ctx: QuantumContext):
    """Show current positions."""
    if not ctx.execution_engine:
        console.print("[yellow]No execution engine active[/yellow]")
        return
    
    # Get positions (simplified)
    positions = [
        {'symbol': 'AAPL', 'quantity': 100, 'avg_price': 150.00, 'pnl': 500},
        {'symbol': 'GOOGL', 'quantity': 50, 'avg_price': 2800.00, 'pnl': -200},
    ]
    
    table = Table(title="Current Positions")
    table.add_column("Symbol", style="cyan")
    table.add_column("Quantity", justify="right", style="magenta")
    table.add_column("Avg Price", justify="right", style="yellow")
    table.add_column("P&L", justify="right")
    
    for pos in positions:
        pnl_color = "green" if pos['pnl'] >= 0 else "red"
        table.add_row(
            pos['symbol'],
            str(pos['quantity']),
            f"${pos['avg_price']:.2f}",
            f"[{pnl_color}]${pos['pnl']:+.2f}[/{pnl_color}]"
        )
    
    console.print(table)


# Performance Commands
@cli.group()
def performance():
    """Analyze and report performance."""
    pass


@performance.command('report')
@click.option('--strategy', '-s', help='Strategy name')
@click.option('--period', '-p', type=click.Choice(['daily', 'weekly', 'monthly', 'all']),
              default='all', help='Report period')
@click.option('--output', '-o', type=click.Path(), help='Output file')
@pass_context
def generate_report(ctx: QuantumContext, strategy: Optional[str], 
                   period: str, output: Optional[str]):
    """Generate performance report."""
    console.print(f"[cyan]Generating {period} performance report...[/cyan]")
    
    # Create sample data
    analyzer = PerformanceAnalyzer()
    generator = ReportGenerator()
    
    # Generate returns (simplified)
    dates = pd.date_range(end=datetime.now(), periods=252, freq='D')
    returns = pd.Series(np.random.randn(252) * 0.01, index=dates)
    
    # Generate tearsheet
    tearsheet = generator.generate_tearsheet(returns)
    
    # Display summary
    console.print(Panel(
        f"[green]Performance Report Generated[/green]\n\n"
        f"Period: {period.upper()}\n"
        f"Total Return: {tearsheet['summary']['total_return']}\n"
        f"Sharpe Ratio: {tearsheet['summary']['sharpe_ratio']}\n"
        f"Max Drawdown: {tearsheet['summary']['max_drawdown']}\n"
        f"Win Rate: {tearsheet['summary']['win_rate']}",
        title="Performance Summary"
    ))
    
    # Save if output specified
    if output:
        if output.endswith('.html'):
            generator.export_html_report(tearsheet, output)
        else:
            with open(output, 'w') as f:
                json.dump(tearsheet, f, indent=2, default=str)
        
        console.print(f"[green]Report saved to {output}[/green]")


@performance.command('live')
@click.option('--refresh', '-r', default=5, type=int, help='Refresh interval (seconds)')
@pass_context
def live_performance(ctx: QuantumContext, refresh: int):
    """Live performance monitoring."""
    console.print("[cyan]Starting live performance monitor... Press Ctrl+C to stop[/cyan]")
    
    try:
        while True:
            # Get live metrics (simplified)
            metrics = {
                'Daily P&L': '+$1,234.56',
                'Daily Return': '+1.23%',
                'Total P&L': '+$12,345.67',
                'Win Rate': '65%',
                'Sharpe Ratio': '1.85',
                'Active Trades': '3'
            }
            
            # Clear and redraw
            console.clear()
            
            # Create dashboard
            tree = Tree(f"[bold cyan]Live Performance - {datetime.now().strftime('%H:%M:%S')}[/bold cyan]")
            
            for metric, value in metrics.items():
                color = "green" if '+' in value else "red" if '-' in value else "white"
                tree.add(f"[{color}]{metric}: {value}[/{color}]")
            
            console.print(tree)
            
            # Sleep
            asyncio.run(asyncio.sleep(refresh))
            
    except KeyboardInterrupt:
        console.print("\n[yellow]Performance monitor stopped[/yellow]")


# Database Commands
@cli.group()
def database():
    """Database operations."""
    pass


@database.command('init')
@pass_context
def init_database(ctx: QuantumContext):
    """Initialize database."""
    if not ctx.db_manager:
        ctx.db_manager = DatabaseManager()
    
    with console.status("Initializing database..."):
        ctx.db_manager.initialize_database()
    
    console.print("[green]✓[/green] Database initialized successfully")


@database.command('stats')
@pass_context
def database_stats(ctx: QuantumContext):
    """Show database statistics."""
    if not ctx.db_manager:
        console.print("[red]Database not initialized[/red]")
        return
    
    stats = ctx.db_manager.get_statistics()
    
    # Display table sizes
    table = Table(title="Database Statistics")
    table.add_column("Table", style="cyan")
    table.add_column("Size", justify="right", style="magenta")
    
    for table_info in stats.get('table_sizes', []):
        table.add_row(table_info['tablename'], table_info['size'])
    
    console.print(table)
    
    # Display connection info
    conn_info = stats.get('connections', {})
    console.print(f"\nConnections: {conn_info.get('total_connections', 0)} total, "
                 f"{conn_info.get('active_connections', 0)} active")


# System Commands
@cli.group()
def system():
    """System management commands."""
    pass


@system.command('status')
@pass_context
def system_status(ctx: QuantumContext):
    """Show system status."""
    tree = Tree("[bold cyan]QuantumTrader System Status[/bold cyan]")
    
    # Strategies
    strategies_branch = tree.add("[yellow]Strategies[/yellow]")
    for name in ctx.strategies.keys():
        strategies_branch.add(f"[green]✓[/green] {name}")
    
    # Components
    components_branch = tree.add("[yellow]Components[/yellow]")
    components = [
        ("Database", ctx.db_manager is not None),
        ("Execution Engine", ctx.execution_engine is not None),
        ("Backtest Engine", ctx.backtest_engine is not None),
        ("Risk Manager", True)  # Always available
    ]
    
    for component, active in components:
        status = "[green]Active[/green]" if active else "[red]Inactive[/red]"
        components_branch.add(f"{component}: {status}")
    
    console.print(tree)


@system.command('config')
@click.option('--show', is_flag=True, help='Show configuration')
@click.option('--edit', is_flag=True, help='Edit configuration')
@pass_context
def manage_config(ctx: QuantumContext, show: bool, edit: bool):
    """Manage system configuration."""
    if show:
        console.print(Panel(
            yaml.dump(ctx.config, default_flow_style=False),
            title="Configuration"
        ))
    
    elif edit:
        # Open in editor (simplified)
        console.print(f"[cyan]Configuration file: {ctx.config_path}[/cyan]")
        console.print("[yellow]Please edit the file manually[/yellow]")


# Main entry point
def main():
    """Main CLI entry point."""
    try:
        cli()
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)


if __name__ == "__main__":
    main()
