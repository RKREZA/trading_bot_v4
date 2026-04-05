"""
BACKTEST ENGINE V4 - Institutional Multi-Strategy Interface
High-fidelity historical simulation with Portfolio Governance.
"""

import argparse
import json
import os
import sys
import logging
from datetime import datetime, timezone, timedelta
from typing import Optional, List, Dict, Any

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import print as rprint

from dotenv import load_dotenv
from core import DataFetcher, MT5Connection, PerformanceTracker
from backtesting import PortfolioBacktester, MonteCarloSimulator
from strategies import create_strategy, STRATEGY_REGISTRY

load_dotenv()

class BacktestCLI:
    def __init__(self):
        self.console = Console()
        self.config = self._load_config("config.json")
        self.data_fetcher = DataFetcher()
        self.connection = MT5Connection()
        
    def _load_config(self, path: str) -> dict:
        if not os.path.exists(path):
            rprint(f"[bold red]Error:[/] {path} not found. Using defaults.")
            return {"initial_balance": 1000.0}
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def run(
        self,
        symbol: str,
        start: str,
        end: str,
        strategy_filter: str = None,
        run_monte_carlo: bool = False,
        seed: Optional[int] = None,
        deterministic: Optional[bool] = None,
    ):
        if not self.connection.connect():
            rprint("[bold red]Critical Error:[/] Could not connect to MT5.")
            return

        try:
            dt_from = datetime.fromisoformat(start).replace(tzinfo=timezone.utc)
            dt_to = datetime.fromisoformat(end).replace(tzinfo=timezone.utc)
        except ValueError:
            rprint("[bold red]Error:[/] Start/End dates must be in YYYY-MM-DD format.")
            return

        # 1. Fetch High-Fidelity Data
        rprint(Panel(f"[bold cyan]Simulation for {symbol}[/]\nRange: {start} to {end}", border_style="bright_blue"))
        
        with self.console.status("[bold green]Fetching Historical Data (M1/M5/M15/H1)...") as status:
            m1 = self.data_fetcher.fetch_candles_range(symbol, "M1", dt_from, dt_to)
            m5 = self.data_fetcher.fetch_candles_range(symbol, "M5", dt_from, dt_to)
            m15 = self.data_fetcher.fetch_candles_range(symbol, "M15", dt_from, dt_to)
            h1 = self.data_fetcher.fetch_candles_range(symbol, "H1", dt_from, dt_to)

        if len(m5) < 100:
            rprint("[bold red]Error:[/] Insufficient data for backtest.")
            return

        # 2. Build strategies from config registry
        strategies = self._build_strategies(strategy_filter)
        if not strategies:
            rprint("[bold red]Error:[/] No strategies available for backtest.")
            return

        # 3. Initialize V4 Backtester
        runtime_config = dict(self.config)
        runtime_backtest = dict(runtime_config.get("backtest", {}))
        if seed is not None:
            runtime_backtest["random_seed"] = int(seed)
        if deterministic is not None:
            runtime_backtest["deterministic"] = bool(deterministic)
        runtime_config["backtest"] = runtime_backtest

        backtester = PortfolioBacktester(runtime_config)
        
        # 4. Execute Simulation
        rprint(f"[green]Executing Institutional Backtest for {len(strategies)} strategies...[/]")
        history = backtester.run(symbol, strategies, m5, h1, m15, m1)
        
        # 5. Performance Report
        partition_initial = float(self.config.get("backtest", {}).get("initial_balance", self.config.get("initial_balance", 1000.0)))
        total_initial = len(strategies) * partition_initial
        stats = PerformanceTracker.calculate_metrics(history, total_initial)
        strat_stats = PerformanceTracker.calculate_per_strategy(history, partition_initial)
        session_stats = PerformanceTracker.calculate_per_session(history, total_initial)
        
        self._display_results(stats, strat_stats, session_stats)
        
        if run_monte_carlo and history:
            self._run_monte_carlo(history)

    def _build_strategies(self, strategy_filter: str = None):
        strategies = []
        configured = self.config.get("strategies", [])

        if configured:
            for strat_cfg in configured:
                sid = strat_cfg.get("id")
                stype = strat_cfg.get("type")
                if not sid or not stype:
                    continue
                if strategy_filter and strategy_filter not in {sid, stype}:
                    continue

                cls = STRATEGY_REGISTRY.get(stype)
                if cls is None:
                    rprint(f"[bold yellow]Warning:[/] Unknown strategy type '{stype}' for '{sid}'.")
                    continue

                merged_cfg = dict(self.config)
                merged_cfg.update(strat_cfg)
                merged_cfg["params"] = strat_cfg.get("params", {})
                merged_cfg["enabled"] = bool(strat_cfg.get("enabled", True))

                try:
                    strategies.append(cls(sid, merged_cfg))
                except Exception as e:
                    rprint(f"[bold yellow]Warning:[/] Could not load strategy {sid} ({stype}): {e}")

            if strategies:
                return strategies

        # Fallback: registry-wide default set
        fallback_types = [strategy_filter] if strategy_filter else ["TREND_FOLLOWING", "MEAN_REVERSION", "BREAKOUT", "LIQUIDITY_SESSION"]
        for stype in fallback_types:
            try:
                strategies.append(create_strategy(stype, self.config))
            except Exception as e:
                rprint(f"[bold yellow]Warning:[/] Could not load strategy {stype}: {e}")
        return strategies

    def _display_results(self, stats: dict, strat_stats: dict, session_stats: dict):
        # Global Table
        table = Table(title="Institutional V4 Portfolio Summary", header_style="bold magenta")
        for k in stats.keys():
            table.add_column(k.replace("_", " ").title(), justify="center")
        
        # Convert all values to string for Rich
        row_values = []
        for v in stats.values():
            if isinstance(v, (int, float)):
                row_values.append(f"{v:,.2f}")
            else:
                row_values.append(str(v))
        table.add_row(*row_values)
        
        self.console.print(table)
        
        # Strategy Table
        s_table = Table(title="Individual Strategy Performance ($1000 Partitioned)", header_style="bold cyan")
        s_table.add_column("Strategy", style="bold")
        s_table.add_column("Trades", justify="right")
        s_table.add_column("Win Rate", justify="right")
        s_table.add_column("Profit ($)", justify="right")
        s_table.add_column("Sharpe", justify="right")
        s_table.add_column("Max DD", justify="right")
        
        for sid, s_data in strat_stats.items():
            if "status" in s_data: continue
            
            profit = s_data.get('net_profit', 0)
            profit_style = "green" if profit >= 0 else "red"
            
            s_table.add_row(
                sid, 
                str(s_data.get('total_trades', 0)), 
                s_data.get('win_rate', '0%'), 
                f"[{profit_style}]${profit:,.2f}[/]", 
                str(s_data.get('sharpe_ratio', 0)),
                s_data.get('max_drawdown', '0%')
            )
        self.console.print(s_table)
        
        # Session Table
        sess_table = Table(title="Institutional Session-Wise Breakdown", header_style="bold yellow")
        sess_table.add_column("Session", style="bold")
        sess_table.add_column("Trades", justify="right")
        sess_table.add_column("Win Rate", justify="right")
        sess_table.add_column("Net Profit ($)", justify="right")
        sess_table.add_column("Expectancy", justify="right")
        
        for sess, s_data in session_stats.items():
            if "status" in s_data: continue
            
            profit = s_data.get('net_profit', 0)
            profit_style = "green" if profit >= 0 else "red"
            
            sess_table.add_row(
                sess, 
                str(s_data.get('total_trades', 0)), 
                s_data.get('win_rate', '0%'), 
                f"[{profit_style}]${profit:,.2f}[/]", 
                str(s_data.get('expectancy', 0))
            )
        self.console.print(sess_table)

    def _run_monte_carlo(self, history):
        rprint("\n[bold yellow]Stress Testing: Running Monte Carlo Simulation (2000 paths)...[/]")
        mc = MonteCarloSimulator(iterations=2000)
        res = mc.run(history)
        
        mc_table = Table(title="Monte Carlo Confidence Forecast", box=None)
        mc_table.add_column("Metric", style="dim")
        mc_table.add_column("Value", style="bold")
        
        for k, v in res.items():
            mc_table.add_row(k.replace("_", " ").title(), str(v))
        self.console.print(Panel(mc_table, border_style="red", title="Robustness Report"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="V4 Institutional Backtester CLI")
    parser.add_argument("--symbol", type=str, default="XAUUSDm")
    parser.add_argument("--from", dest="start_date", type=str, required=True, help="YYYY-MM-DD")
    parser.add_argument("--to", dest="end_date", type=str, required=True, help="YYYY-MM-DD")
    parser.add_argument("--strategy", type=str, default=None, help="Optional: single strategy ID")
    parser.add_argument("--monte-carlo", action="store_true", help="Run robustness stress test")
    parser.add_argument("--seed", type=int, default=None, help="Override backtest random seed")
    parser.add_argument(
        "--deterministic",
        choices=["on", "off"],
        default=None,
        help="Override deterministic execution mode",
    )

    args = parser.parse_args()
    cli = BacktestCLI()
    cli.run(
        args.symbol,
        args.start_date,
        args.end_date,
        args.strategy,
        args.monte_carlo,
        args.seed,
        None if args.deterministic is None else (args.deterministic == "on"),
    )
