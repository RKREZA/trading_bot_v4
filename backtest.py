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
from core import SourceHandler, MT5Connection, PerformanceTracker, setup_logging
from backtesting import PortfolioBacktester, MonteCarloSimulator, StressTester, WalkForwardValidator
from strategies import create_strategy, STRATEGY_REGISTRY

load_dotenv()

class BacktestCLI:
    def __init__(self):
        self.console = Console()
        self.config = self._load_config("config.json")
        self.data_fetcher = SourceHandler()
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
        run_walk_forward: bool = False,
        seed: Optional[int] = None,
        deterministic: Optional[bool] = None,
    ):
        setup_logging()
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
        rprint(Panel(f"[bold cyan]V4-PRO Institutional Benchmark[/]\nSymbol: {symbol} | Range: {start} to {end}", border_style="bright_blue"))
        
        with self.console.status("[bold green]Loading Bid/Ask Execution Data...") as status:
            m1 = self.data_fetcher.fetch_candles_range(symbol, "M1", dt_from, dt_to)
            m5 = self.data_fetcher.fetch_candles_range(symbol, "M5", dt_from, dt_to)
            m15 = self.data_fetcher.fetch_candles_range(symbol, "M15", dt_from, dt_to)
            h1 = self.data_fetcher.fetch_candles_range(symbol, "H1", dt_from, dt_to)

        if len(m5) < 100:
            rprint("[bold red]Error:[/] Insufficient data for benchmark.")
            return

        # 2. Strategy Loader (Step 9)
        strategies = self._build_strategies(strategy_filter, symbol=symbol)
        if not strategies:
            rprint(f"[bold red]Error:[/] No strategies available for benchmark.")
            return

        # 3. Validation: Walk-Forward (Phase 15/16)
        if run_walk_forward:
            self._run_walk_forward(symbol, strategies, {"M5": m5, "M1": m1, "M15": m15, "H1": h1})

        # 4. Core Portfolio Simulation
        runtime_config = dict(self.config)
        runtime_backtest = dict(runtime_config.get("backtest", {}))
        if seed is not None:
            runtime_backtest["random_seed"] = int(seed)
        if deterministic is not None:
            runtime_backtest["deterministic"] = bool(deterministic)
        runtime_config["backtest"] = runtime_backtest

        backtester = PortfolioBacktester(runtime_config)
        rprint(f"[green]Executing Multi-Strategy Auction Simulation...[/]")
        history, equity_history = backtester.run(symbol, strategies, m5, h1, m15, m1)
        
        # 5. Performance Attribution
        partition_initial = float(self.config.get("backtest", {}).get("initial_balance_per_strategy", 1000.0))
        total_initial = len(strategies) * partition_initial
        
        # Aggregate Portfolio Equity
        portfolio_equity = []
        if equity_history:
            import pandas as pd
            eq_df = pd.DataFrame(equity_history)
            portfolio_equity = eq_df.groupby('time')['equity'].sum().tolist()

        stats = PerformanceTracker.calculate_metrics(history, total_initial, equity_curve=portfolio_equity)
        strat_stats = PerformanceTracker.calculate_per_strategy(history, partition_initial)
        session_stats = PerformanceTracker.calculate_per_session(history, total_initial)
        
        full_results = {
            "portfolio": stats,
            "strategies": strat_stats,
            "sessions": session_stats
        }

        # 6. Audit Pack Generation (Step 14)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        session_dir = f"backtest_results/session_{timestamp}"
        PerformanceTracker.save_audit_pack(history, full_results, session_dir)
        
        # 7. Dashboard Display
        dashboard = PerformanceTracker.generate_professional_dashboard(full_results)
        rprint(f"\n[bold white]{dashboard}[/]")
        rprint(f"\n[bold green]Institutional Audit Pack Persistence: {session_dir}[/]")

        # 8. Robustness: Monte Carlo (Step 15/16)
        if run_monte_carlo and history:
            self._run_monte_carlo(history)
            
        if self.config.get("backtest", {}).get("run_stress_test", False):
            self._run_stress_test(symbol, strategies, {"M1": m1, "M5": m5, "M15": m15, "H1": h1})

    def _run_walk_forward(self, symbol, strategies, data):
        rprint("\n[bold yellow]Validation Audit: Running Walk-Forward Optimization (Rolling Windows)...[/]")
        wfo = WalkForwardValidator(self.config)
        results = wfo.run_validation(symbol, strategies, data)
        wfo.summarize_wfo_results(results)

    def _build_strategies(self, strategy_filter: str = None, symbol: str = None):
        strategies = []
        for st_type, st_class in STRATEGY_REGISTRY.items():
            sid = f"{st_type.lower()}_v4"
            if strategy_filter and sid != strategy_filter and st_type != strategy_filter.upper():
                continue
            try:
                strategy_obj = st_class(sid, config=self.config)
                if symbol and not strategy_obj.is_symbol_allowed(symbol):
                    continue
                strategies.append(strategy_obj)
            except Exception as e:
                rprint(f"[bold yellow]Warning:[/] Failed to load {sid}: {e}")
        return strategies

    def _run_monte_carlo(self, history):
        rprint("\n[bold yellow]Robustness Audit: Running Monte Carlo Simulation (2000 paths)...[/]")
        mc = MonteCarloSimulator(iterations=2000)
        res = mc.run(history)
        score = float(res.get("robustness_score", 0))
        is_robust = score >= 40.0
        
        mc_table = Table(title="Monte Carlo Institutional Robustness Report", show_header=False, padding=(0, 2))
        for k, v in res.items():
            if k == "robustness_score":
                score_color = "green" if is_robust else "red"
                mc_table.add_row(k.replace("_", " ").title(), f"[{score_color}]{v}/100[/]")
            else:
                mc_table.add_row(k.replace("_", " ").title(), str(v))
        
        if not is_robust:
            rprint("\n[bold red]!!! HARD CONSTRAINT VIOLATION: CURVE-FITTING DETECTED !!![/]")
        self.console.print(Panel(mc_table, border_style="yellow" if is_robust else "red", title="Robustness Audit"))
        return is_robust

    def _run_stress_test(self, symbol, strategies, data):
        rprint("\n[bold magenta]Worst Case Scenario Hunt: Running Stress Test Suite...[/]")
        tester = StressTester(self.config)
        results = tester.run_stress_test(symbol, strategies, data)
        
        s_table = Table(title="Execution Stress Tolerance Analysis", header_style="bold magenta")
        s_table.add_column("Scenario", style="bold")
        s_table.add_column("Profit ($)", justify="right")
        s_table.add_column("Retention (%)", justify="right")
        s_table.add_column("Result", justify="center")

        for name, res in results.items():
            m = res["metrics"]
            status = "[bold green]PASS[/]" if m.get("net_profit", 0) > 0 else "[bold red]FAIL[/]"
            s_table.add_row(name.replace("_", " ").upper(), f"${m.get('net_profit', 0):,.2f}", f"{res['profit_retention']:.1f}%", status)
        self.console.print(s_table)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="V4 Institutional Backtester CLI")
    parser.add_argument("--symbol", type=str, default="XAUUSDm")
    parser.add_argument("--from", dest="start_date", type=str, required=True)
    parser.add_argument("--to", dest="end_date", type=str, required=True)
    parser.add_argument("--strategy", type=str, default=None)
    parser.add_argument("--monte-carlo", action="store_true")
    parser.add_argument("--walk-forward", action="store_true")
    parser.add_argument("--stress-test", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--deterministic", choices=["on", "off"], default=None)

    args = parser.parse_args()
    cli = BacktestCLI()
    if args.stress_test:
        cli.config.setdefault("backtest", {})["run_stress_test"] = True

    cli.run(args.symbol, args.start_date, args.end_date, args.strategy, args.monte_carlo, args.walk_forward, args.seed, 
            None if args.deterministic is None else (args.deterministic == "on"))
