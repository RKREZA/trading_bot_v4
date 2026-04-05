"""
BACKTEST ENGINE V3 - Multi-Strategy Standalone Interface
High-fidelity historical simulation with per-strategy performance reporting.

Usage:
    # Single strategy:
    python backtest.py --from 2026-01-01 --to 2026-03-31 --strategy sniper_v1

    # All enabled strategies:
    python backtest.py --from 2026-01-01 --to 2026-03-31 --all

    # Compare strategies side-by-side:
    python backtest.py --from 2026-01-01 --to 2026-03-31 --compare

    # Legacy mode (backward compatible):
    python backtest.py --from 2026-01-01 --to 2026-03-31 --legacy SNIPER
"""

import argparse
import json
import os
import sys
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
from typing import Optional, List, Dict, Any

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.live import Live
from rich.layout import Layout
from rich import print as rprint
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn

from dotenv import load_dotenv

from core.strategy_engine import StrategyEngine
from core.backtester import BacktestEngine, MultiStrategyBacktestEngine
from core.data_fetcher import DataFetcher
from core.connection import MT5Connection
from core.logger import setup_logging
from core.walk_forward import WalkForwardValidator
from core.monte_carlo import MonteCarlo
from strategies import create_strategy, STRATEGY_REGISTRY

load_dotenv()


class BacktestCLI:
    def __init__(self):
        self.console = Console()
        setup_logging(console=False)
        self.config = self._load_config("config.json")
        self.data_fetcher = DataFetcher()
        self.connection = MT5Connection()
        self.connection.config = self.config

    def _load_config(self, path: str) -> dict:
        if not os.path.exists(path):
            rprint(f"[bold red]Error:[/] {path} not found.")
            sys.exit(1)
        with open(path, "r") as f:
            return json.load(f)

    def _get_equity_sparkline(self, curve: List[float], width: int = 40) -> str:
        if not curve or len(curve) < 2:
            return "No data"
        min_v, max_v = min(curve), max(curve)
        rng = max_v - min_v if max_v != min_v else 1
        chars = " ▂▃▄▅▆▇█"
        spark = ""
        step = len(curve) / width
        for i in range(width):
            idx = int(i * step)
            if idx >= len(curve):
                break
            val = curve[idx]
            normalized = int((val - min_v) / rng * (len(chars) - 1))
            spark += chars[normalized]
        return spark

    def _build_strategies(self, strategy_filter: str = None) -> list:
        """Build strategy instances from config or defaults."""
        strategies_cfg = self.config.get("strategies", [])

        if not strategies_cfg:
            # Legacy: no strategies array — build from strategy_type
            default_type = self.config.get("strategy_type", "SNIPER")
            return [create_strategy(
                f"{default_type.lower()}_v1", default_type, self.config
            )]

        strategies = []
        for s_cfg in strategies_cfg:
            sid = s_cfg["id"]
            stype = s_cfg["type"]
            enabled = s_cfg.get("enabled", True)

            if strategy_filter and sid != strategy_filter:
                continue
            if not strategy_filter and not enabled:
                continue

            # Merge global config with strategy-specific config
            merged = dict(self.config)
            merged.update(s_cfg)
            strategies.append(create_strategy(sid, stype, merged))

        return strategies

    def run(self, symbol: str, start: str, end: str,
            strategy_filter: str = None, run_all: bool = False,
            compare: bool = False, legacy_type: str = None,
            walk_forward: bool = False, monte_carlo: bool = False):
        self.console.clear()

        if not self.connection.connect():
            rprint("[bold red]Critical Error:[/] Could not connect to MT5 for data fetching.")
            return

        try:
            # Data Fetching
            with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
                          BarColumn(), transient=True) as progress:
                t1 = progress.add_task("[cyan]Fetching candles...", total=100)
                dt_from = datetime.fromisoformat(start).replace(tzinfo=timezone.utc)
                dt_to = datetime.fromisoformat(end).replace(tzinfo=timezone.utc)

                m15 = self.data_fetcher.fetch_candles_range(symbol, "M15", dt_from, dt_to)
                m5 = self.data_fetcher.fetch_candles_range(symbol, "M5", dt_from, dt_to)
                d1 = self.data_fetcher.fetch_candles_range(symbol, "D1", dt_from, dt_to)
                h1 = self.data_fetcher.fetch_candles_range(symbol, "H1", dt_from, dt_to)
                progress.update(t1, completed=100)

                # [INSTITUTIONAL] Data Integrity Audit
                for tf, data in [("M5", m5), ("M15", m15), ("H1", h1)]:
                    report = self.data_fetcher.validate_data_integrity(data, tf)
                    if report["status"] != "OK":
                        rprint(f"[bold yellow]DATA INTEGRITY {tf}:[/] {report['status']} | Missing: {report['missing_total']} candles | Gaps: {report['gap_count']}")
                        if report["status"] == "CRITICAL":
                            rprint("[bold red]CAUTION:[/] Large data gaps detected. Results may be unreliable.")

            # Symbol Injection
            sym_info = self.data_fetcher.get_symbol_info(symbol)
            if sym_info:
                s_cfg = self.config.setdefault("symbols_config", {}).setdefault(symbol, {})
                s_cfg.update({
                    "point": sym_info["point"],
                    "contract_size": sym_info["contract_size"],
                    "lot_step": sym_info["lot_step"],
                    "min_lot": sym_info["min_lot"],
                    "stops_level": sym_info.get("stops_level", 0),
                    "tick_value": sym_info.get("trade_tick_value",
                                               sym_info["contract_size"] * sym_info["point"])
                })

            # ── Legacy mode ────────────────────────────────
            if legacy_type:
                self.console.print(Panel(
                    f"[bold green]LEGACY BACKTEST[/]\nSymbol: {symbol} | Strategy: {legacy_type} | Range: {start} to {end}",
                    border_style="bright_blue"
                ))
                cfg = {**self.config, "strategy_type": legacy_type}
                engine = BacktestEngine(cfg, StrategyEngine(cfg))
                rprint(f"[yellow]Simulating {len(m5)} candles using {legacy_type}...[/]")
                results = engine.run(symbol, h1, m15, m5, d1, quiet=False, data_fetcher=self.data_fetcher)
                self._display_single_result(results)
                return

            # ── Multi-strategy mode ────────────────────────
            if compare or run_all:
                strategies = self._build_strategies()
            elif strategy_filter:
                strategies = self._build_strategies(strategy_filter)
            else:
                strategies = self._build_strategies()

            if not strategies:
                rprint("[bold red]No strategies found or all disabled.[/]")
                return

            names = [s.strategy_id for s in strategies]
            self.console.print(Panel(
                f"[bold green]MULTI-STRATEGY BACKTEST[/]\n"
                f"Symbol: {symbol} | Strategies: {', '.join(names)} | Range: {start} to {end}",
                border_style="bright_blue"
            ))

            engine = MultiStrategyBacktestEngine(self.config, strategies)
            rprint(f"[yellow]Simulating {len(m5)} candles across {len(strategies)} strategies...[/]")
            results = engine.run(symbol, h1, m15, m5, d1, quiet=False,
                                 data_fetcher=self.data_fetcher, monte_carlo=monte_carlo)

            # Walk-Forward validation
            if walk_forward:
                wf_cfg = self.config.get("backtest", {}).get("walk_forward", {})
                is_pct = float(wf_cfg.get("in_sample_pct", 0.70))
                wf = WalkForwardValidator(
                    self.config, strategies, MultiStrategyBacktestEngine, is_pct=is_pct
                )
                wf_result = wf.run(symbol, h1, m15, m5, d1, quiet=False)
                results["walk_forward"] = wf_result

            if compare and len(results) > 1:
                self._display_comparison(results)
            elif len(results) == 1:
                key = list(results.keys())[0]
                self._display_single_result(results[key])
            else:
                self._display_multi_results(results)

        finally:
            self.connection.disconnect()

    def _display_multi_results(self, results: Dict[str, dict]):
        """Display per-strategy results in a unified table."""
        # Strategy comparison table
        table = Table(
            title="Multi-Strategy Backtest Results",
            show_header=True, header_style="bold cyan", expand=True
        )
        table.add_column("Metric")

        strategy_ids = [k for k in results.keys() if k != "portfolio"]
        for sid in strategy_ids:
            table.add_column(sid, justify="right", style="green")

        if "portfolio" in results:
            table.add_column("PORTFOLIO", justify="right", style="bold yellow")
            strategy_ids_with_portfolio = strategy_ids + ["portfolio"]
        else:
            strategy_ids_with_portfolio = strategy_ids

        metrics = [
            ("Net Profit", "net_profit", "$"),
            ("Profit Factor", "profit_factor", ""),
            ("Win Rate", "win_rate", "%"),
            ("Max Drawdown", "max_drawdown_pct", "%"),
            ("Sharpe Ratio", "sharpe_ratio", ""),
            ("Expectancy", "expectancy", "$"),
            ("Total Trades", "total_trades", ""),
        ]

        for label, key, unit in metrics:
            row = [label]
            values = []
            for sid in strategy_ids_with_portfolio:
                val = results.get(sid, {}).get(key, 0)
                values.append(val)
                if unit == "$":
                    row.append(f"${val}")
                elif unit == "%":
                    row.append(f"{val}%")
                else:
                    row.append(str(val))
            table.add_row(*row)

        self.console.print(Panel(table, border_style="bright_blue"))

        # Equity sparklines
        for sid in strategy_ids:
            eq = results.get(sid, {}).get("equity_curve", [])
            color = "green" if results.get(sid, {}).get("net_profit", 0) >= 0 else "red"
            self.console.print(f"\n[bold cyan]{sid} Equity:[/]")
            self.console.print(f"[{color}]{self._get_equity_sparkline(eq, 80)}[/{color}]")

        # Per-strategy session breakdown
        for sid in strategy_ids:
            trades = results.get(sid, {}).get("trades", [])
            if trades:
                self._display_session_breakdown(sid, trades)

    def _display_comparison(self, results: Dict[str, dict]):
        """Side-by-side comparison (same as _display_multi_results but with highlighting)."""
        self._display_multi_results(results)

    def _display_single_result(self, res: dict):
        """Display results for a single strategy."""
        layout = Layout()
        layout.split(
            Layout(name="header", size=3),
            Layout(name="main", ratio=1),
            Layout(name="footer", size=10)
        )
        layout["main"].split_row(
            Layout(name="stats", ratio=1),
            Layout(name="equity", ratio=1)
        )

        stats_table = Table(show_header=False, box=None)
        stats_table.add_column("Key", style="cyan")
        stats_table.add_column("Value", style="bold yellow")
        stats_table.add_row("Strategy", res.get("strategy_id", "legacy"))
        stats_table.add_row("Net Profit", f"${res['net_profit']:.2f}")
        stats_table.add_row("Profit Factor", f"{res['profit_factor']:.2f}")
        stats_table.add_row("Win Rate", f"{res['win_rate']:.1f}%")
        stats_table.add_row("Max Drawdown", f"{res['max_drawdown_pct']:.2f}% (${res.get('max_drawdown_abs', 0):.2f})")
        stats_table.add_row("Sharpe Ratio", f"{res.get('sharpe_ratio', 0):.2f}")
        stats_table.add_row("Expectancy", f"${res['expectancy']:.2f}")
        stats_table.add_row("Total Trades", str(res['total_trades']))

        layout["stats"].update(Panel(stats_table, title="[bold white]Performance Scorecard[/]", border_style="green"))
        spark = self._get_equity_sparkline(res.get("equity_curve", []))
        layout["equity"].update(Panel(
            f"\n\n[bold green]{spark}[/]\n\nBalance: ${res.get('final_balance', 0):.2f}",
            title="[bold white]Equity Growth[/]", border_style="blue"
        ))
        color = 'green' if res['net_profit'] > 0 else 'red'
        layout["header"].update(Panel(
            f"[bold cyan]Simulation Complete[/] | Profit: [bold {color}]${res['net_profit']:.2f}[/]",
            border_style="cyan"
        ))

        # Session breakdown
        trades = res.get("trades", [])
        summary = Table(title="Session Wise Performance", show_header=True, header_style="bold magenta", expand=True)
        summary.add_column("Session")
        summary.add_column("Trades", justify="right")
        summary.add_column("Wins (TP)", justify="right", style="green")
        summary.add_column("WR %", justify="right")
        summary.add_column("Profit", justify="right")

        if trades:
            df = pd.DataFrame(trades)
            for session in ["TOKYO", "LONDON", "LONDON/NY", "NEW_YORK"]:
                sdf = df[df['session'] == session]
                if sdf.empty:
                    continue
                tp_hits = len(sdf[sdf['result'] == "TP"])
                wins = len(sdf[sdf['pnl'] > 0])
                wr = (wins / len(sdf) * 100) if len(sdf) > 0 else 0
                pnl = sdf['pnl'].sum()
                c = 'green' if pnl >= 0 else 'red'
                summary.add_row(session, str(len(sdf)), str(tp_hits), f"{wr:.1f}%",
                                f"[bold {c}]${pnl:.2f}[/]")

        layout["footer"].update(Panel(summary, border_style="dim"))
        self.console.print(layout)

    def _display_session_breakdown(self, strategy_id: str, trades: list):
        """Display session breakdown for a specific strategy."""
        df = pd.DataFrame(trades)
        table = Table(title=f"{strategy_id} Session Breakdown", show_header=True,
                      header_style="bold magenta", expand=True)
        table.add_column("Session")
        table.add_column("Trades", justify="right")
        table.add_column("Wins", justify="right", style="green")
        table.add_column("WR %", justify="right")
        table.add_column("Profit", justify="right")

        for session in ["TOKYO", "LONDON", "LONDON/NY", "NEW_YORK"]:
            sdf = df[df['session'] == session]
            if sdf.empty:
                continue
            wins = len(sdf[sdf['pnl'] > 0])
            wr = (wins / len(sdf) * 100) if len(sdf) > 0 else 0
            pnl = sdf['pnl'].sum()
            c = 'green' if pnl >= 0 else 'red'
            table.add_row(session, str(len(sdf)), str(wins), f"{wr:.1f}%", f"[bold {c}]${pnl:.2f}[/]")

        self.console.print(table)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-Strategy Professional Backtester")
    parser.add_argument("--symbol", type=str, default="XAUUSDm")
    parser.add_argument("--from", dest="start_date", type=str, required=True, help="YYYY-MM-DD")
    parser.add_argument("--to", dest="end_date", type=str, required=True, help="YYYY-MM-DD")

    # Multi-strategy options
    parser.add_argument("--strategy", type=str, default=None,
                        help="Run a specific strategy by ID (e.g. 'sniper_v1')")
    parser.add_argument("--all", dest="run_all", action="store_true",
                        help="Run all enabled strategies simultaneously")
    parser.add_argument("--compare", action="store_true",
                        help="Compare all enabled strategies side-by-side")
    parser.add_argument("--walk-forward", dest="walk_forward", action="store_true",
                        help="Run IS/OOS walk-forward validation (70/30 split)")
    parser.add_argument("--monte-carlo", dest="monte_carlo", action="store_true",
                        help="Run 2000-path Monte Carlo robustness simulation")

    # Legacy compatibility
    parser.add_argument("--legacy", type=str, choices=["SNIPER", "SMC"], default=None,
                        help="Run legacy single-engine backtest (backward compatible)")

    args = parser.parse_args()
    cli = BacktestCLI()
    cli.run(
        args.symbol, args.start_date, args.end_date,
        strategy_filter=args.strategy,
        run_all=args.run_all,
        compare=args.compare,
        legacy_type=args.legacy,
        walk_forward=args.walk_forward,
        monte_carlo=args.monte_carlo,
    )
