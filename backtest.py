"""
BACKTEST ENGINE V3 - Standalone Interface
High-fidelity historical simulation with professional CLI dashboard.
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

# Bot Core Imports
from core.strategy_engine import StrategyEngine
from core.backtester import BacktestEngine
from core.data_fetcher import DataFetcher
from core.connection import MT5Connection
from core.logger import setup_logging

# Load environment variables (MT5 Credentials)
load_dotenv()

class BacktestCLI:
    def __init__(self):
        self.console = Console()
        setup_logging(console=False) # Log to file only during BT
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

    def _create_layout(self) -> Layout:
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
        return layout

    def _get_equity_sparkline(self, curve: List[float], width: int = 40) -> str:
        if not curve or len(curve) < 2: return "No data"
        min_v, max_v = min(curve), max(curve)
        rng = max_v - min_v if max_v != min_v else 1
        chars = " ▂▃▄▅▆▇█"
        spark = ""
        # Resample curve to width
        step = len(curve) / width
        for i in range(width):
            idx = int(i * step)
            if idx >= len(curve): break
            val = curve[idx]
            normalized = int((val - min_v) / rng * (len(chars) - 1))
            spark += chars[normalized]
        return spark

    def run(self, symbol: str, start: str, end: str, strategy_type: str = "SNIPER", compare: bool = False):
        self.console.clear()
        title = "SMC vs SNIPER COMPARISON" if compare else f"{symbol} STANDALONE BACKTESTER"
        self.console.print(Panel(f"[bold green]{title}[/]\nSymbol: {symbol} | Range: {start} to {end}", border_style="bright_blue"))
        
        if not self.connection.connect():
            rprint("[bold red]Critical Error:[/] Could not connect to MT5 for data fetching.")
            return

        try:
            # Data Fetching
            with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), BarColumn(), transient=True) as progress:
                t1 = progress.add_task("[cyan]Fetching M5 Candles...", total=100)
                dt_from = datetime.fromisoformat(start).replace(tzinfo=timezone.utc)
                dt_to = datetime.fromisoformat(end).replace(tzinfo=timezone.utc)
                
                m30 = self.data_fetcher.fetch_candles_range(symbol, "M30", dt_from, dt_to)
                m15 = self.data_fetcher.fetch_candles_range(symbol, "M15", dt_from, dt_to)
                m5 = self.data_fetcher.fetch_candles_range(symbol, "M5", dt_from, dt_to)
                d1 = self.data_fetcher.fetch_candles_range(symbol, "D1", dt_from, dt_to)
                progress.update(t1, completed=100)

            # Symbol Injection
            sym_info = self.data_fetcher.get_symbol_info(symbol)
            if sym_info:
                s_cfg = self.config.setdefault("symbols_config", {}).setdefault(symbol, {})
                s_cfg.update({
                    "point": sym_info["point"], "contract_size": sym_info["contract_size"],
                    "lot_step": sym_info["lot_step"], "min_lot": sym_info["min_lot"],
                    "tick_value": sym_info.get("trade_tick_value", sym_info["contract_size"] * sym_info["point"])
                })

            if compare:
                # Run Sniper
                rprint("[yellow]Phase 1: Simulating SNIPER Strategy...[/]")
                cfg_sniper = {**self.config, "strategy_type": "SNIPER"}
                res_sniper = BacktestEngine(cfg_sniper, StrategyEngine(cfg_sniper)).run(symbol, m30, m15, m5, d1, quiet=True)
                
                # Run SMC
                rprint("[yellow]Phase 2: Simulating SMC Strategy (OB+FVG+LIQ)...[/]")
                cfg_smc = {**self.config, "strategy_type": "SMC"}
                res_smc = BacktestEngine(cfg_smc, StrategyEngine(cfg_smc)).run(symbol, m30, m15, m5, d1, quiet=True)
                
                self._display_comparison(res_sniper, res_smc)
            else:
                cfg = {**self.config, "strategy_type": strategy_type}
                engine = BacktestEngine(cfg, StrategyEngine(cfg))
                rprint(f"[yellow]Simulating {len(m5)} candles using {strategy_type}...[/]")
                results = engine.run(symbol, m30, m15, m5, d1, quiet=False)
                self._display_final_results(results)

        finally:
            self.connection.disconnect()

    def _display_comparison(self, s1: dict, s2: dict):
        """Displays side-by-side performance results."""
        table = Table(title="Strategy Comparison: SNIPER vs SMC", show_header=True, header_style="bold cyan", expand=True)
        table.add_column("Metric")
        table.add_column("SNIPER", justify="right", style="green")
        table.add_column("SMC (OB+FVG)", justify="right", style="magenta")

        metrics = [
            ("Net Profit", "net_profit", "$"),
            ("Profit Factor", "profit_factor", ""),
            ("Win Rate", "win_rate", "%"),
            ("Max Drawdown", "max_drawdown_pct", "%"),
            ("Sharpe Ratio", "sharpe_ratio", ""),
            ("Expectancy", "expectancy", "$"),
            ("Total Trades", "total_trades", "")
        ]

        for label, key, unit in metrics:
            v1, v2 = s1[key], s2[key]
            p1 = f"{unit}{v1}" if unit == "$" else f"{v1}{unit}"
            p2 = f"{unit}{v2}" if unit == "$" else f"{v2}{unit}"
            
            # Highlight the winner
            style1 = "bold green" if v1 > v2 else "white"
            style2 = "bold green" if v2 > v1 else "white"
            if key == "max_drawdown_pct": # lower is better
                style1 = "bold green" if v1 < v2 else "white"
                style2 = "bold green" if v2 < v1 else "white"

            table.add_row(label, f"[{style1}]{p1}[/]", f"[{style2}]{p2}[/]")

        self.console.print(Panel(table, border_style="bright_blue"))
        
        # Show Sparklines
        self.console.print(f"\n[bold cyan]SNIPER Equity Profile:[/]")
        self.console.print(f"[green]{self._get_equity_sparkline(s1['equity_curve'], 80)}[/]")
        
        self.console.print(f"\n[bold magenta]SMC (OB+FVG) Equity Profile:[/]")
        self.console.print(f"[magenta]{self._get_equity_sparkline(s2['equity_curve'], 80)}[/]")

    def _display_final_results(self, res: dict):
        layout = self._create_layout()
        stats_table = Table(show_header=False, box=None)
        stats_table.add_column("Key", style="cyan")
        stats_table.add_column("Value", style="bold yellow")
        stats_table.add_row("Net Profit", f"${res['net_profit']:.2f}")
        stats_table.add_row("Profit Factor", f"{res['profit_factor']:.2f}")
        stats_table.add_row("Win Rate", f"{res['win_rate']:.1f}%")
        stats_table.add_row("Max Drawdown", f"{res['max_drawdown_pct']:.2f}% (${res['max_drawdown_abs']:.2f})")
        stats_table.add_row("Sharpe Ratio", f"{res['sharpe_ratio']:.2f}")
        stats_table.add_row("Expectancy", f"${res['expectancy']:.2f}")
        stats_table.add_row("Total Trades", str(res['total_trades']))
        
        layout["stats"].update(Panel(stats_table, title="[bold white]Performance Scorecard[/]", border_style="green"))
        spark = self._get_equity_sparkline(res.get("equity_curve", []))
        layout["equity"].update(Panel(f"\n\n[bold green]{spark}[/]\n\nBalance: ${res['final_balance']:.2f}", title="[bold white]Equity Growth[/]", border_style="blue"))
        layout["header"].update(Panel(f"[bold cyan]Simulation Complete[/] | Profit: [bold {'green' if res['net_profit'] > 0 else 'red'}]${res['net_profit']:.2f}[/]", border_style="cyan"))
        
        summary = Table(title="Session Wise Performance", show_header=True, header_style="bold magenta", expand=True)
        summary.add_column("Session")
        summary.add_column("Trades", justify="right")
        summary.add_column("Wins (TP)", justify="right", style="green")
        summary.add_column("WR %", justify="right")
        summary.add_column("Profit", justify="right")
        
        trades = res.get("trades", [])
        if trades:
            df = pd.DataFrame(trades)
            for session in ["TOKYO", "LONDON", "LONDON/NY", "NEW_YORK"]:
                sdf = df[df['session'] == session]
                if sdf.empty: continue
                tp_hits = len(sdf[sdf['result'] == "TP"])
                wins = len(sdf[sdf['pnl'] > 0])
                wr = (wins / len(sdf) * 100) if len(sdf) > 0 else 0
                pnl = sdf['pnl'].sum()
                summary.add_row(session, str(len(sdf)), str(tp_hits), f"{wr:.1f}%", f"[bold {'green' if pnl >= 0 else 'red'}]${pnl:.2f}[/]")

        layout["footer"].update(Panel(summary, border_style="dim"))
        self.console.print(layout)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Standalone Professional Backtester")
    parser.add_argument("--symbol", type=str, default="XAUUSDm")
    parser.add_argument("--from", dest="start_date", type=str, required=True, help="YYYY-MM-DD")
    parser.add_argument("--to", dest="end_date", type=str, required=True, help="YYYY-MM-DD")
    parser.add_argument("--strategy", type=str, choices=["SNIPER", "SMC"], default="SNIPER")
    parser.add_argument("--compare", action="store_true", help="Compare SNIPER vs SMC side-by-side")
    args = parser.parse_args()

    cli = BacktestCLI()
    cli.run(args.symbol, args.start_date, args.end_date, strategy_type=args.strategy, compare=args.compare)
