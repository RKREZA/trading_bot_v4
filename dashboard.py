import time
import logging
import os
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional

from rich.console import Console
from rich.text import Text
from rich.live import Live
from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table
from rich.progress import BarColumn, Progress

from core.data.metrics import MetricEngine

class TradingDashboard:
    """
    V6-LIVE: Institutional Production Dashboard.
    Multi-panel, high-fidelity monitoring with real-time risk analytics.
    """

    def __init__(self):
        self.console = Console()
        self.layout = Layout()
        self.prev_price = 0.0
        self.pulse_toggle = True
        self.metric_engine = MetricEngine()
        
        # Initialize Layout structure
        self.layout.split_column(
            Layout(name="header", size=3),
            Layout(name="body"),
            Layout(name="footer", size=8)
        )
        self.layout["body"].split_row(
            Layout(name="left", ratio=3),
            Layout(name="right", ratio=2)
        )
        self.layout["left"].split_column(
            Layout(name="environment", ratio=2),
            Layout(name="risk", ratio=3)
        )
        self.layout["right"].split_column(
            Layout(name="exposure", ratio=2),
            Layout(name="setups", ratio=3)
        )

    def _get_formatted_time(self) -> str:
        now = datetime.now().astimezone()
        offset = now.strftime("%z")
        hours = int(offset[:3])
        tz_str = f"(UTC +{hours}) Dhaka" if hours == 6 else f"(UTC {offset[:3]})"
        return now.strftime(f"%d-%b-%Y %H:%M:%S {tz_str}")

    def _make_header(self, state: dict) -> Panel:
        conn = state.get("connection", {}) or {}
        status_text = "ONLINE" if conn.get("connected") else "OFFLINE"
        status_color = "bold green" if conn.get("connected") else "bold red"
        current_time = self._get_formatted_time()
        
        grid = Table.grid(expand=True)
        grid.add_column(justify="left", ratio=1)
        grid.add_column(justify="center", ratio=1)
        grid.add_column(justify="right", ratio=1)
        grid.add_row(
            Text(current_time, style="cyan"),
            Text("V5-INSIGNIA PRODUCTION (LOCKED)", style="bold cyan"),
            Text(f"STATUS : {status_text}", style=status_color)
        )
        return Panel(grid, style="white on black")

    def _make_environment(self, state: dict) -> Panel:
        symbol = state.get("symbol", "N/A")
        price = state.get("price", 0)
        ask = state.get("ask", 0)
        bid = state.get("bid", 0)
        spread = state.get("spread", 0)
        pips = state.get("pips", 0)
        regime = state.get("regime_type", "N/A")
        volatility = state.get("volatility", "N/A")
        session = state.get("session", "N/A")
        digits = state.get("digits", 2)
        
        price_color = "bright_green" if price > self.prev_price else "bright_red" if price < self.prev_price else "white"
        self.prev_price = price
        
        heartbeat = "●" if self.pulse_toggle else "○"
        self.pulse_toggle = not self.pulse_toggle
        
        table = Table.grid(padding=1)
        table.add_column(style="bold magenta")
        table.add_column()
        table.add_row("SYMBOL", f"[bold yellow]{symbol}[/]")
        table.add_row("PRICE", f"[{price_color}]{ask:,.{digits}f} (ask)[/]")
        table.add_row("SPREAD", f"[cyan]{spread:,.1f} pts ({pips:,.1f} pips)[/]")
        table.add_row("ALIVE", heartbeat)
        table.add_row("SESSION", f"[blue]{session}[/]")
        table.add_row("VOL", f"[yellow]{volatility}[/]")
        
        return Panel(table, title="Environment Monitor", border_style="blue")

    def _make_risk(self, state: dict) -> Panel:
        acc = state.get("account", {}) or {}
        equity_history = state.get("equity_history", [])
        
        balance = acc.get("balance", 0)
        equity = acc.get("equity", 0)
        profit = acc.get("profit", 0)
        
        # Calculate Real-time Metrics
        var_95 = self.metric_engine.calculate_var(equity_history)
        dd = self.metric_engine.calculate_drawdown(equity_history)
        
        pnl_color = "bold green" if profit >= 0 else "bold red"
        
        table = Table.grid(padding=1)
        table.add_column(style="bold white")
        table.add_column(justify="right")
        table.add_row("Balance", f"${balance:,.2f}")
        table.add_row("Equity", f"${equity:,.2f}")
        table.add_row("Live PnL", f"[{pnl_color}]${profit:,.2f}[/]")
        table.add_row("", "") # Spacer
        table.add_row("VaR (95%)", f"[bold yellow]{var_95:.2f}%[/]")
        table.add_row("Current DD", f"[bold red]{dd['current']:.2f}%[/]")
        table.add_row("Max DD", f"[red]{dd['max']:.2f}%[/]")
        
        return Panel(table, title="Institutional Risk & Equity", border_style="red")

    def _make_exposure(self, state: dict) -> Panel:
        positions = state.get("positions", [])
        heatmap = self.metric_engine.get_exposure_heatmap(positions)
        
        table = Table(box=None, show_header=True, header_style="bold blue")
        table.add_column("Basket", ratio=1)
        table.add_column("Net Lots", justify="right", ratio=1)
        
        for basket, value in heatmap.items():
            color = "green" if value > 0 else "red" if value < 0 else "white"
            table.add_row(basket, f"[{color}]{value:+.2f}[/]")
            
        if not heatmap:
            return Panel(Text("N/A", style="dim center"), title="Exposure Heatmap", border_style="cyan")
            
        return Panel(table, title="Exposure Heatmap", border_style="cyan")

    def _make_setups(self, state: dict) -> Panel:
        setups = state.get("setups", {}) or {}
        table = Table(box=None, show_header=True, header_style="bold yellow")
        table.add_column("Strategy", ratio=2)
        table.add_column("Bias", ratio=1)
        table.add_column("Grade", ratio=1)
        
        for sid, s_info in setups.items():
            signal = s_info.get("signal", "NONE")
            sig_dir = signal.direction if hasattr(signal, "direction") else str(signal)
            dfs = s_info.get("fidelity", 1.0)
            
            color = "green" if sig_dir != "NONE" else "white"
            clean_name = sid.replace("_v5", "").replace("_", " ").title()
            table.add_row(clean_name, f"[{color}]{sig_dir}[/]", f"{dfs*100:.0f}%")
            
        return Panel(table, title="Institutional Setups", border_style="yellow")

    def _make_footer(self, state: dict) -> Any:
        logs = state.get("logs", [])
        news = state.get("news_list", [])
        
        grid = Table.grid(expand=True)
        grid.add_column(ratio=2) # Logs
        grid.add_column(ratio=1) # News
        
        # Logs sub-panel
        log_text = Text()
        for log in logs[-5:]:
            color = "cyan" if "[ANALYSIS]" in log else "green" if "[TRADE]" in log else "red" if "[ERROR]" in log else "white"
            log_text.append(f"> {log}\n", style=color)
            
        # News sub-panel
        news_text = Text()
        for ev in news[:3]:
            news_text.append(f"[{ev.get('time', 'N/A')}] {ev.get('title', 'Event')}\n", style="magenta")

        grid.add_row(
            Panel(log_text, title="Live Analysis", border_style="dim"),
            Panel(news_text, title="Economic Calendar", border_style="magenta")
        )
        return grid

    def update(self, state: dict) -> Layout:
        """Update Layout with fresh state objects."""
        self.layout["header"].update(self._make_header(state))
        self.layout["environment"].update(self._make_environment(state))
        self.layout["risk"].update(self._make_risk(state))
        self.layout["exposure"].update(self._make_exposure(state))
        self.layout["setups"].update(self._make_setups(state))
        self.layout["footer"].update(self._make_footer(state))
        return self.layout

def start_dashboard(layout):
    return Live(layout, auto_refresh=False, screen=True)
