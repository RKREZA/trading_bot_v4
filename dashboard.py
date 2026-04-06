import time
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional

from rich.console import Console
from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.live import Live

class TradingDashboard:
    """
    V4-ULTRA Professional Live Monitor.
    Standardized 'State' propagation for multi-strategy institutional trading.
    """

    def __init__(self):
        self.console = Console()
        self.layout = Layout()
        self._init_layout()
        
    def _init_layout(self):
        """Build the institutional grid structure."""
        self.layout.split(
            Layout(name="header", size=3),
            Layout(name="main", ratio=1),
            Layout(name="footer", size=10),
        )
        self.layout["main"].split_row(
            Layout(name="left", ratio=1),
            Layout(name="right", ratio=2),
        )
        self.layout["left"].split_column(
            Layout(name="account", ratio=1),
            Layout(name="market", size=9),
            Layout(name="news", ratio=1),
            Layout(name="setup", ratio=1),
        )
        
        # POPULATE FALLBACK PANELS (To prevent Skeleton View)
        self.layout["header"].update(Panel("[bold red]V4-ULTRA INITIALIZING...[/]", style="white on blue"))
        self.layout["account"].update(Panel("[dim white]Waiting for account telemetry...[/]", title="Account Integrity", border_style="green"))
        self.layout["market"].update(Panel("[dim white]Synchronizing with MT5 Server...[/]", title="Environment Monitor", border_style="cyan"))
        self.layout["news"].update(Panel("[dim white]Connecting to Economic Cloud...[/]", title="Economic Calendar", border_style="magenta"))
        self.layout["setup"].update(Panel("[dim white]Loading strategy micro-services...[/]", title="Institutional Setup", border_style="yellow"))
        self.layout["right"].update(Panel("[dim white]Scanning portfolio for active positions...[/]", title="Portfolio", border_style="blue"))
        self.layout["footer"].update(Panel("[dim white]Establishing secure analysis heartbeat...[/]", title="Live Analysis", border_style="yellow"))

    def generate_header(self, state: dict) -> Panel:
        """Top-tier information bar (Modern Minimalist)."""
        conn = state.get("connection", {}) or {}
        status = "[bold green]ONLINE[/]" if conn.get("connected") else "[bold red]OFFLINE[/]"
        server_time = conn.get("server_time", "00:00:00")
        
        grid = Table.grid(expand=True)
        grid.add_column(justify="left", ratio=1)
        grid.add_column(justify="center", ratio=1)
        grid.add_column(justify="right", ratio=1)
        
        grid.add_row(
            f"  V4-ULTRA Command Center",
            f"[bold blue]MT5 Server Time: {server_time}[/]",
            f"Status: {status}  "
        )
        return Panel(grid, box=None) # NO BACKGROUND

    def generate_account_panel(self, state: dict) -> Panel:
        """Account health metrics."""
        account = state.get("account", {}) or {}
        login = account.get("login", "N/A")
        server = account.get("server", "N/A")
        
        table = Table(show_header=False, box=None, expand=True)
        table.add_column("METRIC", ratio=1)
        table.add_column("DETAILS", justify="right", ratio=2)
        
        table.add_row("ACCOUNT ID", f"[bold white]{login}[/]")
        table.add_row("SERVER", f"[dim white]{server}[/]")
        table.add_row("BALANCE", f"[bold green]${account.get('balance', 0):,.2f}[/]")
        table.add_row("EQUITY", f"[bold white]${account.get('equity', 0):,.2f}[/]")
        
        profit = account.get('profit', 0)
        pnl_color = "green" if profit >= 0 else "red"
        table.add_row("FLOATING P/L", f"[{pnl_color}]${profit:,.2f}[/]")
        
        table.add_row("LEVERAGE", f"[bold yellow]1:{account.get('leverage', 0)}[/]")
        table.add_row("MARGIN", f"[bold red]${account.get('margin', 0):,.2f}[/]")
        table.add_row("FREE MARGIN", f"[bold cyan]${account.get('margin_free', 0):,.2f}[/]")
        
        return Panel(table, title="[bold green]Account Integrity[/]", border_style="green")

    def generate_market_panel(self, state: dict) -> Panel:
        """Dual-metric environment monitor."""
        table = Table(show_header=False, box=None, expand=True)
        table.add_column("METRIC", ratio=1)
        table.add_column("DETAILS", justify="right", ratio=2)

        table.add_row("ACTIVE SYMBOL", f"[bold white]{state.get('symbol', 'N/A')}[/]")
        table.add_row("LIVE PRICE", f"[bold white]{state.get('price', 0):,.2f}[/]")
        table.add_row("CURRENT SPREAD", f"[bold white]{state.get('spread', 0):.1f} pts[/]")
        
        reg_type = state.get('regime_type', 'UNCERTAIN')
        reg_color = "cyan" if reg_type == "TRENDING" else "yellow" if reg_type == "RANGING" else "white"
        table.add_row("MARKET TYPE", f"[{reg_color}]{reg_type}[/]")
        
        vol_status = state.get('volatility', 'NORMAL')
        vol_color = "red" if vol_status == "HIGH" else "green" if vol_status == "LOW" else "white"
        table.add_row("VOLATILITY", f"[{vol_color}]{vol_status}[/]")
        
        table.add_row("TRADING SESSION", f"[bold white]{state.get('session', 'GLOBAL')}[/]")
        
        return Panel(table, title="[bold cyan]Environment Monitor[/]", border_style="cyan")

    def generate_news_panel(self, state: dict) -> Panel:
        """Economic Calendar."""
        news_list = state.get("news_list", []) or []
        table = Table(show_header=False, box=None, padding=(0,1), expand=True)
        table.add_column("Impact", justify="left", width=2)
        table.add_column("Event", justify="left", ratio=1, overflow="ellipsis", no_wrap=True)
        table.add_column("Time", justify="right")
        
        if not news_list:
            return Panel("[dim white]No high-impact news[/]", title="[bold magenta]Economic Calendar[/]", border_style="magenta")
            
        for ev in news_list[:5]:
            impact = "[bold red]![/]" if ev.get("is_active") else "[red]![/]"
            table.add_row(impact, f"[white]{ev.get('title')}[/]", f"[magenta]{ev.get('time')}[/]")
            
        return Panel(table, title="[bold magenta]Economic Calendar[/]", border_style="magenta")

    def generate_setup_panel(self, state: dict) -> Panel:
        """Strategy Confluence Reasons."""
        setups = state.get("setups", {}) or {}
        table = Table(show_header=False, box=None, expand=True)
        table.add_column("Strat", justify="left", style="bold cyan")
        table.add_column("Setup", justify="left")
        
        if not setups:
            return Panel("[dim white]Scanning strategy logic...[/]", title="[bold yellow]Institutional Setup[/]", border_style="yellow")
            
        for sid, reasons in setups.items():
            if not reasons: continue
            reason_str = "\n".join([f"• {r}" for r in reasons])
            table.add_row(sid, f"[dim white]{reason_str}[/]")
            
        return Panel(table, title="[bold yellow]Institutional Setup[/]", border_style="yellow")

    def generate_trade_table(self, state: dict) -> Panel:
        """Open Portfolio positions."""
        positions = state.get("positions", []) or []
        table = Table(box=None, header_style="bold blue", expand=True)
        table.add_column("Symbol", justify="left")
        table.add_column("Type", justify="center")
        table.add_column("Lots", justify="right")
        table.add_column("Price", justify="right")
        table.add_column("SL", justify="right")
        table.add_column("TP", justify="right")
        table.add_column("PnL ($)", justify="right")
        
        for pos in positions:
            pnl_color = "green" if pos.get('profit', 0) >= 0 else "red"
            table.add_row(
                pos.get('symbol', '???'),
                pos.get('type_text', '???'),
                f"{pos.get('volume', 0):.2f}",
                f"{pos.get('price_open', 0):.5f}",
                f"{pos.get('sl', 0):.5f}",
                f"{pos.get('tp', 0):.5f}",
                f"[{pnl_color}]${pos.get('profit', 0):,.2f}[/]"
            )
            
        return Panel(table, title="Active Multi-Service Portfolio", border_style="blue")

    def generate_log_footer(self, state: dict) -> Panel:
        """Live Analysis scrolling logs."""
        logs = state.get("logs", []) or []
        log_text = Text()
        for log in logs[-8:]:
            color = "white"
            if "[ANALYSIS]" in log: color = "cyan"
            if "TRADE" in log: color = "green"
            if "ERROR" in log: color = "red"
            log_text.append(f"• {log}\n", style=color)
            
        return Panel(log_text, title="[bold yellow]Live Analysis[/]", border_style="yellow")

    def update(self, state: dict) -> Layout:
        """Pulse the entire dashboard tree."""
        self.layout["header"].update(self.generate_header(state))
        self.layout["account"].update(self.generate_account_panel(state))
        self.layout["market"].update(self.generate_market_panel(state))
        self.layout["news"].update(self.generate_news_panel(state))
        self.layout["setup"].update(self.generate_setup_panel(state))
        self.layout["right"].update(self.generate_trade_table(state))
        self.layout["footer"].update(self.generate_log_footer(state))
        return self.layout

def start_dashboard(layout: Layout):
    """Factory for the live renderer."""
    return Live(layout, refresh_per_second=2, screen=False)
