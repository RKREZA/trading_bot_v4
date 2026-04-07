import time
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional

from rich.console import Console
from rich.text import Text
from rich.live import Live

class TradingDashboard:
    """
    V4-ULTRA Professional Live Monitor - Triple-aligned header and full-width layout.
    """

    def __init__(self):
        self.console = Console()
        self.layout = Text("V4-ULTRA INITIALIZING...")
        
    def _get_ordinal_suffix(self, day: int) -> str:
        """Calculate ordinal suffix for day of month."""
        if 11 <= day <= 13:
            return "th"
        return {1: "st", 2: "nd", 3: "rd"}.get(day % 10, "th")

    def _get_formatted_time(self) -> str:
        """Return date in format like: 8th April 2026 - 1:44:40PM"""
        now = datetime.now()
        day = now.day
        suffix = self._get_ordinal_suffix(day)
        date_part = f"{day}{suffix} {now.strftime('%B %Y')}"
        time_part = now.strftime("%I:%M:%S%p")
        return f"{date_part} - {time_part}"

    def update(self, state: dict) -> Text:
        """Pulse the entire dashboard with triple-aligned header."""
        conn = state.get("connection", {}) or {}
        acc = state.get("account", {}) or {}
        width = self.console.width
        sep = "=" * width
        
        # Account Info Extraction
        login = acc.get("login", "N/A")
        server = acc.get("server", "N/A")
        balance = acc.get("balance", 0)
        equity = acc.get("equity", 0)
        free_margin = acc.get("margin_free", acc.get("free_margin", 0))
        leverage = acc.get("leverage", 0)
        profit = acc.get("profit", 0)
        
        # Environment Info Extraction
        symbol = state.get("symbol", "N/A")
        price = state.get("price", 0)
        spread = state.get("spread", 0)
        regime = state.get("regime_type", "N/A")
        volatility = state.get("volatility", "N/A")
        session = state.get("session", "N/A")
        
        status_text = "ONLINE" if conn.get("connected") else "OFFLINE"
        status_color = "bold green" if conn.get("connected") else "bold red"
        current_time_str = self._get_formatted_time()
        
        # TRIPLE ALIGNMENT LOGIC
        bot_name = "T-BOT (V4-ULTRA)"
        right_status = f"STATUS : {status_text}"
        
        # Calculate padding for Center position
        left_len = len(current_time_str)
        center_len = len(bot_name)
        right_len = len(right_status)
        
        # Start of center text should be roughly at (width // 2) - (center_len // 2)
        center_start = max(left_len + 1, (width // 2) - (center_len // 2))
        pad_left = " " * (center_start - left_len)
        
        # End of center text is center_start + center_len
        # Start of right text should be width - right_len
        right_start = max(center_start + center_len + 1, width - right_len)
        pad_right = " " * (right_start - (center_start + center_len))
        
        # Assemble Header with markup for the status
        header_markup = f"{current_time_str}{pad_left}[bold blue]{bot_name}[/]{pad_right}STATUS : [{status_color}]{status_text}[/]"

        # Build the Dashboard String
        lines = []
        lines.append(sep)
        lines.append(header_markup)
        lines.append(sep)
        
        lines.append("Account Information")
        lines.append(f"ID : {login} | SERVER : {server} | ")
        lines.append(f"Balance : ${balance:,.2f} | EQUITY : ${equity:,.2f} | FREE MARGIN : ${free_margin:,.2f} | LEVERAGE : {leverage}x")
        lines.append(f"LIVE PnL : [bold cyan]${profit:,.2f}[/]")
        
        lines.append(sep)
        lines.append("Environment Monitor")
        lines.append(f"SYMBOL : {symbol} | PRICE : {price:,.2f} | SPREAD : {spread:.1f} pts")
        lines.append(f"REGIME : {regime} | VOLATILITY : {volatility} | SESSION : {session}")
        
        lines.append(sep)
        lines.append("Institutional Setup")
        setups = state.get("setups", {}) or {}
        found_setup = False
        for sid, reasons in setups.items():
            if reasons:
                reason_str = " | ".join(reasons)
                lines.append(f"[bold yellow]{sid}[/]: {reason_str}")
                found_setup = True
        if not found_setup:
            lines.append("[dim]NO ACTIVE STRATEGY SIGNALS[/]")
        
        lines.append(sep)
        lines.append("Live Analysis")
        logs = state.get("logs", [])
        if not logs:
            lines.append("[dim]Waiting for market events...[/]")
        else:
            for log in logs[-5:]:
                log_color = "white"
                if "[ANALYSIS]" in log: log_color = "cyan"
                if "[TRADE]" in log: log_color = "green"
                if "[ERROR]" in log: log_color = "red"
                lines.append(f"[{log_color}]> {log}[/]")
            
        lines.append(sep)
        lines.append("Economic Calendar")
        news = state.get("news_list", [])
        if not news:
            lines.append("[dim]NO UPCOMING HIGH-IMPACT NEWS[/]")
        else:
            for ev in news[:3]:
                lines.append(f"[magenta][{ev.get('time', 'N/A')}][/] [white]{ev.get('title', 'Unknown Event')}[/]")
        
        lines.append(sep)
        lines.append("Portfolio Summary")
        positions = state.get("positions", [])
        if not positions:
            lines.append("[dim]NO OPEN POSITIONS[/]")
        else:
            lines.append(f"{'SYM':<10} | {'TYPE':<5} | {'LOTS':<6} | {'PnL':<10}")
            for pos in positions:
                pnl = pos.get('profit', 0)
                pnl_color = "green" if pnl >= 0 else "red"
                lines.append(f"{pos.get('symbol', '???'):<10} | {pos.get('type_text', '???'):<5} | {pos.get('volume', 0):<6.2f} | [{pnl_color}]${pnl:<9.2f}[/]")

        lines.append(sep)
        
        content = "\n".join(lines)
        self.layout = Text.from_markup(content)
        return self.layout

def start_dashboard(layout):
    """Factory for the live renderer - uses full-screen mode."""
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)
    return Live(layout, refresh_per_second=2, screen=True)
