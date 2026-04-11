import time
import re
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
        self.prev_price = 0.0
        self.pulse_toggle = True
        
    def _get_ordinal_suffix(self, day: int) -> str:
        """Calculate ordinal suffix for day of month."""
        if 11 <= day <= 13:
            return "th"
        return {1: "st", 2: "nd", 3: "rd"}.get(day % 10, "th")

    @staticmethod
    def format_local_tz(dt: datetime) -> str:
        """Returns local timezone in format: (UTC +6) Dhaka"""
        try:
            offset = dt.strftime("%z") # e.g. +0600
            if not offset: return ""
            hours = int(offset[:3])
            # Specific requested format for Dhaka
            if hours == 6:
                return f"(UTC +6) Dhaka"
            # Generic fallback
            sign = "+" if hours >= 0 else "-"
            return f"(UTC {sign}{abs(hours)})"
        except:
            return ""

    def _get_formatted_time(self) -> str:
        """Return date in format like: 09-Apr-2026 01:44:40 (UTC +6) Dhaka"""
        now = datetime.now().astimezone()
        tz_str = self.format_local_tz(now)
        return now.strftime(f"%d-%b-%Y %H:%M:%S {tz_str}")

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
        ask = state.get("ask", 0)
        bid = state.get("bid", 0)
        spread = state.get("spread", 0)
        pips = state.get("pips", 0)
        regime = state.get("regime_type", "N/A")
        volatility = state.get("volatility", "N/A")
        session = state.get("session", "N/A")
        
        status_text = "ONLINE" if conn.get("connected") else "OFFLINE"
        status_color = "bold green" if conn.get("connected") else "bold red"
        
        # UNIFIED LOCAL TIME (User Request: Sync everything with local time)
        current_time_str = state.get("local_time") or self._get_formatted_time()
        
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

        # Environment Monitor Data Preparation
        digits = state.get("digits", 2)
        price_color = "white"
        if self.prev_price > 0:
            if price > self.prev_price: price_color = "bright_green"
            elif price < self.prev_price: price_color = "bright_red"
        
        # Assemble Header with markup for the status
        header_markup = f"{current_time_str}{pad_left}[bold blue]{bot_name}[/]{pad_right}STATUS : [{status_color}]{status_text}[/]"

        # Build the Dashboard String
        lines = []
        lines.append(sep)
        lines.append(header_markup)
        lines.append(sep)
        
        lines.append("Environment Monitor")
        # Resolution & Pulse Logic
        heartbeat = "●" if self.pulse_toggle else "○"
        self.pulse_toggle = not self.pulse_toggle
        
        self.prev_price = price

        spread_color = "bold cyan"
        if spread > 50: spread_color = "bold yellow"
        if spread > 100: spread_color = "bold red"

        # Account Verification & Lag Monitor
        login_val = state.get("login", "N/A")
        name = state.get("account_name", "N/A")
        path = state.get("terminal_path", "N/A")
        lag = state.get("tick_lag", 0)
        
        lag_color = "green" if lag < 3 else "yellow" if lag < 10 else "bold red"
        
        lines.append(f"SYMBOL   : [bold yellow]{symbol}[/] | PRICE : [{price_color}]{ask:,.{digits}f}(ask) - {bid:,.{digits}f} (bid)[/] | SPREAD : [{spread_color}]{spread:,.1f} pts ({pips:,.1f} pips)[/] | LIVE : {heartbeat}")
        lines.append(f"SESSION  : {session} | REGIME : {regime} | VOLATILE : {volatility} | LAG: [{lag_color}]{lag}s[/]")
        
        lines.append(sep)
        lines.append("Account Information")
        lines.append(f"ID : {login_val} | SERVER : {server}")
        lines.append(f"Balance : ${balance:,.2f} | EQUITY : ${equity:,.2f} | FREE MARGIN : ${free_margin:,.2f} | LEVERAGE : {leverage}x | LIVE PnL : [bold cyan]${profit:,.2f}[/]")
        
        lines.append(sep)
        lines.append("Institutional Setup")
        setups = state.get("setups", {}) or {}
        found_setup = False
        
        for sid, s_info in setups.items():
            found_setup = True
            metrics = s_info.get("metrics", {})
            thresholds = s_info.get("thresholds", {})
            signal = s_info.get("signal", "NONE")
            sig_dir = signal.direction if hasattr(signal, "direction") else str(signal)
            
            # Professional Display Name Mapping
            display_names = {
                "breakout_v4": "Breakout",
                "trendfollowing_v4": "Trend Following",
                "meanreversion_v4": "Mean Reversion",
                "liquidity_session_v4": "Liquidity Session"
            }
            clean_name = display_names.get(sid.lower(), sid.replace("_", " ").replace("v4", "").strip().title())

            # Header for Strategy
            color = "green" if sig_dir != "NONE" else "yellow"
            lines.append(f"[bold {color}]{clean_name}[/] | Bias: [bold]{sig_dir}[/]")
            
            if metrics:
                lines.append(f"  {'Metric':<12} | {'Required':<12} | {'Actual':<10} | Status")
                for m_name, live_val in metrics.items():
                    target = thresholds.get(m_name, "N/A")
                    
                    # Status logic (Simple substring/numeric check)
                    status_icon = "[green]✅[/]"
                    if "Inside" in str(live_val) or "Neutral" in str(live_val):
                        status_icon = "[red]❌[/]"
                    elif isinstance(live_val, (int, float)):
                        try:
                            # Try to extract the number from target string like "> 25"
                            import re
                            t_num_match = re.search(r"(\d+\.?\d*)", str(target))
                            if t_num_match:
                                t_num = float(t_num_match.group(1))
                                if ">" in str(target) and live_val < t_num: status_icon = "[red]❌[/]"
                                elif "<" in str(target) and live_val > t_num: status_icon = "[red]❌[/]"
                        except: pass
                    
                    val_repr = f"{live_val:.2f}" if isinstance(live_val, float) else str(live_val)
                    if m_name == "Volume": val_repr += "x"
                    
                    lines.append(f"  {m_name:<12} | {str(target):<12} | {val_repr:<10} | {status_icon}")
            lines.append("") # Spacer
            
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
        # News Status Indicator (Relocated)
        news_stale = state.get("news_stale", True)
        # Relaxed status: Economic calendars stay valid for days.
        news_status = "[bold green]SYNCED[/]" if not news_stale else "[bold yellow]SYNCED (STALE)[/]"
        lines.append(f"STATUS: {news_status}")
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
        lines.append("System Metrics")
        metrics = state.get("metrics", {}) or {}
        if metrics:
            uptime = metrics.get("uptime_seconds", 0)
            uptime_str = f"{uptime // 3600}h {(uptime % 3600) // 60}m" if uptime > 0 else "N/A"
            cycles = metrics.get("cycles_completed", 0)
            signals = metrics.get("signals_generated", 0)
            trades = metrics.get("trades_executed", 0)
            errors = metrics.get("errors", 0)
            lines.append(f"UPTIME : {uptime_str} | CYCLES : {cycles} | SIGNALS : {signals} | TRADES : {trades} | ERRORS : [red]{errors}[/]" if errors else f"UPTIME : {uptime_str} | CYCLES : {cycles} | SIGNALS : {signals} | TRADES : {trades} | ERRORS : {errors}")
        else:
            lines.append("[dim]METRICS UNAVAILABLE[/]")

        lines.append(sep)
        
        content = "\n".join(lines)
        self.layout = Text.from_markup(content)
        return self.layout

def start_dashboard(layout):
    """Factory for the live renderer - uses full-screen mode."""
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)
    return Live(layout, auto_refresh=False, screen=True)
