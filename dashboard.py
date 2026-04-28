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
from rich import box
from rich.columns import Columns

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
        # All functional tiers are set to exact content-height (fitting)
        # The 'vacuum' absorbing everything else to ensure top-pinning
        self.layout.split_column(
            Layout(name="header", size=4),
            Layout(name="body", ratio=1),
            Layout(name="logs_footer", size=10)
        )
        
        # Logs Footer: Split horizontally between Analysis and System Logs
        self.layout["logs_footer"].split_row(
            Layout(name="footer_analysis", ratio=1),
            Layout(name="footer_logs", ratio=1)
        )
        
        # Body Refactor: Tiered Architecture
        self.layout["body"].split_column(
            Layout(name="body_top", size=8),
            Layout(name="body_middle", size=8),
            Layout(name="setups", size=5)
        )
        self.layout["body_top"].split_row(
            Layout(name="environment", ratio=1),
            Layout(name="risk", ratio=1),
            Layout(name="exposure", ratio=1)
        )
        self.layout["body_middle"].split_row(
            Layout(name="performance", ratio=1),
            Layout(name="trades", ratio=1),
            Layout(name="calendar", ratio=1)
        )

    def _get_formatted_time(self) -> str:
        now = datetime.now().astimezone()
        tz_name = now.tzname() or "Local"
        offset_secs = now.utcoffset().total_seconds() if now.utcoffset() else 0
        offset_hours = int(offset_secs / 3600)
        sign = "+" if offset_hours >= 0 else ""
        return now.strftime(f"%d-%b-%Y %I:%M:%S %p (UTC {sign}{offset_hours}) {tz_name}")

    def _make_header(self, state: dict) -> Panel:
        conn = state.get("connection", {}) or {}
        status_text = "ONLINE" if conn.get("connected") else "OFFLINE"
        status_color = "bold green" if conn.get("connected") else "bold red"
        
        heartbeat_icon = "●" if self.pulse_toggle else "○"
        heartbeat = f"[bold bright_green blink]{heartbeat_icon}[/]"
        self.pulse_toggle = not self.pulse_toggle
        
        local_time = self._get_formatted_time()
        server_time = state.get("server_time", "Syncing Server Time...")
        
        grid = Table.grid(expand=True)
        grid.add_column(justify="left", ratio=1)
        grid.add_column(justify="center", ratio=1)
        grid.add_column(justify="right", ratio=1)
        
        status_markup = f"\nSTATUS : [{status_color}]{status_text}[/]  {heartbeat}"
        
        grid.add_row(
            Text(f"LOCAL  : {local_time}\nSERVER : {server_time}", style="cyan"),
            Text(f"\nV5-INSIGNIA", style="bold cyan"),
            Text.from_markup(status_markup)
        )
        return Panel(grid, style="white on black")

    def _make_environment(self, state: dict) -> Panel:
        symbol = state.get("symbol", "N/A")
        price = state.get("price", 0)
        ask = state.get("ask", 0)
        bid = state.get("bid", 0)
        spread = state.get("spread", 0)
        spread_pts = spread if spread > 0 else 0
        regime = state.get("regime_type", "N/A")
        volatility = state.get("volatility", "N/A")
        session = state.get("session", "N/A")
        digits = state.get("digits", 2)
        
        price_color = "bright_green" if price > self.prev_price else "bright_red" if price < self.prev_price else "white"
        self.prev_price = price
        
        table = Table.grid(padding=0)
        table.add_column(style="bold magenta")
        table.add_column()
        table.add_row("SYMBOL  ", f"[bold yellow]{symbol}[/]")
        table.add_row("PRICE", f"[{price_color}]{ask:,.{digits}f} (ask) / {bid:,.{digits}f} (bid)[/]")
        spread_pips = state.get("pips", 0.0)
        spread_display = f"[cyan]{spread_pts:,.0f} pts ({spread_pips:.1f} pips)[/]" if spread > 0 else "[cyan]0[/]"
        table.add_row(f"SPREAD", spread_display)
        session_color = "red" if "CLOSED" in session else "bold bright_blue"
        table.add_row("SESSION", f"[{session_color}]{session}[/]")
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
        
        table = Table.grid(padding=0)
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

    def _make_performance(self, state: dict) -> Panel:
        """Create performance metrics panel with Sharpe ratio, win rate, etc."""
        metrics = state.get("metrics", {})
        if not metrics:
            return Panel(Text("No metrics available", style="dim"), title="Performance Metrics", border_style="green")
        
        total_trades = metrics.get("total_trades", 0)
        win_rate = metrics.get("win_rate", 0.0)
        profit_factor = metrics.get("profit_factor", 0.0)
        sharpe_ratio = metrics.get("sharpe_ratio", 0.0)
        
        pf_color = "green" if profit_factor > 1.5 else "yellow" if profit_factor > 1.0 else "red"
        sr_color = "green" if sharpe_ratio > 1.5 else "yellow" if sharpe_ratio > 0.5 else "red"
        wr_color = "green" if win_rate > 60 else "yellow" if win_rate > 40 else "red"
        
        table = Table.grid(padding=0)
        table.add_column(style="bold white")
        table.add_column(justify="right")
        table.add_row("Total Trades", f"{total_trades}")
        table.add_row("Win Rate", f"[{wr_color}]{win_rate:.1f}%[/]")
        table.add_row("Profit Factor", f"[{pf_color}]{profit_factor:.2f}[/]")
        table.add_row("Sharpe Ratio", f"[{sr_color}]{sharpe_ratio:.2f}[/]")
        
        return Panel(table, title="Performance Analytics", border_style="green")

    def _make_trades(self, state: dict) -> Panel:
        """Create trade history panel with recent trades and P&L sparkline."""
        trade_history = state.get("trade_history", [])
        if not trade_history:
            return Panel(Text("No trade history", style="dim"), title="Trade History", border_style="blue")
        
        recent_trades = trade_history[-15:] if len(trade_history) > 15 else trade_history
        
        table = Table(show_header=True, header_style="bold blue", box=box.SIMPLE)
        table.add_column("Time", width=6)
        table.add_column("Dir", width=4)
        table.add_column("P&L", width=8, justify="right")
        
        for trade in recent_trades:
            timestamp = trade.get('timestamp', '')
            if timestamp and len(timestamp) >= 5:
                time_str = timestamp[-5:]
            else:
                time_str = "--:--"
            
            direction = trade.get('direction', 'N/A')[:4]
            pnl = trade.get('profit', 0.0)
            pnl_color = "green" if pnl > 0 else "red" if pnl < 0 else "white"
            
            table.add_row(time_str, direction, f"[{pnl_color}]{pnl:+.2f}[/]")
        
        return Panel(table, title="Trade History", border_style="blue")

    def _format_dict_styled(self, d: Dict[str, Any], color_scheme: str = "cyan") -> Text:
        if not d: return Text("N/A", style="dim")
        text = Text()
        entries = list(d.items())
        # Show all entries as requested by institutional users
        for i, (k, v) in enumerate(entries):
            if str(k).startswith("_"): continue
            
            # Key styling: Clean labels with consistent padding
            clean_k = k.replace("_", " ").title()
            text.append(f" {clean_k:<12} ", style=f"dim {color_scheme}")
            
            # Value styling
            if isinstance(v, float):
                val_str = f"{v:.2f}"
            elif isinstance(v, list):
                val_str = ", ".join([str(x) for x in v[:3]])
                if len(v) > 3: val_str += "..."
            elif isinstance(v, dict):
                sub_keys = [sk for sk in v.keys() if not str(sk).startswith("_")]
                val_str = ", ".join([str(sk).title() for sk in sub_keys[:1]])
                if len(sub_keys) > 1: val_str += "..."
            else:
                val_str = str(v)
            
            if len(val_str) > 20: val_str = val_str[:17] + "..."
            
            # Values are emphasized
            text.append(f"{val_str}\n", style="bold white")
            
        return text

    def _make_setups(self, state: dict) -> Panel:
        setups = state.get("setups", {}) or {}
        
        # Debug table for empty state
        if not setups:
            table = Table(box=None, show_header=True, header_style="bold yellow", expand=True, padding=(0, 2))
            table.add_column("STRATEGY", width=25, style="bold cyan", no_wrap=True)
            table.add_column("STATUS", width=35)
            table.add_column("METRICS", width=25)
            table.add_column("BIAS", width=10, justify="center")
            rtc = state.get("debug_runimes_count", 0)
            table.add_row("DEBUG: No setups", f"Runimes: {rtc}", "Check config", "-")
            return Panel(table, title="Institutional Setups (HFT-Fidelity)", border_style="yellow", padding=0)
        
        # Table for valid setups
        table = Table(box=None, show_header=True, header_style="bold yellow", expand=True, padding=(0, 2))
        table.add_column("STRATEGY", width=25, style="bold cyan", no_wrap=True)
        table.add_column("REQUIREMENTS [dim](Targets)[/]", ratio=1)
        table.add_column("ANALYSIS [dim](Live)[/]", ratio=1)
        table.add_column("BIAS", width=12, justify="center")
        table.add_column("GRADE", width=10, justify="right")
        
        for sid, s_info in setups.items():
            signal = s_info.get("signal", "NONE")
            sig_dir = signal.direction if hasattr(signal, "direction") else str(signal)
            
            dfs = s_info.get("fidelity")
            if dfs is None:
                dfs = signal.confidence if hasattr(signal, "confidence") else 1.0
            
            thresholds = s_info.get("thresholds", {})
            metrics = s_info.get("metrics", {})
            
            key_metrics = []
            for k, v in list(metrics.items())[:3]:
                if not str(k).startswith("_"):
                    key_metrics.append(f"{k}:{v}")
            req_str = ",".join([f"{k}:{v}" for k, v in list(thresholds.items())[:3]]) or "---"
            analysis_str = " ".join(key_metrics) if key_metrics else "N/A"
            
            bias_color = "bold green" if "BUY" in sig_dir.upper() else "bold red" if "SELL" in sig_dir.upper() else "dim white"
            
            clean_name = sid.replace("_v5", "").replace("_", " ").title()[:20]
            table.add_row(
                clean_name[:25], 
                req_str[:35], 
                analysis_str[:25], 
                Text(sig_dir[:10], style=bias_color),
                f"[bold white]{dfs*100:.0f}%[/]"
            )
        
        return Panel(table, title="Institutional Setups (HFT-Fidelity)", border_style="yellow", padding=0)

    def _make_footer(self, state: dict) -> Any:
        analysis_logs = state.get("analysis_logs", [])
        system_logs = state.get("system_logs", [])
        news = state.get("news_list", [])
        
        grid = Table.grid(expand=True)
        grid.add_column(ratio=1) # Analysis
        grid.add_column(ratio=1) # Live Logs
        grid.add_column(ratio=1) # Economic Calendar
        
        # 1. ANALYSIS LOGS (Strategy heartbeats)
        analysis_text = Text()
        if not analysis_logs:
            analysis_text.append("> WAITING FOR STRATEGY SIGNAL...\n", style="dim cyan")
        else:
            for log in analysis_logs[-8:]:
                analysis_text.append(f"> {log}\n", style="cyan")
            
        # 2. SYSTEM LOGS (Trades, Errors, Environment)
        system_text = Text()
        if not system_logs:
            system_text.append("> INITIALIZING SYSTEM MODULES...\n", style="dim white")
        else:
            for log in system_logs[-8:]:
                color = "green" if "[TRADE]" in log else "red" if "[ERROR]" in log or "[FATAL]" in log else "white"
                system_text.append(f"> {log}\n", style=color)
            
        # 3. ECONOMIC CALENDAR
        news_text = Text()
        if not news:
            news_text.append("No high-impact news\nin next 24 hours.", style="dim italic")
        else:
            for ev in news[:4]:
                ev_time = ev.get('time', 'N/A')
                news_text.append(f"[{ev_time}] {ev.get('title')}\n", style="magenta")

        return (
            Panel(analysis_text, title="Live Analysis", border_style="dim"),
            Panel(system_text, title="Live Logs", border_style="blue"),
            Panel(news_text, title="Economic Calendar", border_style="magenta")
        )

    def _make_indicators(self, state: dict) -> Panel:
        rsi = state.get("rsi", 0)
        atr = state.get("atr", 0)
        adx = state.get("adx", 0)
        rsi_color = "red" if rsi > 70 or rsi < 30 else "green"
        adx_color = "bright_green" if adx > 25 else "yellow"
        table = Table.grid(padding=0)
        table.add_column(style="bold cyan", width=12)
        table.add_column(justify="right", style="bold white")
        table.add_row("RSI (14)", f"[{rsi_color}]{rsi:.2f}[/]")
        table.add_row("ATR (14)", f"{atr:.5f}")
        table.add_row("ADX (14)", f"[{adx_color}]{adx:.2f}[/]")
        return Panel(table, title="Indicators [M5]", border_style="cyan")

    def _make_account(self, state: dict) -> Panel:
        acc = state.get("account", {})
        balance = acc.get("balance", 0)
        equity = acc.get("equity", 0)
        margin_level = acc.get("margin_level", 0)
        table = Table.grid(padding=0)
        table.add_column(style="bold yellow", width=12)
        table.add_column(justify="right", style="bold white")
        table.add_row("Balance", f"${balance:,.2f}")
        table.add_row("Equity", f"${equity:,.2f}")
        table.add_row("Margin Lvl", f"{margin_level:.1f}%")
        return Panel(table, title="Account Health", border_style="yellow")

    def _make_body_bottom(self, state: dict) -> Panel:
        positions = state.get("detailed_positions", [])
        if not positions:
            return Panel(Text("No active positions", style="dim center"), title="Active Positions Tracker", border_style="bright_blue")
        table = Table(show_header=True, header_style="bold blue", box=box.SIMPLE, expand=True)
        table.add_column("ID", style="dim")
        table.add_column("Symbol")
        table.add_column("Type")
        table.add_column("Lots", justify="right")
        table.add_column("Entry", justify="right")
        table.add_column("Current", justify="right")
        table.add_column("Pips", justify="right")
        table.add_column("Profit", justify="right")
        for pos in positions:
            pnl_color = "bright_green" if pos['profit'] > 0 else "bright_red"
            type_color = "cyan" if pos['type'] == "BUY" else "magenta"
            table.add_row(
                str(pos['id'])[-6:],
                pos['symbol'],
                f"[{type_color}]{pos['type']}[/]",
                f"{pos['lots']:.2f}",
                f"{pos['entry']:.5f}",
                f"{pos['current']:.5f}",
                f"[{pnl_color}]{pos['pips']:+.1f}[/]",
                f"[{pnl_color}]${pos['profit']:+.2f}[/]"
            )
        return Panel(table, title="Active Positions Tracker", border_style="bright_blue")

    def update(self, state: dict) -> Layout:
        """Update Layout with fresh state objects."""
        self.layout["header"].update(self._make_header(state))
        self.layout["environment"].update(self._make_environment(state))
        self.layout["risk"].update(self._make_risk(state))
        self.layout["exposure"].update(self._make_exposure(state))
        self.layout["performance"].update(self._make_performance(state))
        self.layout["trades"].update(self._make_trades(state))
        self.layout["setups"].update(self._make_setups(state))
        
        # Footer Synchronization
        f_analysis, f_logs, f_calendar = self._make_footer(state)
        self.layout["footer_analysis"].update(f_analysis)
        self.layout["footer_logs"].update(f_logs)
        self.layout["calendar"].update(f_calendar)
        return self.layout

def start_dashboard(layout):
    return Live(layout, auto_refresh=True, refresh_per_second=4, screen=True)
