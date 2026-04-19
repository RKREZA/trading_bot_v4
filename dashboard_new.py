import time
import logging
import os
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, Deque
from collections import deque

from rich.console import Console
from rich.text import Text
from rich.live import Live
from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table
from rich.progress import BarColumn, Progress, Sparkline
from rich.sparkline import Sparkline
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
            Layout(name="body", size=30),  # Increased size to accommodate additional panels
            Layout(name="footer", size=15),
            Layout(name="vacuum", ratio=1)
        )
        
        # Footer Synchronization: Row-based horizontal split
        self.layout["footer"].split_row(
            Layout(name="footer_analysis", ratio=2),
            Layout(name="footer_logs", ratio=3),
            Layout(name="footer_calendar", ratio=3)
        )
        
        # Body Refactor: Tiered Architecture (Precision Vertical Fitting)
        # Top Row: Fixed height (fitted)
        # Middle Row: Performance metrics
        # Setups Row: Fixed height (fitted)
        self.layout["body"].split_column(
            Layout(name="body_top", size=9),
            Layout(name="body_middle", size=9),  # New panel for performance metrics
            Layout(name="setups", size=15)
        )
        self.layout["body_top"].split_row(
            Layout(name="environment", ratio=5),
            Layout(name="risk", ratio=6),
            Layout(name="exposure", ratio=6)
        )
        # Split the middle section for performance analytics
        self.layout["body_middle"].split_row(
            Layout(name="performance", ratio=6),
            Layout(name="trades", ratio=6)
        )

    def _get_formatted_time(self) -> str:
        now = datetime.now().astimezone()
        tz_name = now.tzname() or "Local"
        offset = now.strftime("%z")
        # Format: 14-Apr-2026 07:54:30 (UTC+0600) BST
        return now.strftime(f"%d-%b-%Y %H:%M:%S (UTC{offset}) {tz_name}")

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
        
        table = Table.grid(padding=0)
        table.add_column(style="bold magenta")
        table.add_column()
        table.add_row("SYMBOL  ", f"[bold yellow]{symbol}[/]")
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
        profit = state.get("live_profit", acc.get("profit", 0))
        
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
         
         # Extract key metrics
         total_trades = metrics.get("total_trades", 0)
         win_rate = metrics.get("win_rate", 0.0)
         profit_factor = metrics.get("profit_factor", 0.0)
         sharpe_ratio = metrics.get("sharpe_ratio", 0.0)
         sortino_ratio = metrics.get("sortino_ratio", 0.0)
         max_consecutive_losses = metrics.get("max_consecutive_losses", 0)
         avg_profit = metrics.get("avg_profit", 0.0)
         avg_loss = metrics.get("avg_loss", 0.0)
         
         # Format values for display
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
         table.add_row("Sortino Ratio", f"[{sr_color}]{sortino_ratio:.2f}[/]")
         table.add_row("Max Cons. Losses", f"{max_consecutive_losses}")
         table.add_row("Avg Profit", f"[green]${avg_profit:.2f}[/]")
         table.add_row("Avg Loss", f"[red]${avg_loss:.2f}[/]")
         
         return Panel(table, title="Performance Analytics", border_style="green")

     def _make_trades(self, state: dict) -> Panel:
         """Create trade history panel with recent trades and P&L sparkline."""
         trade_history = state.get("trade_history", [])
         if not trade_history:
             return Panel(Text("No trade history", style="dim"), title="Trade History", border_style="blue")
         
         # Get recent trades (last 20)
         recent_trades = trade_history[-20:] if len(trade_history) > 20 else trade_history
         
         # Create table for recent trades
         table = Table(show_header=True, header_style="bold blue", box=box.SIMPLE)
         table.add_column("Time", width=8)
         table.add_column("Dir", width=4)
         table.add_column("Symbol", width=8)
         table.add_column("P&L", width=10, justify="right")
         table.add_column("Lots", width=6, justify="right")
         
         for trade in recent_trades:
             timestamp = trade.get('timestamp', '')
             if timestamp:
                 try:
                     # Format timestamp to HH:MM
                     dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                     time_str = dt.strftime("%H:%M")
                 except:
                     time_str = timestamp[-5:] if len(timestamp) >= 5 else timestamp
             else:
                 time_str = "--"
             
             direction = trade.get('direction', 'N/A')
             dir_display = "BUY" if direction == "BUY" else "SELL" if direction == "SELL" else direction[:4]
             
             symbol = trade.get('symbol', 'N/A')
             if len(symbol) > 8:
                 symbol = symbol[:8]
             
             pnl = trade.get('profit', 0.0)
             pnl_color = "green" if pnl > 0 else "red" if pnl < 0 else "white"
             
             lots = trade.get('volume', 0.0)
             
             table.add_row(
                 time_str,
                 dir_display,
                 symbol,
                 f"[{pnl_color}]{pnl:+.2f}[/]",
                 f"{lots:.2f}"
             )
         
         # Create sparkline for recent P&L values
         pnl_values = [trade.get('profit', 0.0) for trade in recent_trades]
         if pnl_values:
             # Normalize values for sparkline (shift to make positive for display)
             min_pnl = min(pnl_values) if pnl_values else 0
             max_pnl = max(pnl_values) if pnl_values else 0
             range_pnl = max_pnl - min_pnl if max_pnl != min_pnl else 1
             
             if range_pnl > 0:
                 normalized = [(p - min_pnl) / range_pnl for p in pnl_values]
             else:
                 normalized = [0.5] * len(pnl_values)
             
             sparkline = Sparkline(normalized, height=3)
             spark_panel = Panel(
                 sparkline,
                 title="Recent P&L Trend",
                 border_style="blue",
                 padding=(0, 1)
             )
             
             # Combine table and sparkline
             from rich.columns import Columns
             combined = Columns([table, spark_panel], equal=True, expand=True)
             return Panel(combined, title="Trade History & P&L Trend", border_style="blue")
         else:
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
        # Maximize width with expand=True and tight padding
        table = Table(box=None, show_header=True, header_style="bold yellow", expand=True, padding=(0, 2))
        table.add_column("STRATEGY", width=25, style="bold cyan", no_wrap=True)
        table.add_column("REQUIREMENTS [dim](Targets)[/]", ratio=1)
        table.add_column("ANALYSIS [dim](Live)[/]", ratio=1)
        table.add_column("BIAS", width=12, justify="center")
        table.add_column("COOLDOWN", width=10, justify="center")
        table.add_column("GRADE", width=10, justify="right")
        
        for sid, s_info in setups.items():
            signal = s_info.get("signal", "NONE")
            sig_dir = signal.direction if hasattr(signal, "direction") else str(signal)
            
            # Use confidence as fidelity if not explicitly provided
            dfs = s_info.get("fidelity")
            if dfs is None:
                dfs = signal.confidence if hasattr(signal, "confidence") else 1.0
            
            thresholds = s_info.get("thresholds", {})
            metrics = s_info.get("metrics", {})
            
            req_text = Text()
            analysis_text = Text()
            
            # Synchronized Row-by-Row Aligment
            # We iterate through keys in thresholds to ensure vertical alignment
            all_keys = list(thresholds.keys())
            # Also add metrics keys that might not be in thresholds (extra telemetry)
            for mk in metrics.keys():
                if mk not in all_keys and not str(mk).startswith("_"):
                    all_keys.append(mk)
            
            for k in all_keys:
                target_val = thresholds.get(k, "---")
                live_val = metrics.get(k, "N/A")
                
                # Requirements Column
                req_text.append(f" {k:<12} ", style="dim magenta")
                req_text.append(f"{target_val}\n", style="bold white")
                
                # Analysis Column
                if isinstance(live_val, float):
                    live_str = f"{live_val:.2f}"
                else:
                    live_str = str(live_val)
                    
                analysis_text.append(f" {k:<12} ", style="dim green")
                # Highlight if value is N/A or empty
                style = "bold white" if live_str != "N/A" else "dim red"
                analysis_text.append(f"{live_str}\n", style=style)
            
            # Bias Color logic
            bias_color = "bold green" if "BUY" in sig_dir.upper() else "bold red" if "SELL" in sig_dir.upper() else "dim white"
            
            # Cooldown Timer logic
            cooldown = metrics.get("Cooldown", 0)
            if cooldown > 0:
                cooldown_text = Text(f"[red]{cooldown}[/]", style="bold red")
                cooldown_label = f"[red]CYCLES[/]"
            else:
                cooldown_text = Text("READY", style="bold green")
                cooldown_label = "[green]READY[/]"
            
            clean_name = sid.replace("_v5", "").replace("_", " ").title()
            table.add_row(
                clean_name, 
                req_text, 
                analysis_text, 
                Text(sig_dir, style=bias_color),
                cooldown_label,
                f"[bold white]{dfs*100:.0f}%[/]"
            )
            
        return Panel(table, title="Institutional Setups (HFT-Fidelity)", border_style="yellow", padding=0)

    def _make_footer(self, state: dict) -> Any:
        analysis_logs = state.get("analysis_logs", [])
        system_logs = state.get("system_logs", [])
        news = state.get("news_list", [])
        
        grid = Table.grid(expand=True)
        grid.add_column(ratio=1.0) # Analysis
        grid.add_column(ratio=1.5) # Live Logs
        grid.add_column(ratio=1.5) # Economic Calendar
        
        # 1. ANALYSIS LOGS (Strategy heartbeats)
        analysis_text = Text()
        for log in analysis_logs[-5:]:
            analysis_text.append(f"> {log}\n", style="cyan")
            
        # 2. SYSTEM LOGS (Trades, Errors, Environment)
        system_text = Text()
        for log in system_logs[-5:]:
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
         self.layout["footer_calendar"].update(f_calendar)
         
         self.layout["vacuum"].update("")
         return self.layout

def start_dashboard(layout):
    return Live(layout, auto_refresh=False, screen=True)
