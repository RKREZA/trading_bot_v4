"""
TRADING BOT V3 - Dashboard
Rich-based CLI dashboard
"""

from datetime import datetime
from collections import deque
from typing import Optional, List

from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich import box

ACCENT, GREEN, RED, YELLOW, MAGENTA, WHITE, BLUE, DIM = "bright_cyan", "bright_green", "bright_red", "bright_yellow", "bright_magenta", "bright_white", "bright_blue", "dim"
BAR_FULL, BAR_EMPTY = "█", "·"


class AnalysisLogger:
    """
    In-memory log buffer for the dashboard.
    Maintains a rolling queue of logs to prevent memory bloat while keeping recent history.
    """
    def __init__(self, max_entries: int = 100):
        """
        Initializes the logger.
        
        Args:
            max_entries (int): Maximum number of log lines to retain.
        """
        self._logs = deque(maxlen=max_entries)

    def log(self, message: str, level: str = "INFO"):
        timestamp = datetime.now().strftime("%H:%M:%S")
        self._logs.append(f"[{timestamp}] [{level}] {message}")

    def get_recent(self, count: int = 50) -> List[str]:
        """
        Retrieves the most recent log entries.
        
        Args:
            count (int): Number of lines to fetch.
            
        Returns:
            List[str]: List of formatted log strings.
        """
        return list(self._logs)[-count:]


import logging
class AnalysisLoggerHandler(logging.Handler):
    """
    Bridge between Python's standard logging module and the Dashboard's AnalysisLogger.
    Allows library-level logs to appear in the CLI UI.
    """
    def __init__(self, analysis_logger: AnalysisLogger, level=logging.INFO):
        super().__init__(level)
        self.analysis_logger = analysis_logger

    def emit(self, record):
        try:
            msg = self.format(record)
            self.analysis_logger.log(msg, record.levelname)
        except Exception:
            self.handleError(record)


class Dashboard:
    """
    Real-time CLI dashboard using the 'rich' library.
    Provides a visual representation of account status, market data, signals, 
    and AI advisory context.
    """
    def __init__(self, config: dict, logger: Optional[AnalysisLogger] = None):
        """
        Initializes the dashboard.
        
        Args:
            config (dict): Global configuration.
            logger (Optional[AnalysisLogger]): Log buffer to display.
        """
        self.config = config
        self.logger = logger or AnalysisLogger()
        self.console = Console()
        self._live = None
        self.account_info = {}
        self.tick = {}
        self.signal = None
        self.daily_pnl = 0.0
        self.daily_trades = 0
        self.win_count = 0
        self.loss_count = 0
        self.selected_symbol = config.get("symbol", "BTCUSDm")
        self.session = "CLOSED"
        self.running = True
        self.h4_trend = "RANGING"
        self.m30_structure = "NEUTRAL"
        self.ai_context = {}  # Provided by TradingBot
        self.signal_history = deque(maxlen=8)
        self.positions = []
        self.market_open = True # Flag for dashboard
        self.fetch_status = ""
        self.fetch_ms = 0
        self.analysis_context = {} # Latest data from StrategyEngine


    def start(self):
        """Starts the full-screen 'rich.live' UI."""
        self._live = Live(self._render(), console=self.console, refresh_per_second=4, screen=True)
        self._live.start()

    def stop(self):
        """Stops the UI and returns the terminal to a normal state."""
        if self._live:
            self._live.stop()

    def update(self, cycle_ms: float = 0.0):
        if self._live:
            self._live.update(self._render())

    def _render(self) -> Group:
        """
        The main UI layout generator. 
        Combines various panels into a single Group for rendering.
        
        Returns:
            Group: The root layout object.
        """
        parts = []
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        header = Text()
        header.append(" TRADING BOT V3", style=f"bold {ACCENT}")
        header.append(f" | {now}", style=DIM)
        if self.running:
            header.append(" *", style=GREEN)
        parts.append(Panel(header, box=box.DOUBLE_EDGE, style=ACCENT, expand=True, padding=(0, 1)))

        # Row 1: Connection + Account
        parts.append(self._equal_row(self._render_conn(), self._render_account()))
        # Row 2: Market + Performance
        parts.append(self._equal_row(self._render_market(), self._render_perf()))
        # Row 3: Signal
        parts.append(self._render_signal())
        # Row 4: AI Advisor
        if self.config.get("use_ai_filter", False):
            parts.append(self._render_ai())
        # Row 5: Analysis + Setup
        parts.append(self._equal_row(self._render_analysis(), self._render_setup()))
        # Row 6: Open Positions
        parts.append(self._render_positions())
        # Row 7: Signal History & Logs (Side-by-side)
        parts.append(self._equal_row(self._render_signal_history(), self._render_logs()))
        
        return Group(*parts)

    def _render_conn(self) -> Panel:
        t = Table(show_header=False, box=None, padding=(0, 1), expand=True)
        t.add_column(style=DIM, width=10)
        t.add_column(style=WHITE)
        t.add_row("STATUS", Text("CONNECTED" if self.account_info.get('connected') else "OFFLINE", style=GREEN if self.account_info.get('connected') else RED))
        # Safely fall back to '-' if server isn't available yet or config is empty
        server_name = self.account_info.get('server') or self.config.get('mt5', {}).get('server', '-')
        t.add_row("SERVER", str(server_name))
        t.add_row("SESSION", Text(self.session, style=ACCENT))
        return Panel(t, title="[bold cyan]CONNECTION[/]", border_style="cyan", box=box.ROUNDED, expand=True, padding=(0, 1))

    def _render_account(self) -> Panel:
        acc = self.account_info
        t = Table(show_header=False, box=None, padding=(0, 1), expand=True)
        t.add_column(style=DIM, width=10)
        t.add_column(style=WHITE)
        t.add_row("BALANCE", Text(f"${acc.get('balance', 0):,.2f}", style=f"bold {WHITE}"))
        t.add_row("EQUITY", Text(f"${acc.get('equity', 0):,.2f}", style=f"bold {ACCENT}"))
        pl = acc.get('profit', 0)
        pl_color = GREEN if pl >= 0 else RED
        t.add_row("P/L", Text(f"{'+' if pl >= 0 else ''}${pl:,.2f}", style=f"bold {pl_color}"))
        return Panel(t, title="[bold green]ACCOUNT[/]", border_style="green", box=box.ROUNDED, expand=True, padding=(0, 1))

    def _render_market(self) -> Panel:
        tick = self.tick or {}
        t = Table(show_header=False, box=None, padding=(0, 1), expand=True)
        t.add_column(style=DIM, width=10)
        t.add_column(style=WHITE)
        status_text = "[bold green][OPEN][/]" if self.market_open else "[bold red][CLOSED][/]"
        symbol_line = Text.from_markup(f"[bold {YELLOW}]* {self.selected_symbol}[/] {status_text}")
        t.add_row("SYMBOL", symbol_line)
        price = tick.get('price', 0)
        t.add_row("PRICE", Text(f"${price:,.2f}", style=f"bold {WHITE}"))
        t.add_row("SPREAD", f"{tick.get('spread', 0):.2f}")
        if self.fetch_status:
            t.add_row("DATA TICK", Text.from_markup(self.fetch_status))
        else:
            t.add_row("DATA LAT.", f"{self.fetch_ms}ms" if self.fetch_ms > 0 else "-")
        return Panel(t, title="[bold yellow]MARKET[/]", border_style="yellow", box=box.ROUNDED, expand=True, padding=(0, 1))

    def _render_perf(self) -> Panel:
        content = Text()
        tp = self.daily_pnl
        tp_color = GREEN if tp >= 0 else RED
        content.append(f" {'+' if tp >= 0 else ''}${tp:,.2f}\n", style=f"bold {tp_color}")
        wr = (self.win_count / (self.win_count + self.loss_count) * 100) if (self.win_count + self.loss_count) > 0 else 0
        content.append(f" W:{self.win_count} L:{self.loss_count} WR:{wr:.0f}%", style=ACCENT)
        return Panel(content, title="[bold green]TODAY[/]", border_style="green", box=box.ROUNDED, expand=True, padding=(0, 1))

    def _render_signal(self) -> Panel:
        content = Text()
        if self.signal:
            s = self.signal
            icon = "+" if s['direction'] == 'BUY' else "-"
            # Signal state: LATCHED (if position open) or PENDING
            is_latched = s.get('is_latched', False)
            style = "bold white on green" if s['direction'] == 'BUY' else "bold white on red"
            
            content.append(f" {icon} {s['direction']}", style=style)
            if is_latched:
                content.append(" [ACTIVE TRADE]", style="bold yellow")
            
            content.append(f" | Entry: ${s['entry_price']:,.2f}", style=WHITE)
            content.append(f" | SL: ${s['stop_loss']:,.2f}", style=RED)
            content.append(f" | TP: ${s['take_profit']:,.2f}", style=GREEN)
            content.append(f" | Conf: {s['confidence']:.0f}%", style=ACCENT)
        else:
            # LIVE ANALYSIS VIEW (When no signal is active)
            ctx = self.analysis_context
            if ctx:
                bias = ctx.get("bias", "NEUTRAL")
                bias_color = GREEN if bias == "BULLISH" else (RED if bias == "BEARISH" else DIM)
                
                zone = "OUTSIDE"
                zone_color = DIM
                depth = 0.0
                if ctx.get("in_demand"):
                    zone, zone_color, depth = "DEMAND", GREEN, ctx.get("d_depth", 50.0)
                elif ctx.get("in_supply"):
                    zone, zone_color, depth = "SUPPLY", RED, ctx.get("s_depth", 50.0)
                
                vol_sma = ctx.get("vol_sma", 1.0)
                curr_vol = ctx.get("current_vol", 0.0)
                vol_expansion = curr_vol > vol_sma * 1.1
                vol_color = GREEN if vol_expansion else DIM
                vol_text = "EXPANDING" if vol_expansion else "STABLE"
                
                content.append(" M15 BIAS: ", style=DIM)
                content.append(f"{bias}", style=f"bold {bias_color}")
                content.append(" | ", style=DIM)
                content.append("H1 ZONE: ", style=DIM)
                content.append(f"{zone}", style=f"bold {zone_color}")
                content.append(f" ({depth:.1f}%)", style=zone_color)
                content.append(" | ", style=DIM)
                content.append("VOL: ", style=DIM)
                content.append(f"{vol_text}", style=vol_color)
            else:
                content.append(" Initializing analysis pipeline...", style=DIM)
        return Panel(content, title="[bold cyan]SIGNAL / LIVE ANALYSIS[/]", border_style="cyan", box=box.HEAVY, expand=True, padding=(0, 1))

    def _render_signal_history(self) -> Panel:
        if not self.signal_history:
            return Panel(Text("  No recent signals", style=DIM), title="[bold magenta]SIGNAL HISTORY[/]", border_style="magenta", box=box.ROUNDED, expand=True, padding=(0, 1))
        
        t = Table(show_header=True, box=None, padding=(0, 1), expand=True, header_style=DIM)
        t.add_column("TIME", width=10)
        t.add_column("SYMBOL", width=12)
        t.add_column("DIR", width=6)
        t.add_column("CONF", justify="right", width=6)
        t.add_column("VERDICT")
        
        for s in reversed(list(self.signal_history)):
            timestamp = s.get('time', '--:--:--')
            symbol = s.get('symbol', '-')
            dir_style = GREEN if s['direction'] == 'BUY' else RED
            v_style = GREEN if s.get('verdict') == 'ENTRY' else (RED if s.get('verdict') == 'REJECT' else YELLOW)
            
            t.add_row(
                timestamp,
                Text(symbol, style=YELLOW),
                Text(s['direction'], style=dir_style),
                f"{s['confidence']:.0f}%",
                Text(s.get('reason', 'VALID'), style=v_style)
            )
        
        return Panel(t, title="[bold magenta]SIGNAL HISTORY[/]", border_style="magenta", box=box.ROUNDED, expand=True, padding=(0, 1))

    def _render_positions(self) -> Panel:
        if not self.positions:
            return Panel(Text("  No open positions", style=DIM), title="[bold green]OPEN POSITIONS[/]", border_style="green", box=box.ROUNDED, expand=True, padding=(0, 1))

        t = Table(show_header=True, box=None, padding=(0, 1), expand=True, header_style=DIM)
        t.add_column("TICKET", width=10)
        t.add_column("SYMBOL", width=12)
        t.add_column("DIR", width=6)
        t.add_column("LOT", justify="right")
        t.add_column("ENTRY", justify="right")
        t.add_column("CURRENT", justify="right")
        t.add_column("P/L", justify="right")

        for p in self.positions:
            try:
                # MT5 position objects are namedtuples or similar
                ticket = str(getattr(p, 'ticket', '-'))
                symbol = str(getattr(p, 'symbol', '-'))
                pos_type = getattr(p, 'type', 0)
                dir_str = "BUY" if pos_type == 0 else "SELL"
                dir_style = GREEN if pos_type == 0 else RED
                volume = f"{getattr(p, 'volume', 0):.2f}"
                price_open = f"{getattr(p, 'price_open', 0):,.2f}"
                price_current = f"{getattr(p, 'price_current', 0):,.2f}"
                profit = getattr(p, 'profit', 0)
                pnl_style = GREEN if profit >= 0 else RED
                pnl_str = f"{'+' if profit >= 0 else ''}${profit:,.2f}"

                t.add_row(
                    ticket, 
                    Text(symbol, style=YELLOW), 
                    Text(dir_str, style=dir_style), 
                    volume, 
                    price_open, 
                    price_current, 
                    Text(pnl_str, style=f"bold {pnl_style}")
                )
            except Exception as e:
                # If we received a dict instead of a positional object (unlikely but safe)
                if isinstance(p, dict):
                    t.add_row(str(p.get('ticket')), p.get('symbol'), p.get('type'), str(p.get('volume')), str(p.get('price_open')), str(p.get('price_current')), str(p.get('profit')))
        
        return Panel(t, title="[bold green]OPEN POSITIONS[/]", border_style="green", box=box.ROUNDED, expand=True, padding=(0, 1))

    def _render_ai(self) -> Panel:
        sess = self.ai_context.get("session")
        if not sess:
            return Panel(Text("  AI analyzing pre-session context...", style=DIM), title="[bold blue]AI ADVISOR[/]", border_style="blue", box=box.ROUNDED, expand=True, padding=(0, 1))
        
        t = Table(show_header=False, box=None, padding=(0, 1), expand=True)
        t.add_column(style=DIM, width=12)
        t.add_column()

        # Risk & Multiplier
        risk = sess.get("risk_level", "?")
        r_color = {"LOW": GREEN, "MEDIUM": YELLOW, "HIGH": RED}.get(risk, WHITE)
        mult = sess.get("recommended_lot_multiplier", 1.0)
        t.add_row("SESSION RISK", Text(f"{risk} (Lot ×{mult})", style=r_color))

        # Bias
        bias = sess.get("overall_bias", "?")
        b_color = {"BULLISH": GREEN, "BEARISH": RED, "NEUTRAL": DIM}.get(bias, WHITE)
        t.add_row("MACRO BIAS", Text(f"{bias}", style=b_color))

        # Last signal verdict
        last = self.ai_context.get("last_signal_review")
        if last:
            v_color = GREEN if last.get("verdict") == "VALID" else (RED if last.get("verdict") == "AVOID" else YELLOW)
            aligned = "✓" if last.get("aligned_with_bias") else "✗"
            t.add_row("LAST SIGNAL", Text(f"{last.get('direction', '?')} → {last.get('verdict', '?')} [Bias {aligned}]", style=v_color))
            t.add_row("REASONING", Text(last.get("reasoning", ""), style=DIM))

        return Panel(t, title="[bold blue]AI ADVISOR[/]", border_style="blue", box=box.ROUNDED, expand=True, padding=(0, 1))

    def _render_analysis(self) -> Panel:
        t = Table(show_header=False, box=None, padding=(0, 1), expand=True)
        t.add_column(style=DIM, width=14)
        t.add_column(style=WHITE)
        trend_color = GREEN if self.h4_trend == "BULLISH" else (RED if self.h4_trend == "BEARISH" else DIM)
        t.add_row("H4 TREND", Text(f"{self.h4_trend}", style=trend_color))
        struct_color = GREEN if self.m30_structure == "BULLISH" else (RED if self.m30_structure == "BEARISH" else DIM)
        t.add_row("M30 STRUCTURE", Text(f"{self.m30_structure}", style=struct_color))
        return Panel(t, title="[bold magenta]ANALYSIS[/]", border_style="magenta", box=box.ROUNDED, expand=True, padding=(0, 1))

    def _render_setup(self) -> Panel:
        t = Table(show_header=False, box=None, padding=(0, 1), expand=True)
        t.add_column(style=DIM, width=14)
        t.add_column(style=WHITE)
        
        # Base defaults
        defaults = self.config.get('strategy_defaults', {})
        
        # Try to get session-specific values if available
        session_cfg = self.config.get('session_config', {}).get(self.session, {})
        session_strat = session_cfg.get('strategy', {})
        
        min_conf = session_strat.get('min_confidence', defaults.get('min_confidence', 60))
        min_score = session_strat.get('min_confluence_score', defaults.get('min_confluence_score', 4))
        risk_mult = session_cfg.get('risk_multiplier', 1.0)
        
        t.add_row("MIN CONF", f"{min_conf}%")
        t.add_row("MIN SCORE", f"{min_score}")
        t.add_row("RISK MULT", f"x{risk_mult:.2f}")
        return Panel(t, title="[bold yellow]SETUP[/]", border_style="yellow", box=box.ROUNDED, expand=True, padding=(0, 1))

    def _render_logs(self) -> Panel:
        content = Text()
        logs = self.logger.get_recent(20)
        for i, entry in enumerate(logs):
            style = GREEN if "SIGNAL" in entry or "+" in entry else (RED if "-" in entry or "ERROR" in entry else DIM)
            content.append(f"  {entry}\n", style=style)
        return Panel(content or Text("  No logs\n", style=DIM), title="[bold yellow]LOGS[/]", border_style="grey37", box=box.ROUNDED, expand=True, padding=(0, 1))

    def _equal_row(self, left: Panel, right: Panel) -> Table:
        """
        Renders two panels side-by-side with equal height.
        Calculates the height of both and sets them to the max.
        """
        # Estimate heights based on content
        def get_h(p):
            if isinstance(p.renderable, Table):
                return len(p.renderable.rows) + 2
            if isinstance(p.renderable, Text):
                return len(p.renderable.plain.split('\n')) + 2
            return 8 # Default
            
        h = max(get_h(left), get_h(right))
        left.height = h
        right.height = h
        
        grid = Table.grid(expand=True)
        grid.add_column(ratio=1)
        grid.add_column(ratio=1)
        grid.add_row(left, right)
        return grid


