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
    def __init__(self, max_entries: int = 100):
        self._logs = deque(maxlen=max_entries)

    def log(self, message: str, level: str = "INFO"):
        timestamp = datetime.now().strftime("%H:%M:%S")
        self._logs.append(f"[{timestamp}] [{level}] {message}")

    def get_recent(self, count: int = 50) -> List[str]:
        return list(self._logs)[-count:]


class Dashboard:
    def __init__(self, config: dict, logger: Optional[AnalysisLogger] = None):
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

    def start(self):
        self._live = Live(self._render(), console=self.console, refresh_per_second=4, screen=True)
        self._live.start()

    def stop(self):
        if self._live:
            self._live.stop()

    def update(self, cycle_ms: float = 0.0):
        if self._live:
            self._live.update(self._render())

    def _render(self) -> Group:
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
        parts.append(self._render_ai())
        # Row 5: Analysis + Setup
        parts.append(self._equal_row(self._render_analysis(), self._render_setup()))
        # Row 6: Logs
        parts.append(self._render_logs())
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
        t.add_row("SYMBOL", Text(f"* {self.selected_symbol}", style=f"bold {YELLOW}"))
        price = tick.get('price', 0)
        t.add_row("PRICE", Text(f"${price:,.2f}", style=f"bold {WHITE}"))
        t.add_row("SPREAD", f"{tick.get('spread', 0):.2f}")
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
            style = "bold white on green" if s['direction'] == 'BUY' else "bold white on red"
            content.append(f" {icon} {s['direction']}", style=style)
            content.append(f" | Entry: ${s['entry_price']:,.2f}", style=WHITE)
            content.append(f" | SL: ${s['stop_loss']:,.2f}", style=RED)
            content.append(f" | TP: ${s['take_profit']:,.2f}", style=GREEN)
            content.append(f" | Conf: {s['confidence']:.0f}%", style=ACCENT)
        else:
            content.append(" Scanning for opportunities...", style=DIM)
        return Panel(content, title="[bold cyan]SIGNAL[/]", border_style="cyan", box=box.HEAVY, expand=True, padding=(0, 1))

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
        min_conf = self.config.get('strategy', {}).get('min_confidence', 60)
        min_score = self.config.get('strategy', {}).get('min_confluence_score', 4)
        t.add_row("MIN CONF", f"{min_conf}%")
        t.add_row("MIN SCORE", f"{min_score}")
        return Panel(t, title="[bold yellow]SETUP[/]", border_style="yellow", box=box.ROUNDED, expand=True, padding=(0, 1))

    def _render_logs(self) -> Panel:
        content = Text()
        logs = self.logger.get_recent(20)
        for i, entry in enumerate(logs):
            style = GREEN if "SIGNAL" in entry or "+" in entry else (RED if "-" in entry or "ERROR" in entry else DIM)
            content.append(f"  {entry}\n", style=style)
        return Panel(content or Text("  No logs\n", style=DIM), title="[bold yellow]LOGS[/]", border_style="grey37", box=box.ROUNDED, expand=True, padding=(0, 1))

    def _equal_row(self, left, right) -> Table:
        grid = Table.grid(expand=True)
        grid.add_column(ratio=1)
        grid.add_column(ratio=1)
        grid.add_row(left, right)
        return grid


class BacktestDashboard:
    def __init__(self, config):
        self.config = config
        self.console = Console()
        self._live = None

    def show_progress(self, current, total, current_time, signal_count, trade_count):
        if not self._live:
            self._live = Live(console=self.console, refresh_per_second=10)
            self._live.start()

        pct = (current / total * 100) if total > 0 else 0
        filled = int(pct / 100 * 40)
        progress = Text()
        progress.append(f"\n BACKTEST [{current}/{total}]\n\n ", style=f"bold {ACCENT}")
        progress.append(BAR_FULL * filled, style=GREEN)
        progress.append(BAR_EMPTY * (40 - filled), style=DIM)
        progress.append(f" {pct:.1f}%\n\n {current_time} | Signals: {signal_count} | Trades: {trade_count}")
        
        panel = Panel(progress, title="[bold cyan]PROGRESS[/]", border_style="cyan", box=box.HEAVY, expand=True, padding=(1, 2))
        self._live.update(panel)

    def show_results(self, r):
        if self._live:
            self._live.stop()
        self.console.clear()
        summary = Text()
        summary.append(f"\n {r['symbol']}: {r['start_date']} to {r['end_date']}\n", style=WHITE)
        profit = r['final_balance'] - r['initial_balance']
        summary.append(f" Initial: ${r['initial_balance']:,.2f} | Final: ${r['final_balance']:,.2f}\n", style=DIM)
        summary.append(f" Return: {r['return_pct']:.1f}%\n", style=f"bold {GREEN if profit >= 0 else RED}")
        if r.get('halted'):
            summary.append(" ⚠ HALTED — Max drawdown breached\n", style=f"bold {RED}")
        self.console.print(Panel(summary, title="[bold green]SUMMARY[/]", border_style="green", box=box.ROUNDED, expand=True, padding=(1, 2)))

        stats = Table(box=box.SIMPLE_HEAVY, expand=True)
        stats.add_column("METRIC", style=ACCENT, width=20)
        stats.add_column("VALUE", style=WHITE, justify="right")
        wr_color = GREEN if r['win_rate'] >= 60 else (YELLOW if r['win_rate'] >= 40 else RED)
        stats.add_row("Total Trades", str(r['total_trades']))
        stats.add_row("  TP / SL / TSL", f"{r.get('tp_count', 0)} / {r.get('sl_count', 0)} / {r.get('tsl_count', 0)}")
        stats.add_row("Win Rate", Text(f"{r['win_rate']:.1f}%", style=f"bold {wr_color}"))
        stats.add_row("Profit Factor", f"{r['profit_factor']:.2f}")
        stats.add_row("Total Profit", f"${r['total_profit']:,.2f}")
        stats.add_row("Total Loss", f"${r['total_loss']:,.2f}")
        stats.add_row("Avg Win", f"${r.get('avg_win', 0):,.2f}")
        stats.add_row("Avg Loss", f"${r.get('avg_loss', 0):,.2f}")
        stats.add_row("Max Drawdown", Text(f"{r['max_drawdown']:.1f}%", style=f"bold {RED if r['max_drawdown'] > 20 else YELLOW}"))
        stats.add_row("Sharpe Ratio", f"{r.get('sharpe_ratio', 0):.2f}")
        stats.add_row("Win Streak", str(r.get('max_win_streak', 0)))
        stats.add_row("Loss Streak", str(r.get('max_loss_streak', 0)))
        stats.add_row("Daily Limit Hits", str(r.get('daily_limit_hits', 0)))
        self.console.print(Panel(stats, title="[bold cyan]STATS[/]", border_style="cyan", box=box.ROUNDED, expand=True, padding=(1, 2)))

        tt = Table(box=box.SIMPLE_HEAVY, expand=True)
        tt.add_column("TIME", style=DIM, width=18)
        tt.add_column("DIR", width=5)
        tt.add_column("LOT", justify="right")
        tt.add_column("ENTRY", justify="right")
        tt.add_column("EXIT", justify="right")
        tt.add_column("SL", justify="right", style=RED)
        tt.add_column("TP", justify="right", style=GREEN)
        tt.add_column("RESULT", justify="center")
        tt.add_column("P/L", justify="right")
        for t in r.get('trades', [])[-50:]:
            dir_style = GREEN if t['direction'] == 'BUY' else RED
            res_style = GREEN if t['result'] == 'TP' else (YELLOW if t['result'] == 'TSL' else RED)
            pnl_color = GREEN if t['pnl'] >= 0 else RED
            tt.add_row(
                t['time'],
                Text(t['direction'], style=f"bold {dir_style}"),
                f"{t.get('lot', 0):.3f}",
                f"{t['entry']:.2f}",
                f"{t.get('exit', 0):.2f}",
                f"{t.get('sl', 0):.2f}",
                f"{t.get('tp', 0):.2f}",
                Text(t['result'], style=f"bold {res_style}"),
                Text(f"{'+' if t['pnl'] >= 0 else ''}${t['pnl']:.2f}", style=pnl_color)
            )
        self.console.print(Panel(tt, title="[bold yellow]TRADES[/]", border_style="yellow", box=box.ROUNDED, expand=True, padding=(1, 2)))
