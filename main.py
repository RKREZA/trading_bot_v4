"""
TRADING BOT V3 - Main Entry Point
Multi-Strategy Execution Framework.
Orchestrates MT5 connection, data fetching, strategy runtimes, and dashboard.
"""

import argparse
import json
import re
import logging
import os
import sys
import time
import threading
import queue
import copy
import pandas as pd
import numpy as np
from datetime import datetime, timezone, date, timedelta
from typing import Optional, List, Tuple, Dict, Any

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import print as rprint
from rich.live import Live

from core.walk_forward import WalkForwardValidator
from core.health import start_health_server
from core.types import BotConfig

try:
    import MetaTrader5 as mt5
except ImportError:
    mt5 = None

from dotenv import load_dotenv

from core.logger import setup_logging
from core.connection import MT5Connection, PositionManager
from core.data_fetcher import DataFetcher
from core.backtester import BacktestEngine
from core.strategy_engine import StrategyEngine, TradeSignal
from core.ai_advisor import AIAdvisor
from core.risk_manager import RiskManager
from core.notifications import NotificationManager
from core.regime import MarketRegime
from core.state_manager import SecureStateManager
from core.execution_pipeline import ExecutionPipeline
from core.trailing_stop import TrailingStopManager
from core.strategy_orchestrator import StrategyOrchestrator
from core.strategy_runtime import StrategyRuntime
from dashboard import Dashboard, AnalysisLogger, AnalysisLoggerHandler

# Multi-Strategy
from strategies import create_strategy

load_dotenv()

logger = logging.getLogger("trading_bot.main")


class TradingBot:
    """
    The central orchestrator of the Trading Bot V3 system.
    Supports multi-strategy execution with fully isolated strategy runtimes.
    """

    def __init__(self, config_path: str = "config.json"):
        self.config = self._load_config(config_path)
        self.analysis_logger = AnalysisLogger(max_entries=100)

        # Bridge standard logging to the dashboard logger
        bridged_handler = AnalysisLoggerHandler(self.analysis_logger)
        bridged_handler.setFormatter(logging.Formatter("%(name)s: %(message)s"))
        logging.getLogger("trading_bot").addHandler(bridged_handler)

        self.ai_advisor = AIAdvisor(self.config, self.analysis_logger)
        self.strategy = StrategyEngine(self.config, self.analysis_logger)
        self.dashboard = Dashboard(self.config, self.analysis_logger)
        self.connection = MT5Connection()
        self.connection.config = self.config
        self.position_manager = PositionManager(self.connection)
        self.data_fetcher = DataFetcher()
        self.risk_manager = RiskManager(self.config)
        self.notification_manager = NotificationManager(self.config)
        self.state_manager = SecureStateManager()

        self.daily_pnl = 0.0
        self.daily_trades = 0
        self.win_count = 0
        self.loss_count = 0
        self.running = False
        self.last_reset_day = date.today()
        self.max_drawdown_reached = 0.0
        self.last_logged_session: Optional[str] = None
        self.position_meta = {}
        self.notified_deals = set()
        self.state_file = "bot_state.json"
        self.state_lock = threading.Lock()
        self.execution_queue = queue.Queue()
        self._shutdown_event = threading.Event()

        # ── Multi-Strategy Framework ─────────────────────
        self.strategy_runtimes: List[StrategyRuntime] = []
        self.orchestrator: Optional[StrategyOrchestrator] = None
        self._build_strategy_runtimes()

        # Legacy ExecutionPipeline (kept for --legacy mode)
        self.execution_pipeline = ExecutionPipeline(
            config=self.config,
            connection=self.connection,
            position_manager=self.position_manager,
            strategy=self.strategy,
            ai_advisor=self.ai_advisor,
            risk_manager=self.risk_manager,
            notification_manager=self.notification_manager,
            position_meta=self.position_meta,
            state_lock=self.state_lock
        )
        self.trailing_stop_manager = TrailingStopManager(
            config=self.config,
            connection=self.connection,
            position_meta=self.position_meta,
            state_lock=self.state_lock
        )

        self._load_state()

    def _build_strategy_runtimes(self):
        """Build StrategyRuntime instances from config."""
        strategies_cfg = self.config.get("strategies", [])
        initial_balance = self.config.get("backtest", {}).get("initial_balance", 1000.0)

        if not strategies_cfg:
            # Auto-generate from legacy config
            strategy_type = self.config.get("strategy_type", "SNIPER")
            sid = f"{strategy_type.lower()}_v1"
            try:
                strat = create_strategy(sid, strategy_type, self.config)
                runtime = StrategyRuntime(strat, self.config, initial_balance)
                self.strategy_runtimes.append(runtime)
                logger.info("Auto-created strategy runtime: %s", sid)
            except ValueError as e:
                logger.warning("Could not auto-create strategy: %s", e)
            return

        for s_cfg in strategies_cfg:
            sid = s_cfg["id"]
            stype = s_cfg["type"]
            enabled = s_cfg.get("enabled", True)

            # Merge global config with strategy-specific overrides
            merged = dict(self.config)
            merged.update(s_cfg)

            try:
                strat = create_strategy(sid, stype, merged)
                if not enabled:
                    strat.enabled = False
                runtime = StrategyRuntime(strat, self.config, initial_balance)
                self.strategy_runtimes.append(runtime)
                logger.info("Strategy runtime created: %s (%s) enabled=%s", sid, stype, enabled)
            except ValueError as e:
                logger.error("Failed to create strategy '%s': %s", sid, e)

        # Build orchestrator
        if self.strategy_runtimes:
            self.orchestrator = StrategyOrchestrator(
                runtimes=self.strategy_runtimes,
                config=self.config,
                connection=self.connection,
                position_manager=self.position_manager,
                notification_manager=self.notification_manager,
            )
            logger.info(
                "StrategyOrchestrator initialized with %d runtimes",
                len(self.strategy_runtimes)
            )

    def _load_config(self, config_path: str) -> dict:
        paths_to_check = ["config_optimized.json", config_path]
        for path in paths_to_check:
            if os.path.exists(path):
                try:
                    with open(path, "r") as f:
                        content = f.read()
                        cleaned_content = re.sub(r'#.*$', '', content, flags=re.MULTILINE)
                        cfg = json.loads(cleaned_content)
                        logger.info(f"Configuration loaded from {path}")
                        return cfg
                except Exception as e:
                    logger.error(f"Failed to load {path}: {e}")
        return {"symbol": "XAUUSDm", "magic_number": 234000}

    def _save_state(self):
        with self.state_lock:
            state = {
                "position_meta": self.position_meta,
                "notified_deals": list(self.notified_deals),
                "daily_stats": {
                    "pnl": self.daily_pnl,
                    "trades": self.daily_trades,
                    "win_count": self.win_count,
                    "loss_count": self.loss_count,
                    "last_reset": self.last_reset_day.isoformat()
                },
                "max_drawdown": self.max_drawdown_reached,
            }
            # Save per-strategy runtime states
            if self.orchestrator:
                state["strategy_states"] = self.orchestrator.get_states()

        self.state_manager.save(state, self.state_file)

    def _load_state(self):
        state = self.state_manager.load(self.state_file)
        if not state:
            return

        with self.state_lock:
            self.position_meta = {int(k): v for k, v in state.get("position_meta", {}).items()}
            self.notified_deals = set(state.get("notified_deals", []))
            stats = state.get("daily_stats", {})
            self.daily_pnl = stats.get("pnl", 0.0)
            self.daily_trades = stats.get("trades", 0)
            self.win_count = stats.get("win_count", 0)
            self.loss_count = stats.get("loss_count", 0)
            self.last_reset_day = date.fromisoformat(stats.get("last_reset", date.today().isoformat()))
            self.max_drawdown_reached = state.get("max_drawdown", 0.0)

        # Restore per-strategy states
        if self.orchestrator and "strategy_states" in state:
            self.orchestrator.load_states(state["strategy_states"])

    def _reconcile_positions(self) -> None:
        if mt5 is None:
            return
        try:
            magic = int(self.config.get("magic_number", 234000))
            with MT5Connection.MT5_LOCK:
                active = mt5.positions_get()
            live_tickets = {p.ticket: p for p in active if p.magic == magic} if active else {}

            # Reconcile legacy position_meta
            with self.state_lock:
                for t in list(self.position_meta.keys()):
                    if t not in live_tickets:
                        del self.position_meta[t]
                for t, p in live_tickets.items():
                    if t not in self.position_meta:
                        risk = abs(p.price_open - p.sl) if p.sl > 0 else 0
                        self.position_meta[t] = {
                            "ticket": t, "best_price": p.price_current,
                            "partial_closed_count": 0, "risk": risk, "ai_score": 0.5
                        }

            # Reconcile per-strategy position trackers
            if self.orchestrator:
                live_set = set(live_tickets.keys())
                for runtime in self.strategy_runtimes:
                    runtime.positions.reconcile(live_set)

            self._save_state()
        except Exception as e:
            logger.error("Reconciliation failed: %s", e)

    def _write_performance_report(self):
        import csv
        file_path = "performance_report.csv"
        fieldnames = ["Date", "PnL", "Trades", "WinRate", "MaxDrawdown", "EndingBalance"]
        balance = self.connection.get_account_snapshot().get('balance', 0)
        win_rate = (self.win_count / self.daily_trades * 100) if self.daily_trades > 0 else 0
        row = {
            "Date": (date.today() - timedelta(days=1)).isoformat(),
            "PnL": f"{self.daily_pnl:.2f}",
            "Trades": self.daily_trades,
            "WinRate": f"{win_rate:.1f}%",
            "MaxDrawdown": f"{self.max_drawdown_reached:.2f}",
            "EndingBalance": f"{balance:.2f}"
        }
        try:
            write_header = not os.path.exists(file_path) or os.stat(file_path).st_size == 0
            with open(file_path, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                if write_header:
                    writer.writeheader()
                writer.writerow(row)
        except Exception as e:
            logger.error("Reporting failed: %s", e)

    def _reset_daily_stats(self):
        today = date.today()
        if today > self.last_reset_day:
            self._write_performance_report()
            with self.state_lock:
                self.daily_pnl = 0.0
                self.daily_trades = 0
                self.win_count = 0
                self.loss_count = 0
                self.last_reset_day = today
                acc = self.connection.get_account_snapshot()
                balance = acc.get("balance", 0.0)
                self.risk_manager.reset_daily_stats(balance)
                # Reset all strategy runtimes
                if self.orchestrator:
                    self.orchestrator.reset_daily(balance)
            self._save_state()

    def _detect_closed_trades(self, symbol: str):
        """Detect closed trades — routes to orchestrator for strategy attribution."""
        if self.orchestrator:
            self.orchestrator.detect_closed_trades(symbol)
            # Aggregate stats for dashboard
            total_pnl = 0.0
            total_trades = 0
            total_wins = 0
            total_losses = 0
            for rt in self.strategy_runtimes:
                summary = rt.performance.get_summary()
                total_pnl += summary.get("daily_pnl", 0)
                total_trades += summary.get("daily_trades", 0)
                total_wins += rt.performance.win_count
                total_losses += rt.performance.loss_count
            self.daily_pnl = total_pnl
            self.daily_trades = total_trades
            self.win_count = total_wins
            self.loss_count = total_losses
            return

        # Legacy fallback
        self._detect_closed_trades_legacy(symbol)

    def _detect_closed_trades_legacy(self, symbol: str):
        """Legacy closed-trade detection without strategy attribution."""
        if mt5 is None:
            return
        try:
            magic = int(self.config.get("magic_number", 234000))
            with MT5Connection.MT5_LOCK:
                active = mt5.positions_get()
            live_tickets = {p.ticket for p in active if p.magic == magic} if active else set()

            with self.state_lock:
                closed_tickets = [t for t in self.position_meta if t not in live_tickets]

            for ticket in closed_tickets:
                try:
                    with MT5Connection.MT5_LOCK:
                        deals = mt5.history_deals_get(position=ticket)
                    if deals:
                        total_pnl = sum(d.profit + d.commission + d.swap for d in deals)
                        with self.state_lock:
                            self.daily_pnl += total_pnl
                            self.daily_trades += 1
                            if total_pnl >= 0:
                                self.win_count += 1
                            else:
                                self.loss_count += 1
                            del self.position_meta[ticket]
                        self.notification_manager.notify_trade_close(
                            symbol=symbol, direction="BUY",
                            exit_price=0, pnl=total_pnl,
                            result="WIN" if total_pnl >= 0 else "LOSS"
                        )
                        logger.info("Trade closed: ticket=%s pnl=$%.2f", ticket, total_pnl)
                except Exception as e:
                    logger.error("Error processing closed ticket %s: %s", ticket, e)
                    with self.state_lock:
                        if ticket in self.position_meta:
                            del self.position_meta[ticket]

            if closed_tickets:
                self._save_state()
        except Exception as e:
            logger.error("Trade close detection failed: %s", e)

    def _manage_trailing_stops(self, symbol: str, current_bid: float,
                                current_ask: float, atr: float, last_candle: dict) -> None:
        """Route trailing stop management to orchestrator or legacy manager."""
        if self.orchestrator:
            self.orchestrator.manage_trailing_stops(symbol, current_bid, current_ask, atr, last_candle)
        else:
            self.trailing_stop_manager.manage_positions(symbol, current_bid, current_ask, atr, last_candle)

    def _startup_checks(self) -> bool:
        checks = []
        try:
            BotConfig(**self.config)
            checks.append(("Config validation", True))
        except Exception as e:
            checks.append(("Config validation", False, str(e)))

        checks.append(("MT5 connection", self.connection.connected))
        market_open = self.connection.get_market_status(self.config.get("symbol", "XAUUSDm"))
        checks.append(("Market open", True, "Yes" if market_open else "No (Waiting for market open)"))

        acc_info = self.connection.get_account_snapshot()
        balance = acc_info.get("balance", 0)
        checks.append(("Balance > $100", balance > 100))

        research = self.config.get("research_mode", False)
        if research:
            checks.append(("RESEARCH MODE", True, "Restrictions Disabled"))

        # Multi-strategy status
        enabled_count = sum(1 for rt in self.strategy_runtimes if rt.enabled)
        total_count = len(self.strategy_runtimes)
        checks.append(("Strategy Runtimes", enabled_count > 0,
                        f"{enabled_count}/{total_count} enabled"))

        logger.info("-" * 30)
        logger.info(" STARTUP HEALTH CHECKS")
        logger.info("-" * 30)
        all_passed = all(c[1] for c in checks)
        for c in checks:
            status = "[OK]" if c[1] else "[FAIL]"
            if c[0] == "Market open" and not market_open:
                status = "[WARN]"
            logger.info(f"  {status} {c[0]}" + (f" — {c[2]}" if len(c) > 2 else ""))

        return all_passed

    def run_live(self):
        """
        The main entry point for live multi-strategy trading.
        """
        if not self.connection.connect():
            logger.critical("Could not initialize MT5. Check terminal status and credentials.")
            return

        # ── Interactive Boot Menu ──
        from rich.prompt import Prompt
        from rich.console import Console
        from rich.panel import Panel
        c = Console()
        c.clear()
        c.print(Panel("[bold cyan]Trading Bot V3 — Institutional Edition[/]", expand=False))
        
        # Symbol Select
        sym_choices = {"1": "XAUUSDm", "2": "EURUSDm", "3": "GBPUSDm", "4": "USDJPYm"}
        c.print("Available Pairs: [1] XAUUSDm  [2] EURUSDm  [3] GBPUSDm  [4] USDJPYm")
        sym_choice = Prompt.ask("Select Symbol", choices=["1", "2", "3", "4"], default="1")
        selected_symbol = sym_choices[sym_choice]
        self.config["symbol"] = selected_symbol
        
        # Strategy Select
        c.print("\nAvailable Strategies: [1] Sniper V4.2  [2] SMC V4  [3] Both (Portfolio)")
        strat_choice = Prompt.ask("Select Strategy", choices=["1", "2", "3"], default="3")
        
        for rt in self.strategy_runtimes:
            if strat_choice == "1":
                rt.strategy.enabled = (rt.strategy_id == "sniper_v1")
            elif strat_choice == "2":
                rt.strategy.enabled = (rt.strategy_id == "smc_v1")
            else:
                rt.strategy.enabled = True
        
        c.print(f"\n[bold green]Booting [{selected_symbol}] Live Feed...[/]\n")

        if not self._startup_checks():
            logger.critical("Startup checks failed. Exiting.")
            return

        symbol = self.config.get("symbol", selected_symbol)
        start_health_server(self, port=8081)

        self.dashboard.start()
        self.running = True
        logger.info("Bot starting LIVE EXECUTIONS — %d strategy runtimes",
                     len(self.strategy_runtimes))

        # Log individual strategies
        for rt in self.strategy_runtimes:
            logger.info("  → %s (enabled=%s)", rt.strategy_id, rt.enabled)

        self._reconcile_positions()

        try:
            while not self._shutdown_event.is_set():
                try:
                    start_cycle = time.time()
                    self._reset_daily_stats()

                    if not self.connection.ensure_connected():
                        logger.warning("MT5 connection lost. Attempting reconnect...")
                        self._shutdown_event.wait(10.0)
                        continue

                    # Fetch fresh account and tick data
                    acc = self.connection.get_account_snapshot()
                    tick = None
                    try:
                        with MT5Connection.MT5_LOCK:
                            tick = mt5.symbol_info_tick(symbol) if mt5 else None
                    except Exception as e:
                        logger.warning("Tick fetch failed: %s", e)

                    # Update Dashboard State
                    self.dashboard.account_info = acc
                    if tick:
                        self.dashboard.tick = {
                            "price": tick.bid,
                            "spread": (tick.ask - tick.bid) / (mt5.symbol_info(symbol).point or 0.01)
                        }

                    self.dashboard.daily_pnl = self.daily_pnl
                    self.dashboard.daily_trades = self.daily_trades
                    self.dashboard.win_count = self.win_count
                    self.dashboard.loss_count = self.loss_count
                    self.dashboard.positions = self.connection.get_positions(symbol)

                    # Detect closed trades (orchestrator routes to owning strategy)
                    self._detect_closed_trades(symbol)

                    # Fetch candles
                    h1_candles = self.data_fetcher.fetch_candles(symbol, "H1", 200)
                    m15_candles = self.data_fetcher.fetch_candles(symbol, "M15", 200)
                    m5_candles = self.data_fetcher.fetch_candles(symbol, "M5", 500)
                    d1_candles = self.data_fetcher.fetch_candles(symbol, "D1", 50)

                    # Trailing Stops (per-strategy via orchestrator)
                    if tick and len(m5_candles) > 30:
                        atr = self.strategy._calculate_atr(m5_candles, 14)
                        self._manage_trailing_stops(symbol, tick.bid, tick.ask, atr, m5_candles[-1])
                        if self.orchestrator:
                            self.orchestrator.manage_partials(symbol, tick.bid, tick.ask)

                    if len(m5_candles) > 30:
                        current_price = m5_candles.close[-1]
                        session = self.strategy.get_session_from_hour(
                            datetime.now(timezone.utc).hour
                        )
                        self.dashboard.session = session

                        # ── Multi-Strategy Execution ──
                        if self.orchestrator:
                            self.orchestrator.execute_cycle(
                                symbol, h1_candles, m15_candles, m5_candles,
                                d1_candles, current_price, session
                            )
                            analysis = self.orchestrator.last_analysis
                        else:
                            # Legacy fallback
                            self.execution_pipeline.execute_cycle(
                                symbol, h1_candles, m15_candles, m5_candles,
                                d1_candles, current_price, session
                            )
                            analysis = self.execution_pipeline.last_analysis

                        self.dashboard.h4_trend = analysis.get("trend", "NEUTRAL")
                        self.dashboard.m30_structure = analysis.get("regime", "NEUTRAL")
                        self.dashboard.analysis_context = analysis

                    self.dashboard.fetch_ms = int((time.time() - start_cycle) * 1000)
                    self.dashboard.update()

                except Exception as cycle_err:
                    logger.error("Cycle error (recovering): %s", cycle_err, exc_info=True)
                    self._save_state()

                self._shutdown_event.wait(5.0)
        except KeyboardInterrupt:
            logger.info("Ctrl+C received — shutting down gracefully")
        finally:
            self.running = False
            self.dashboard.stop()
            self._shutdown_event.set()
            self._save_state()
            self.connection.disconnect()
            logger.info("Shutdown complete.")

    def run_optimization(self, symbol="XAUUSDm", start_date=None, end_date=None,
                         count=10000, mode="anchored"):
        logger.info(f"Starting WFO Optimization for {symbol} (Mode: {mode})")
        if not self.connection.connect():
            return

        try:
            if start_date and end_date:
                dt_from = datetime.fromisoformat(start_date).replace(tzinfo=timezone.utc)
                dt_to = datetime.fromisoformat(end_date).replace(tzinfo=timezone.utc)
                h1 = self.data_fetcher.fetch_candles_range(symbol, "H1", dt_from, dt_to)
                m15 = self.data_fetcher.fetch_candles_range(symbol, "M15", dt_from, dt_to)
                m5 = self.data_fetcher.fetch_candles_range(symbol, "M5", dt_from, dt_to)
                d1 = self.data_fetcher.fetch_candles_range(symbol, "D1", dt_from, dt_to)
            else:
                h1 = self.data_fetcher.fetch_candles(symbol, "H1", count)
                m15 = self.data_fetcher.fetch_candles(symbol, "M15", count)
                m5 = self.data_fetcher.fetch_candles(symbol, "M5", count)
                d1 = self.data_fetcher.fetch_candles(symbol, "D1", count)

            wfo = WalkForwardValidator(self.config, self.strategy)
            results = wfo.run_validation(symbol, h1, m15, m5, d1, mode=mode)

            print("\n" + "=" * 40)
            print(" WFO OPTIMIZATION COMPLETE ")
            print("=" * 40)
            print(f"Windows Validated: {len(results)}")
            avg_cons = sum(r['consistency'] for r in results) / len(results) if results else 0
            print(f"Aggregate OOS Consistency: {avg_cons:.2f}")
            if avg_cons >= 0.7:
                print("[SUCCESS] Institutional Target (>0.7) Reached!")
            else:
                print("[WARNING] Robustness below target. Further calibration required.")
            print("Best parameters saved to config_optimized.json")

        finally:
            self.connection.disconnect()


if __name__ == "__main__":
    setup_logging(console=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--backtest", action="store_true")
    parser.add_argument("--optimize", action="store_true", help="Run Walk-Forward Optimization")
    parser.add_argument("--mode", type=str, choices=["anchored", "rolling"], default="anchored")
    parser.add_argument("--symbol", type=str, default="XAUUSDm")
    parser.add_argument("--from", dest="start_date", type=str, help="YYYY-MM-DD", default=None)
    parser.add_argument("--to", dest="end_date", type=str, help="YYYY-MM-DD", default=None)
    parser.add_argument("--count", type=int, default=10000)
    args = parser.parse_args()

    bot = TradingBot()
    if args.optimize:
        bot.run_optimization(args.symbol, args.start_date, args.end_date, args.count, args.mode)
    elif args.backtest:
        rprint("[bold red]Notice:[/] use `python backtest.py --from YYYY-MM-DD --to YYYY-MM-DD` for standalone backtesting.")
    else:
        bot.run_live()
