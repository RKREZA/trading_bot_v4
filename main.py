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
from core.broker_clock import BrokerClock
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
from core.state_manager import SecureStateManager
from core.news_filter import NewsFilter
from core.orchestrator import StrategyOrchestrator as PortfolioOrchestrator
from core.portfolio_manager import PortfolioManager
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

        # ── Broker Clock: Authoritative time source ──
        self.broker_clock = BrokerClock()

        self.analysis_logger = AnalysisLogger(max_entries=100, broker_clock=self.broker_clock)

        # Bridge standard logging to the dashboard logger
        bridged_handler = AnalysisLoggerHandler(self.analysis_logger)
        bridged_handler.setFormatter(logging.Formatter("%(name)s: %(message)s"))
        logging.getLogger("trading_bot").addHandler(bridged_handler)

        self.ai_advisor = AIAdvisor(self.config, self.analysis_logger, broker_clock=self.broker_clock)
        self.dashboard = Dashboard(self.config, self.analysis_logger, broker_clock=self.broker_clock)
        self.connection = MT5Connection()
        self.connection.config = self.config
        self.position_manager = PositionManager(self.connection)
        self.data_fetcher = DataFetcher()
        self.notification_manager = NotificationManager(self.config)
        self.state_manager = SecureStateManager()
        self.news_filter = NewsFilter(broker_clock=self.broker_clock)

        self.daily_pnl = 0.0
        self.daily_trades = 0
        self.win_count = 0
        self.loss_count = 0
        self.consecutive_losses = {"LONDON": 0, "NEW_YORK": 0, "LONDON/NY": 0, "TOKYO": 0}
        self.running = False
        self.last_reset_day = None  # Will be set after first broker_clock sync
        self.max_drawdown_reached = 0.0
        self.last_logged_session: Optional[str] = None
        self.notified_deals = set()
        self.state_file = "bot_state.json"
        self._start_time = time.time()  # For health endpoint uptime
        self.state_lock = threading.Lock()
        self.execution_queue = queue.Queue()
        self._shutdown_event = threading.Event()

        self.strategy_runtimes: List[StrategyRuntime] = []
        self.orchestrator: Optional[PortfolioOrchestrator] = None
        self.portfolio_manager: Optional[PortfolioManager] = None
        self._build_portfolio_system()

        self._load_state()

    def _build_portfolio_system(self):
        """Initialize the new multi-strategy portfolio system."""
        self.risk_manager = RiskManager(self.config, broker_clock=self.broker_clock)
        self.risk_manager.silent = True

        strategies = []
        strategies_config = self.config.get("strategies", [])
        for strat_cfg in strategies_config:
            if not strat_cfg.get("enabled", True):
                continue
                
            sid = strat_cfg.get("id")
            if sid == "sniper_v1":
                from strategies.sniper_strategy import SniperStrategy
                strategy = SniperStrategy(sid, strat_cfg)
            elif sid == "smc_v1":
                from strategies.smc_strategy import SMCStrategy
                strategy = SMCStrategy(sid, strat_cfg)
            else:
                continue
            strategies.append(strategy)

        self.orchestrator = PortfolioOrchestrator(strategies, self.data_fetcher)
        self.portfolio_manager = PortfolioManager(self.risk_manager, self.config, self.state_manager)
        
        logger.info(f"Portfolio system initialized with {len(strategies)} strategies.")

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
                "notified_deals": list(self.notified_deals),
                "daily_stats": {
                    "pnl": self.daily_pnl,
                    "trades": self.daily_trades,
                    "win_count": self.win_count,
                    "loss_count": self.loss_count,
                    "last_reset": self.last_reset_day.isoformat() if self.last_reset_day else ""
                },
                "max_drawdown": self.max_drawdown_reached,
                "positions": []
            }
            # Save per-strategy runtime states
            if self.orchestrator:
                state["strategy_states"] = self.orchestrator.get_states()
                
            # Track Active Positions across all strategies
            if mt5 is not None:
                magic = int(self.config.get("magic_number", 234000))
                with MT5Connection.MT5_LOCK:
                    active = mt5.positions_get()
                if active:
                    for p in active:
                        if p.magic == magic:
                            state["positions"].append({
                                "symbol": p.symbol,
                                "strategy": p.comment if p.comment else "unknown",
                                "entry": p.price_open,
                                "direction": "BUY" if p.type == mt5.POSITION_TYPE_BUY else "SELL"
                            })

        self.state_manager.save(state, self.state_file)

    def _load_state(self):
        state = self.state_manager.load(self.state_file)
        if not state:
            return

        with self.state_lock:
            self.notified_deals = set(state.get("notified_deals", []))
            stats = state.get("daily_stats", {})
            self.daily_pnl = stats.get("pnl", 0.0)
            self.daily_trades = stats.get("trades", 0)
            self.win_count = stats.get("win_count", 0)
            self.loss_count = stats.get("loss_count", 0)
            last_reset_str = stats.get("last_reset", "")
            self.last_reset_day = date.fromisoformat(last_reset_str) if last_reset_str else None
            self.max_drawdown_reached = state.get("max_drawdown", 0.0)

        # Restore per-strategy states
        if self.orchestrator and "strategy_states" in state:
            self.orchestrator.load_states(state["strategy_states"])

    def _reconcile_positions(self) -> None:
        if mt5 is None or not self.orchestrator:
            return
        try:
            magic = int(self.config.get("magic_number", 234000))
            with MT5Connection.MT5_LOCK:
                active = mt5.positions_get()
            live_tickets = {p.ticket: p for p in active if p.magic == magic} if active else {}

            # Delegate strictly to orchestrator runtimes
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
            "Date": (self.broker_clock.today() - timedelta(days=1)).isoformat(),
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
        today = self.broker_clock.today()
        if self.last_reset_day is None:
            self.last_reset_day = today
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
        c.print("\nAvailable Strategies: [1] Sniper V4.2  [2] SMC V4")
        strat_choice = Prompt.ask("Select Strategy", choices=["1", "2"], default="1")
        
        # Explicitly pass selected info to Dashboard
        self.dashboard.selected_symbol = selected_symbol
        self.dashboard.selected_symbol_title = selected_symbol
        self.dashboard.selected_strategy_title = "Sniper V4.2" if strat_choice == "1" else "SMC V4"
        
        for rt in self.strategy_runtimes:
            if strat_choice == "1":
                rt.strategy.enabled = (rt.strategy_id == "sniper_v1")
            elif strat_choice == "2":
                rt.strategy.enabled = (rt.strategy_id == "smc_v1")
        
        # Disable orchestrator if only 1 strategy is needed? No, orchestrator handles both natively, but we disabled the unused one.
        c.print(f"\n[bold green]Booting {self.dashboard.selected_strategy_title} on [{selected_symbol}] Live Feed...[/]\n")

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
                    start_cycle = self.broker_clock.timestamp()

                    if not self.connection.ensure_connected():
                        logger.warning("MT5 connection lost. Attempting reconnect...")
                        self.dashboard.account_info = {"connected": False}
                        self.dashboard.update()
                        self._shutdown_event.wait(10.0)
                        continue

                    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                    # PHASE 1: Always sync dashboard (real-time data)
                    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                    acc = self.connection.get_account_snapshot()
                    tick = None
                    try:
                        with MT5Connection.MT5_LOCK:
                            tick = mt5.symbol_info_tick(symbol) if mt5 else None
                    except Exception as e:
                        logger.warning("Tick fetch failed: %s", e)

                    # Sync broker clock (handles stale ticks automatically)
                    self.broker_clock.sync(symbol, MT5Connection.MT5_LOCK)
                    self._reset_daily_stats()

                    # Update dashboard — account, price, positions, PnL
                    self.dashboard.account_info = acc
                    if tick:
                        sym_info = mt5.symbol_info(symbol)
                        point = sym_info.point if sym_info else 0.01
                        self.dashboard.tick = {
                            "price": tick.bid,
                            "spread": (tick.ask - tick.bid) / point
                        }

                    self.dashboard.daily_pnl = self.daily_pnl
                    self.dashboard.daily_trades = self.daily_trades
                    self.dashboard.win_count = self.win_count
                    self.dashboard.loss_count = self.loss_count
                    self.dashboard.positions = self.connection.get_positions(symbol)

                    # Always detect closed trades (even during quiet periods)
                    if self.orchestrator:
                        self.orchestrator.detect_closed_trades(symbol)

                    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                    # PHASE 2: Market open check — idle if closed
                    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                    market_open = self.connection.get_market_status(symbol)
                    self.dashboard.market_open = market_open

                    if not market_open:
                        self.dashboard.session = "CLOSED"
                        self.dashboard.fetch_status = "[dim]Market closed — waiting...[/]"
                        self.dashboard.fetch_ms = 0
                        self.dashboard.update()
                        self._shutdown_event.wait(5.0)  # Faster poll to catch exactly on open
                        continue

                    # Market is open — clear any stale status
                    self.dashboard.fetch_status = ""

                    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                    # PHASE 3: Active trading — analysis & execution
                    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

                    # Fetch candles
                    h1_candles = self.data_fetcher.fetch_candles(symbol, "H1", 200)
                    m15_candles = self.data_fetcher.fetch_candles(symbol, "M15", 200)
                    m5_candles = self.data_fetcher.fetch_candles(symbol, "M5", 500)
                    d1_candles = self.data_fetcher.fetch_candles(symbol, "D1", 50)

                    # Trailing Stops (per-strategy via orchestrator)
                    if tick and len(m5_candles) > 30:
                        if self.orchestrator:
                            atr = self.orchestrator._preprocessing_engine._calculate_atr(m5_candles, 14)
                            self.orchestrator.manage_trailing_stops(symbol, tick.bid, tick.ask, atr, m5_candles[-1], session)
                            self.orchestrator.manage_partials(symbol, tick.bid, tick.ask)

                    if len(m5_candles) > 30:
                        current_price = m5_candles.close[-1]
                        session = self.orchestrator._preprocessing_engine.get_session_from_hour(
                            self.broker_clock.hour()
                        )
                        self.dashboard.session = session

                        # ── News Filter ──
                        blocked, news_reason = self.news_filter.is_trading_blocked(symbol)
                        self.dashboard.news_events = self.news_filter.get_upcoming_news(symbol)
                        self.dashboard.news_blocked = blocked
                        self.dashboard.news_reason = news_reason

                        if blocked:
                            self.dashboard.fetch_ms = int((self.broker_clock.timestamp() - start_cycle) * 1000)
                            self.dashboard.update()
                            self._shutdown_event.wait(5.0)
                            continue

                        # ── Multi-Strategy Execution ──
                        if self.orchestrator:
                            # 1. Update risk manager state
                            self.risk_manager.update_state(
                                equity=acc.get("equity", 0.0),
                                balance=acc.get("balance", 0.0),
                                daily_trades=self.daily_trades,
                                consecutive_losses=self.consecutive_losses[session] if hasattr(self, 'consecutive_losses') and session in self.consecutive_losses else 0
                            )
                            
                            # 2. Run run cycle to collect signals
                            signals = self.orchestrator.run_cycle(symbol, session, current_price, self.broker_clock)
                            
                            # 3. Portfolio Manager processes signals
                            open_positions = self.connection.get_positions(symbol)
                            approved_trades = self.portfolio_manager.process_signals(signals, acc, open_positions)
                            
                            # 4. Centralized Execution
                            for trade in approved_trades:
                                self._execute_portfolio_trade(trade)
                            
                            analysis = {"trend": "PORTFOLIO", "regime": f"{len(approved_trades)} approved"}
                        else:
                            analysis = {}

                        self.dashboard.h4_trend = analysis.get("trend", "NEUTRAL")
                        self.dashboard.m30_structure = analysis.get("regime", "NEUTRAL")
                        self.dashboard.analysis_context = analysis

                    self.dashboard.fetch_ms = int((self.broker_clock.timestamp() - start_cycle) * 1000)
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

    def _execute_portfolio_trade(self, signal: dict):
        """Standardized execution for signals approved by PortfolioManager."""
        symbol = signal["symbol"]
        strategy_id = signal["strategy"]
        direction = signal["direction"]
        entry = signal["entry"]
        sl = signal["sl"]
        tp = signal["tp"]
        risk_pct = signal["risk"]  # already fraction 0.0-1.0
        scale = signal.get("allocation_scale", 1.0)
        
        # Calculate Lot Size
        acc = self.connection.get_account_snapshot()
        balance = acc.get("balance", 0.0)
        
        sym_info = self.connection.get_symbol_info(symbol)
        if not sym_info:
            return

        point = sym_info.get("point", 0.01)
        sl_dist = abs(entry - sl)
        
        # Enforce Minimum SL
        if sl_dist < point * 10:
            logger.warning(f"[{strategy_id}] SL too tight ({sl_dist}). Scaling out.")
            return

        # Portfolio Allocation Math: allocated_equity = total_equity * allocation[strategy]
        # Then risk is applied to that allocated equity.
        risk_dollar = (balance * scale) * risk_pct
        
        from core.lot_calculator import LotCalculator
        lot = LotCalculator.calculate(
            risk_amount=risk_dollar,
            sl_distance=sl_dist,
            tick_size=point,
            tick_value=sym_info.get("trade_tick_value", 1.0),
            volume_min=sym_info.get("volume_min", 0.01),
            volume_max=sym_info.get("volume_max", 100.0),
            volume_step=sym_info.get("volume_step", 0.01),
        )

        from core.order_tagger import OrderTagger
        trade_id = f"{int(time.time())}"
        comment = OrderTagger.create_comment(strategy_id, trade_id)

        logger.info(f"PORTFOLIO EXEC: {strategy_id} | {direction} {symbol} | Lot: {lot} | Comment: {comment}")

        from core.strategy_engine import TradeSignal as _LegacySignal
        # Connection still expects the TradeSignal object for now
        legacy_sig = _LegacySignal(direction, entry, sl, tp, session=signal.get("session", "UNKNOWN"))
        legacy_sig.tp1_price = signal.get("tp1", 0.0)
        legacy_sig.tp2_price = signal.get("tp2", 0.0)

        result = self.connection.place_order(symbol, legacy_sig, lot, comment=comment)
        
        if result:
            self.daily_trades += 1
            if self.notification_manager:
                self.notification_manager.notify_trade_open(
                    symbol=symbol, direction=direction,
                    entry=entry, lot=lot, sl=sl, tp=tp
                )
            # Register in state? 
            # We'll use MT5 comments for reconciliation now.
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

            engine = StrategyEngine(self.config, self.analysis_logger)
            wfo = WalkForwardValidator(self.config, engine)
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
    setup_logging(console=False)
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
