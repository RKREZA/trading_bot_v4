"""
TRADING BOT V3 - Main Entry Point
Orchestrates MT5 connection, data fetching, strategy analysis, and dashboard.
"""

import argparse
import json
import logging
import os
import sys
import time
import threading
import queue
import itertools
import copy
import pandas as pd
import numpy as np
from datetime import datetime, timezone, date, timedelta
from typing import Optional, List, Tuple
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
from core.ai_filter import AIFilter
from core.risk_manager import RiskManager
from core.notifications import NotificationManager
from core.regime import MarketRegime
from core.state_manager import SecureStateManager
from core.execution_pipeline import ExecutionPipeline
from core.trailing_stop import TrailingStopManager
from dashboard import Dashboard, AnalysisLogger, AnalysisLoggerHandler

# Load .env file if it exists
load_dotenv()

logger = logging.getLogger("trading_bot.main")


class TradingBot:
    """
    Main trading bot orchestrator.
    Coordinates Connection, Data, Strategy, and Risk management.
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
        
        self.execution_pipeline = ExecutionPipeline(
            config=self.config,
            connection=self.connection,
            position_manager=self.position_manager,
            strategy=self.strategy,
            ai_advisor=self.ai_advisor,
            risk_manager=self.risk_manager,
            notification_manager=self.notification_manager
        )
        self.trailing_stop_manager = TrailingStopManager(
            config=self.config,
            connection=self.connection,
            position_meta=self.position_meta,
            state_lock=self.state_lock
        )
        
        self._load_state()

    def _load_config(self, config_path: str) -> dict:
        try:
            with open(config_path, "r") as f:
                return json.load(f)
        except Exception:
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
                "max_drawdown": self.max_drawdown_reached
            }
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

    def _reconcile_positions(self) -> None:
        if mt5 is None: return
        try:
            magic = int(self.config.get("magic_number", 234000))
            with MT5Connection.MT5_LOCK:
                active = mt5.positions_get()
            live_tickets = {p.ticket: p for p in active if p.magic == magic} if active else {}
            with self.state_lock:
                for t in list(self.position_meta.keys()):
                    if t not in live_tickets: del self.position_meta[t]
                for t, p in live_tickets.items():
                    if t not in self.position_meta:
                        risk = abs(p.price_open - p.sl) if p.sl > 0 else 0
                        self.position_meta[t] = {"ticket": t, "best_price": p.price_current, "partial_closed_count": 0, "risk": risk, "ai_score": 0.5}
            self._save_state()
        except Exception as e:
            logger.error("Reconciliation failed: %s", e)

    def _write_performance_report(self):
        import csv
        file_path = "performance_report.csv"
        fieldnames = ["Date", "PnL", "Trades", "WinRate", "MaxDrawdown", "EndingBalance"]
        balance = self.connection.get_account_snapshot().get('balance', 0)
        win_rate = (self.win_count / self.daily_trades * 100) if self.daily_trades > 0 else 0
        row = {"Date": (date.today() - timedelta(days=1)).isoformat(), "PnL": f"{self.daily_pnl:.2f}", "Trades": self.daily_trades, "WinRate": f"{win_rate:.1f}%", "MaxDrawdown": f"{self.max_drawdown_reached:.2f}", "EndingBalance": f"{balance:.2f}"}
        try:
            with open(file_path, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                if not os.path.exists(file_path) or os.stat(file_path).st_size == 0: writer.writeheader()
                writer.writerow(row)
        except Exception as e:
            logger.error("Reporting failed: %s", e)

    def _reset_daily_stats(self):
        today = date.today()
        if today > self.last_reset_day:
            self._write_performance_report()
            with self.state_lock:
                self.daily_pnl = 0.0; self.daily_trades = 0; self.win_count = 0; self.loss_count = 0; self.last_reset_day = today
                # Reset RiskManager stats (Equity tracking)
                acc = self.connection.get_account_snapshot()
                self.risk_manager.reset_daily_stats(acc.get("balance", 0.0))
            self._save_state()

    def _manage_trailing_stops(self, symbol: str, current_bid: float, current_ask: float) -> None:
        self.trailing_stop_manager.manage_positions(symbol, current_bid, current_ask)

    def _startup_checks(self) -> bool:
        """Run before entering the main loop."""
        checks = []
        try:
            BotConfig(**self.config)
            checks.append(("Config validation", True))
        except Exception as e:
            checks.append(("Config validation", False, str(e)))
            
        checks.append(("MT5 connection", self.connection.connected))
        market_open = self.connection.get_market_status(self.config.get("symbol", "XAUUSDm"))
        checks.append(("Market open", market_open))
        
        acc_info = self.connection.get_account_snapshot()
        balance = acc_info.get("balance", 0)
        checks.append(("Balance > $100", balance > 100))
        
        logger.info("-" * 30)
        logger.info(" STARTUP HEALTH CHECKS")
        logger.info("-" * 30)
        all_passed = all(c[1] for c in checks)
        for c in checks:
            status = "[OK]" if c[1] else "[FAIL]"
            logger.info(f"  {status} {c[0]}" + (f" — {c[2]}" if len(c) > 2 else ""))
            
        return all_passed

    def run_live(self):
        if not self.connection.connect(): 
            logger.critical("Could not initialize MT5. Check terminal status and credentials.")
            return
            
        if not self._startup_checks():
            logger.critical("Startup checks failed. Exiting.")
            return

        symbol = self.config.get("symbol", "XAUUSDm")
        start_health_server(self, port=8081)
        
        # Start the CLI Dashboard
        self.dashboard.start()
        self.running = True
        logger.info("Bot starting LIVE EXECUTIONS")
        
        try:
            while not self._shutdown_event.is_set():
                start_cycle = time.time()
                self._reset_daily_stats()
                
                # Fetch fresh account and tick data
                acc = self.connection.get_account_snapshot()
                tick = mt5.symbol_info_tick(symbol) if mt5 else None
                
                # Update Dashboard State
                self.dashboard.account_info = acc
                if tick:
                    self.dashboard.tick = {"price": tick.bid, "spread": (tick.ask - tick.bid) / (mt5.symbol_info(symbol).point or 0.01)}
                
                self.dashboard.daily_pnl = self.daily_pnl
                self.dashboard.daily_trades = self.daily_trades
                self.dashboard.win_count = self.win_count
                self.dashboard.loss_count = self.loss_count
                self.dashboard.positions = self.connection.get_positions(symbol)
                
                # Update tick prices for trailing
                if tick:
                    self._manage_trailing_stops(symbol, tick.bid, tick.ask)

                # Fetch candles securely
                d1_candles = self.data_fetcher.fetch_candles(symbol, "D1", 100)
                h4_candles = self.data_fetcher.fetch_candles(symbol, "H4", 100)
                h1_candles = self.data_fetcher.fetch_candles(symbol, "H1", 100)
                m30_candles = self.data_fetcher.fetch_candles(symbol, "M30", 100)
                m5_candles = self.data_fetcher.fetch_candles(symbol, "M5", 100)
                
                if len(m30_candles) > 0:
                    current_price = m30_candles.close[-1]
                    session = "NEW_YORK" # simplified fallback if TimeManager missing
                    if hasattr(self.strategy, 'get_session_from_hour'):
                        session = self.strategy.get_session_from_hour(datetime.now(timezone.utc).hour)
                    
                    self.dashboard.session = session
                    
                    # Log to dashboard if needed
                    # self.analysis_logger.log(f"Analyzed {symbol} at {current_price}")
                        
                    self.execution_pipeline.execute_cycle(
                        symbol, m30_candles, h1_candles, h4_candles, m5_candles, d1_candles, current_price, session
                    )
                
                # Final Dashboard update for the cycle
                self.dashboard.fetch_ms = int((time.time() - start_cycle) * 1000)
                self.dashboard.update()
                
                self._shutdown_event.wait(5.0)  # Sleep 5 seconds between signal checks
        except KeyboardInterrupt:
            logger.info("Ctrl+C received — shutting down gracefully")
        finally:
            self.running = False
            self.dashboard.stop()
            self._shutdown_event.set()
            self._save_state()
            self.connection.disconnect()
            logger.info("Shutdown complete.")

    def run_backtest(self, symbol="XAUUSDm", start_date=None, end_date=None, count=10000):
        print(f"Project 10/10 Final Validation: {symbol}")
        if not self.connection.connect():
            logger.error("Failed to connect to MT5 for backtesting.")
            return None
        
        # Inject live symbol info for accurate backtest lot sizing
        sym_info = self.data_fetcher.get_symbol_info(symbol)
        if sym_info:
            self.config.setdefault("symbols_config", {}).setdefault(symbol, {})
            s_cfg = self.config["symbols_config"][symbol]
            s_cfg["tick_size"] = sym_info["point"]
            # Adjusted: use contract_size * point as fallback for tick_value
            s_cfg["tick_value"] = sym_info.get("trade_tick_value", sym_info["contract_size"] * sym_info["point"])
            s_cfg["point"] = sym_info["point"]
            s_cfg["contract_size"] = sym_info["contract_size"]
            s_cfg["lot_step"] = sym_info["lot_step"]
            s_cfg["min_lot"] = sym_info["min_lot"]
            logger.info(f"[Backtest] Injected live symbol info for {symbol}")

        try:
            if start_date and end_date:
                # Use UTC for backtest date range to match candles
                dt_from = datetime.fromisoformat(start_date).replace(tzinfo=timezone.utc)
                dt_to = datetime.fromisoformat(end_date).replace(tzinfo=timezone.utc)
                h4 = self.data_fetcher.fetch_candles_range(symbol, "H4", dt_from, dt_to)
                h1 = self.data_fetcher.fetch_candles_range(symbol, "H1", dt_from, dt_to)
                m30 = self.data_fetcher.fetch_candles_range(symbol, "M30", dt_from, dt_to)
                m5 = self.data_fetcher.fetch_candles_range(symbol, "M5", dt_from, dt_to)
                d1 = self.data_fetcher.fetch_candles_range(symbol, "D1", dt_from, dt_to)
            else:
                h4 = self.data_fetcher.fetch_candles(symbol, "H4", count)
                h1 = self.data_fetcher.fetch_candles(symbol, "H1", count)
                m30 = self.data_fetcher.fetch_candles(symbol, "M30", count)
                m5 = self.data_fetcher.fetch_candles(symbol, "M5", count)
                d1 = self.data_fetcher.fetch_candles(symbol, "D1", count)
                
            # Backtest Strategy Setup
            engine = BacktestEngine(self.config, self.strategy)

            
            # Use dedicated AIFilter (with backtest mode) instead of live AIAdvisor
            bt_ai_filter = AIFilter(self.config)
            bt_ai_filter.backtest_mode = True
            engine.ai_filter = bt_ai_filter
            
            return engine.run(symbol, h4, h1, m30, m5, d1)
        finally:
            self.connection.disconnect()

if __name__ == "__main__":
    setup_logging(console=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--backtest", action="store_true")
    parser.add_argument("--symbol", type=str, default="XAUUSDm")
    parser.add_argument("--from", dest="start_date", type=str, help="YYYY-MM-DD", default=None)
    parser.add_argument("--to", dest="end_date", type=str, help="YYYY-MM-DD", default=None)
    parser.add_argument("--count", type=int, default=10000)
    args = parser.parse_args()

    bot = TradingBot()
    if args.backtest:
        bot.run_backtest(args.symbol, args.start_date, args.end_date, args.count)
    else:
        bot.run_live()
