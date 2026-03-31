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
from core.risk_manager import RiskManager
from core.notifications import NotificationManager
from core.regime import MarketRegime
from dashboard import Dashboard, AnalysisLogger

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
        self.ai_advisor = AIAdvisor(self.config, self.analysis_logger)
        self.strategy = StrategyEngine(self.config, self.analysis_logger)
        self.dashboard = Dashboard(self.config, self.analysis_logger)
        self.connection = MT5Connection()
        self.connection.config = self.config
        self.position_manager = PositionManager(self.connection)
        self.data_fetcher = DataFetcher()
        self.risk_manager = RiskManager(self.config)
        self.notification_manager = NotificationManager(self.config)

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
        try:
            with open(self.state_file, "w") as f:
                json.dump(state, f)
        except Exception as e:
            logger.error("Failed to save state: %s", e)

    def _load_state(self):
        if not os.path.exists(self.state_file): return
        try:
            with open(self.state_file, "r") as f:
                state = json.load(f)
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
        except Exception as e:
            logger.error("Failed to load state: %s", e)

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
            self._save_state()

    def _manage_trailing_stops(self, symbol: str, m30_candles: list) -> None:
        if mt5 is None: return
        positions = mt5.positions_get(symbol=symbol)
        if not positions: return
        for pos in positions:
            if pos.magic != int(self.config.get("magic_number", 234000)): continue
            # Trailing/Partial Profit logic...
            pass

    def _execute_partial_close(self, ticket, symbol, pos_type, volume, magic, comment):
        req = {"action": mt5.TRADE_ACTION_DEAL, "symbol": symbol, "volume": volume, "type": mt5.ORDER_TYPE_SELL if pos_type == 0 else mt5.ORDER_TYPE_BUY, "position": ticket, "price": mt5.symbol_info_tick(symbol).bid if pos_type == 0 else mt5.symbol_info_tick(symbol).ask, "deviation": 20, "magic": magic, "comment": comment, "type_time": mt5.ORDER_TIME_GTC, "type_filling": self.connection.get_filling_mode(symbol)}
        mt5.order_send(req)

    def _modify_sl_tp(self, ticket, symbol, sl, tp):
        req = {"action": mt5.TRADE_ACTION_SLTP, "symbol": symbol, "position": ticket, "sl": round(float(sl), 2), "tp": round(float(tp), 2)}
        mt5.order_send(req)

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
        balance = self.connection.account_info.get("balance", 0)
        checks.append(("Balance > $100", balance > 100))
        
        all_passed = all(c[1] for c in checks)
        for c in checks:
            status = "✓" if c[1] else "✗"
            logger.info(f"  {status} {c[0]}" + (f" — {c[2]}" if len(c) > 2 else ""))
            
        return all_passed

    def run_live(self):
        if not self.connection.connect(): return
        self._startup_checks()
        start_health_server(self, port=8081)
        self.running = True
        while self.running:
            self._reset_daily_stats()
            time.sleep(1)

    def run_backtest(self, symbol="XAUUSDm"):
        print(f"Project 10/10 Final Validation: {symbol}")
        h4 = self.data_fetcher.fetch_candles(symbol, "H4", 100)
        h1 = self.data_fetcher.fetch_candles(symbol, "H1", 100)
        m30 = self.data_fetcher.fetch_candles(symbol, "M30", 100)
        m5 = self.data_fetcher.fetch_candles(symbol, "M5", 100)
        d1 = self.data_fetcher.fetch_candles(symbol, "D1", 100)
        engine = BacktestEngine(self.config, self.strategy)
        return engine.run(symbol, h4, h1, m30, m5, d1)

if __name__ == "__main__":
    bot = TradingBot()
    if "--backtest" in sys.argv: bot.run_backtest()
    else: bot.run_live()
