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
from typing import Optional, List, Tuple, Dict, Any

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import print as rprint
from rich.live import Live

from core.walk_forward import WalkForwardValidation
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
    The central orchestrator of the Trading Bot V3 system.
    Responsible for initializing all core components, managing the life cycle 
    of the trading process (Live, Backtest, or Optimization), and ensuring 
    state persistence and thread safety.
    """

    def __init__(self, config_path: str = "config.json"):
        """
        Initializes the TradingBot and all its sub-components.
        
        Args:
            config_path (str): Path to the JSON configuration file.
        """
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

    def _load_config(self, config_path: str) -> dict:
        # Priority: config_optimized.json > passed config_path > default
        paths_to_check = ["config_optimized.json", config_path]
        for path in paths_to_check:
            if os.path.exists(path):
                try:
                    with open(path, "r") as f:
                        cfg = json.load(f)
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

    def _manage_trailing_stops(self, symbol: str, current_bid: float, current_ask: float, atr: float, last_candle: dict) -> None:
        self.trailing_stop_manager.manage_positions(symbol, current_bid, current_ask, atr, last_candle)

    def _startup_checks(self) -> bool:
        """
        Performs a set of critical health checks (Config, MT5, Market, Balance) 
        before allowing the bot to enter the live trading loop.
        
        Returns:
            bool: True if all critical checks pass.
        """
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
        
        research = self.config.get("research_mode", False)
        if research:
            checks.append(("RESEARCH MODE", True, "Restrictions Disabled"))
        
        logger.info("-" * 30)
        logger.info(" STARTUP HEALTH CHECKS")
        logger.info("-" * 30)
        all_passed = all(c[1] for c in checks)
        for c in checks:
            status = "[OK]" if c[1] else "[FAIL]"
            logger.info(f"  {status} {c[0]}" + (f" — {c[2]}" if len(c) > 2 else ""))
            
        return all_passed

    def run_live(self):
        """
        The main entry point for live trading.
        Initializes the dashboard, health server, and runs the continuous 
        polling loop for signal generation and trade management.
        """
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
                
                # Fetch candles securely
                h1_candles = self.data_fetcher.fetch_candles(symbol, "H1", 200)
                m15_candles = self.data_fetcher.fetch_candles(symbol, "M15", 200)
                m5_candles = self.data_fetcher.fetch_candles(symbol, "M5", 500)
                d1_candles = self.data_fetcher.fetch_candles(symbol, "D1", 50)

                # 4. Trailing Stops (Sniper Mode M5)
                if tick and len(m5_candles) > 30:
                    atr = self.strategy._calculate_atr(m5_candles, 14) 
                    self._manage_trailing_stops(symbol, tick.bid, tick.ask, atr, m5_candles[-1])
                
                if len(m5_candles) > 30:
                    current_price = m5_candles.close[-1]
                    session = "NEW_YORK" 
                    if hasattr(self.strategy, 'get_session_from_hour'):
                        session = self.strategy.get_session_from_hour(datetime.now(timezone.utc).hour)
                    
                    self.dashboard.session = session
                        
                    self.execution_pipeline.execute_cycle(
                        symbol, h1_candles, m15_candles, m5_candles, d1_candles, current_price, session
                    )
                    
                    # Update Dashboard with latest analysis context
                    analysis = self.execution_pipeline.last_analysis
                    self.dashboard.h4_trend = analysis.get("trend", "NEUTRAL")
                    self.dashboard.m30_structure = analysis.get("regime", "NEUTRAL")
                    self.dashboard.analysis_context = analysis
                
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

    def _print_backtest_summary(self, res: dict):
        console = Console()
        
        # 1. Main Metrics Table
        table = Table(title="SNIPER MODE: FINAL VALIDATION", show_header=True, header_style="bold cyan")
        table.add_column("Metric", style="dim")
        table.add_column("Value", justify="right", style="bold yellow")
        
        table.add_row("Initial Balance", f"${res.get('initial_balance', 1000.0):.2f}")
        table.add_row("Final Balance", f"${res.get('final_balance', 0.0):.2f}")
        table.add_row("Net Profit", f"[bold {'green' if res.get('net_profit', 0) >= 0 else 'red'}]${res.get('net_profit', 0.0):.2f}[/]")
        table.add_row("-", "-")
        table.add_row("Win Rate", f"{res.get('win_rate', 0.0):.1f}%")
        table.add_row("Total Trades", str(res.get('total_trades', 0)))
        
        # Calculate PF and Sharpe if not present
        df = pd.DataFrame(res.get('trades', []))
        if not df.empty:
            pos = df[df['pnl'] > 0]['pnl'].sum()
            neg = abs(df[df['pnl'] < 0]['pnl'].sum())
            pf = pos / neg if neg > 0 else 99.0
            table.add_row("Profit Factor", f"{pf:.2f}")
            
            # Drawdown
            df['cum_pnl'] = df['pnl'].cumsum()
            df['peak'] = df['cum_pnl'].cummax()
            df['dd'] = df['peak'] - df['cum_pnl']
            max_dd = df['dd'].max()
            table.add_row("Max Drawdown ($)", f"${max_dd:.2f}")
        
        console.print(Panel(table, border_style="green", title="[bold white]Institutional Analytics[/]"))
        
        # 2. Session Breakdown Table
        if not df.empty:
            s_table = Table(title="Session performance", show_header=True, header_style="bold magenta")
            s_table.add_column("Session")
            s_table.add_column("Trades", justify="right")
            s_table.add_column("Wins (TP)", justify="right", style="green")
            s_table.add_column("Losses (SL)", justify="right", style="red")
            s_table.add_column("Win Rate", justify="right")
            s_table.add_column("Net PnL", justify="right")
            
            for session in ["TOKYO", "LONDON", "LONDON/NY", "NEW_YORK"]:
                s_df = df[df['session'] == session]
                if s_df.empty: continue
                wins = len(s_df[s_df['pnl'] > 0])
                losses = len(s_df[s_df['pnl'] < 0])
                s_wr = (wins / len(s_df) * 100)
                s_pnl = s_df['pnl'].sum()
                s_table.add_row(
                    session, str(len(s_df)), str(wins), str(losses), f"{s_wr:.1f}%", 
                    f"[{'green' if s_pnl >= 0 else 'red'}]${s_pnl:.2f}[/]"
                )
            console.print(s_table)

    def run_backtest(self, symbol="XAUUSDm", start_date=None, end_date=None, count=10000):
        """
        Executes a historical simulation of the strategy.
        Fetches range-based or count-based data and runs the BacktestEngine.
        
        Args:
            symbol (str): Trading instrument.
            start_date (Optional[str]): ISO start date.
            end_date (Optional[str]): ISO end date.
            count (int): Number of candles to fetch if dates are not provided.
            
        Returns:
            dict: Backtest result summary.
        """
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
                h1 = self.data_fetcher.fetch_candles_range(symbol, "H1", dt_from, dt_to)
                m15 = self.data_fetcher.fetch_candles_range(symbol, "M15", dt_from, dt_to)
                m5 = self.data_fetcher.fetch_candles_range(symbol, "M5", dt_from, dt_to)
                d1 = self.data_fetcher.fetch_candles_range(symbol, "D1", dt_from, dt_to)
            else:
                h1 = self.data_fetcher.fetch_candles(symbol, "H1", count)
                m15 = self.data_fetcher.fetch_candles(symbol, "M15", count)
                m5 = self.data_fetcher.fetch_candles(symbol, "M5", count)
                d1 = self.data_fetcher.fetch_candles(symbol, "D1", count)
                
            # Backtest Strategy Setup
            engine = BacktestEngine(self.config, self.strategy)
            res = engine.run(symbol, h1, m15, m5, d1)
            if res:
                res['initial_balance'] = self.config.get("backtest", {}).get("initial_balance", 1000.0)
                self._print_backtest_summary(res)
            return res
        finally:
            self.connection.disconnect()

    def run_optimization(self, symbol="XAUUSDm", start_date=None, end_date=None, count=10000, mode="anchored"):
        """
        Performs Walk-Forward Optimization to find stable parameters.
        Iterates through historical windows to vet parameter consistency.
        
        Args:
            symbol (str): Trading instrument.
            start_date, end_date (Optional[str]): Date range.
            count (int): Candle count fallback.
            mode (str): 'anchored' or 'rolling' WFO.
        """
        logger.info(f"Starting WFO Optimization for {symbol} (Mode: {mode})")
        if not self.connection.connect(): return
        
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

            wfo = WalkForwardValidation(self.config, self.strategy)
            results = wfo.run_validation(symbol, h1, m15, m5, d1, mode=mode)
            
            # Print Summary
            print("\n" + "="*40)
            print(" WFO OPTIMIZATION COMPLETE ")
            print("="*40)
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
        bot.run_backtest(args.symbol, args.start_date, args.end_date, args.count)
    else:
        bot.run_live()
