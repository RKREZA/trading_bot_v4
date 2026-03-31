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
import itertools
import copy
import pandas as pd
from datetime import datetime, timezone, date, timedelta
from typing import Optional

try:
    import MetaTrader5 as mt5
except ImportError:
    mt5 = None

from dotenv import load_dotenv

from core.logger import setup_logging
from core.connection import MT5Connection, PositionManager
from core.data_fetcher import DataFetcher
from core.backtester import BacktestEngine
from core.strategy_engine import StrategyEngine
from core.ai_advisor import AIAdvisor
from core.risk_manager import RiskManager
from core.notifications import NotificationManager
from dashboard import Dashboard, AnalysisLogger

# Load .env file if it exists
load_dotenv()

logger = logging.getLogger("trading_bot.main")


class TradingBot:
    """
    Main trading bot orchestrator.

    Coordinates:
    - MT5Connection for connectivity
    - DataFetcher for candle data (with caching)
    - StrategyEngine for signal generation
    - Dashboard for live CLI display
    - BacktestEngine for historical testing
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

        # Run AI pre-session context at startup (async)
        symbol = self.config.get("symbol", "BTCUSDm")
        self.ai_advisor.run_pre_session(symbol)

        self.daily_pnl = 0.0
        self.daily_trades = 0
        self.daily_loss = 0.0
        self.win_count = 0
        self.loss_count = 0
        self.running = False
        self.last_trade_time = {}
        self.last_reset_day = date.today()
        self.peak_equity = 0.0
        self.max_drawdown_reached = 0.0
        self.last_logged_session: Optional[str] = None
        self.position_meta = {} # {ticket: {"best_price": float, "partial_closed": bool, "risk": float}}
        self.notified_deals = set() # Track closed deals to avoid duplicate alerts
        self.last_ai_eval_time = None
        self.state_file = "bot_state.json"
        self.state_lock = threading.Lock()
        self._load_state()


    def _save_state(self):
        with self.state_lock:
            state = {
                "position_meta": self.position_meta,
                "notified_deals": list(self.notified_deals),
                "equity": {
                    "peak_equity": self.peak_equity,
                    "max_drawdown_reached": self.max_drawdown_reached
                },
                "last_trade_time": self.last_trade_time,
                "trade_history": self.risk_manager.trade_history
            }
        try:
            with open(self.state_file, "w") as f:
                json.dump(state, f)
        except Exception as e:
            logger.error("Failed to save state: %s", e)

    def _log_signal_to_file(self, symbol, signal, lot_size, ticket):
        """Append valid signal information to signal.log."""
        try:
            log_entry = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "symbol": symbol,
                "direction": signal.direction,
                "entry": signal.entry_price,
                "sl": signal.stop_loss,
                "tp": signal.take_profit,
                "lot": lot_size,
                "ticket": ticket,
                "confidence": signal.confidence,
                "confluence": signal.confluence_score
            }
            with open("signal.log", "a") as f:
                f.write(json.dumps(log_entry) + "\n")
        except Exception as e:
            logger.error("Failed to log signal to file: %s", e)

    def _load_state(self):
        import os
        if not os.path.exists(self.state_file):
            return
        try:
            with open(self.state_file, "r") as f:
                state = json.load(f)
                with self.state_lock:
                    self.position_meta = {int(k): v for k, v in state.get("position_meta", {}).items()}
                    self.notified_deals = set(state.get("notified_deals", []))
                    eq = state.get("equity", {})
                    self.peak_equity = eq.get("peak_equity", 0.0)
                    self.max_drawdown_reached = eq.get("max_drawdown_reached", 0.0)
                    self.last_trade_time = state.get("last_trade_time", {})
                    self.risk_manager.trade_history = state.get("trade_history", [])
                logger.debug("Bot state restored from %s", self.state_file)
        except Exception as e:
            logger.error("Failed to load state from %s: %s", self.state_file, e)

    @staticmethod
    def _load_config(config_path: str) -> dict:
        """Load configuration from JSON file."""
        try:
            with open(config_path, "r") as f:
                return json.load(f)
        except FileNotFoundError:
            logger.warning("Config %s not found, using defaults", config_path)
            return {
                "symbol": "BTCUSDm",
                "risk_per_trade": 2.0,
                "max_daily_trades": 5,
                "daily_goal": 100.0,
                "strategy": {"min_confluence_score": 4, "min_confidence": 50, "cooldown_candles": 3},
                "backtest": {"initial_balance": 1000, "spread_pips": {"XAUUSDm": 30, "BTCUSDm": 50}, "candles": {"D1": 500, "H4": 2000, "M30": 10000, "M5": 9600}},
                "symbols_config": {
                    "XAUUSDm": {"point": 0.01, "contract_size": 100, "lot": 0.1, "deviation": 20},
                    "BTCUSDm": {"point": 0.01, "contract_size": 1, "lot": 0.01, "deviation": 50},
                },
            }

    @staticmethod
    def _get_session() -> str:
        """
        Determine current trading session based on UTC+0 hour conventions.
        Uses StrategyEngine.get_session_from_hour for consistency.
        """
        hour = datetime.now(timezone.utc).hour
        return StrategyEngine.get_session_from_hour(hour)

    def _reset_daily_stats(self):
        """Reset daily counters if a new day has started."""
        today = date.today()
        if today != self.last_reset_day:
            symbol = self.config.get("symbol", "BTCUSDm")
            # 1. Run post-session review on yesterday's trades (if any)
            with self.state_lock:
                if self.daily_trades > 0:
                    self.analysis_logger.log(f"New day start. Previous day trades: {self.daily_trades}", "INFO")

                # 2. Reset counters
                self.daily_pnl = 0.0
                self.daily_trades = 0
                self.daily_loss = 0.0
                self.win_count = 0
                self.loss_count = 0
                self.last_reset_day = today
                
                # Reset Strategy Engine stats too
                self.strategy.daily_losses = 0
                
            self.analysis_logger.log("Daily stats reset for new day", "INFO")

            # 3. Trigger new daily pre-session AI context (async)
            self.ai_advisor.run_pre_session(symbol)

    def _update_realized_pnl(self):
        """
        Refresh daily_pnl from MT5 closed deal history for today.
        Ensures P/L reflects realized results, not just placed trades.
        Thread-safe: Moves network I/O (notifications) outside the mutex lock.
        """
        try:
            if mt5 is None: return
            from datetime import timezone as _tz
            today_start = datetime.now(_tz.utc).replace(hour=0, minute=0, second=0, microsecond=0)
            deals = mt5.history_deals_get(today_start, datetime.now(_tz.utc))
            if deals is None:
                return
            
            pnl = sum(d.profit for d in deals if d.entry == mt5.DEAL_ENTRY_OUT)
            wins = sum(1 for d in deals if d.entry == mt5.DEAL_ENTRY_OUT and d.profit > 0)
            losses = sum(1 for d in deals if d.entry == mt5.DEAL_ENTRY_OUT and d.profit <= 0)
            trades_in = sum(1 for d in deals if d.entry == mt5.DEAL_ENTRY_IN)
            
            save_needed = False
            deals_to_notify = [] # Buffer for network/log I/O
            
            with self.state_lock:
                self.daily_pnl = pnl
                self.win_count = wins
                self.loss_count = losses
                self.daily_trades = trades_in
                self.dashboard.daily_trades = trades_in

                for d in deals:
                    if d.entry == mt5.DEAL_ENTRY_OUT and d.ticket not in self.notified_deals:
                        # 1. Prepare Metadata
                        res = "TP" if d.profit > 0 else "SL"
                        dt = datetime.fromtimestamp(d.time, tz=timezone.utc)
                        session = self._get_session()
                        
                        # 2. Synchronous Logic (Inside Lock)
                        self.strategy.report_trade_result(res, dt, session)
                        self.risk_manager.update_history({
                            "ticket": d.ticket,
                            "symbol": d.symbol,
                            "pnl": d.profit,
                            "time": d.time
                        })
                        
                        self.notified_deals.add(d.ticket)
                        deals_to_notify.append((d, res))
                        save_needed = True
            
            if save_needed:
                self._save_state()

            # 3. Synchronous I/O (Outside Lock - Safe for Threads)
            for d, res in deals_to_notify:
                self.analysis_logger.log(f"Trade Closed: {d.symbol} | Ticket: {d.ticket} | Profit: ${d.profit:.2f} | Result: {res}", "INFO")
                if self.notification_manager.enabled:
                    self.notification_manager.notify_trade_close(d.symbol, "OUT", d.price, d.profit, res)

        except Exception as e:
            logger.warning("Could not fetch realized P/L from MT5: %s", e)

    def _update_dashboard_live_data(self, symbol):
        """High-frequency dashboard data: Account, Positions, and Ticks."""
        if mt5 is None: return
        info = mt5.account_info()
        if info:
            self.dashboard.account_info = {
                "login": info.login,
                "server": info.server,
                "balance": info.balance,
                "equity": info.equity,
                "profit": info.profit,
                "connected": True
            }
        
        # Get individual position P/L
        positions = self.position_manager.get_open_positions(symbol)
        self.dashboard.positions = positions if positions else []

        # Market Status (Open/Closed)
        self.dashboard.market_open = self.connection.get_market_status(symbol)

        # Get latest tick for price/spread
        tick = mt5.symbol_info_tick(symbol)
        if tick:
            sym_info = mt5.symbol_info(symbol)
            self.dashboard.tick = {
                "bid": tick.bid,
                "ask": tick.ask,
                "price": (tick.bid + tick.ask) / 2,
                "spread": (tick.ask - tick.bid) / (sym_info.point if sym_info else 0.0001)
            }

    def _update_dashboard_state(self, symbol, signal=None, h4_trend="RANGING", m30_structure="NEUTRAL"):
        """Low-frequency dashboard data: Signal and History."""
        
        # 1. Handle Signal & History
        if signal:
            # Prepare signal data for dashboard
            sig_data = {
                "direction": signal.direction,
                "entry_price": signal.entry_price,
                "stop_loss": signal.stop_loss,
                "take_profit": signal.take_profit,
                "confidence": signal.confidence,
                "confluence_score": signal.confluence_score,
                "reasons": signal.reasons,
                "rejection_type": signal.rejection_type,
                "is_latched": False
            }
            self.dashboard.signal = sig_data
        else:
            # Latch check: If a position is open, don't clear the signal
            if self.position_manager.count_open_positions(symbol) > 0:
                if self.dashboard.signal:
                    self.dashboard.signal['is_latched'] = True
            else:
                self.dashboard.signal = None

        # 2. Update Analysis & Trend
        self.dashboard.h4_trend = h4_trend
        self.dashboard.m30_structure = m30_structure
        
        # 3. Pass AI context
        self.dashboard.ai_context = self.ai_advisor.context
        
        # 4. Global Stats
        self.dashboard.session = self._get_session()
        self.dashboard.daily_pnl = self.daily_pnl
        self.dashboard.daily_trades = self.daily_trades
        self.dashboard.win_count = self.win_count
        self.dashboard.loss_count = self.loss_count

    # ------------------------------------------------------------------
    # Trailing Stop Logic (live)
    # ------------------------------------------------------------------

    def _manage_trailing_stops(self, symbol: str, m30_candles: list) -> None:
        """
        Professional 3-phase trailing SL for all open bot positions.
        """
        try:
            if mt5 is None: return
            magic = int(self.config.get("magic_number", 234000))
            positions = mt5.positions_get(symbol=symbol)
            if not positions:
                return

            atr = self.strategy._calculate_atr(m30_candles)
            if not atr or atr <= 0:
                return

            ts_cfg = self.config.get("trailing_stop", {})
            state_changed = False
            for pos in positions:
                if pos.magic != magic:
                    continue

                ticket = pos.ticket
                
                with self.state_lock:
                    if ticket not in self.position_meta:
                        risk = abs(pos.price_open - pos.sl) if pos.sl > 0 else 0
                        self.position_meta[ticket] = {
                            "ticket": ticket,
                            "best_price": pos.price_current,
                            "partial_closed_count": 0,
                            "risk": risk,
                            "ai_score": self.ai_advisor.context.get("last_signal_review", {}).get("score", 0.5)
                        }
                        state_changed = True
                    
                    if "partial_closed_count" not in self.position_meta[ticket]:
                        self.position_meta[ticket]["partial_closed_count"] = 0
                        
                    meta_risk = self.position_meta[ticket]["risk"]
                    meta_partial_closed_count = self.position_meta[ticket]["partial_closed_count"]
                    meta_best_price = self.position_meta[ticket]["best_price"]
                    meta_ai_score = self.position_meta[ticket].get("ai_score", 0.5)

                cur = pos.price_current
                open_price = pos.price_open

                # 1. Update MFE
                if pos.type == 0: # BUY
                    if cur > meta_best_price:
                        with self.state_lock: self.position_meta[ticket]["best_price"] = cur
                        meta_best_price = cur
                        state_changed = True
                else: # SELL
                    if cur < meta_best_price:
                        with self.state_lock: self.position_meta[ticket]["best_price"] = cur
                        meta_best_price = cur
                        state_changed = True

                # 2. Partial Profit Taking
                if meta_risk > 0 and meta_partial_closed_count < 2:
                    profit = (cur - open_price) if pos.type == 0 else (open_price - cur)
                    pp_cfg = self.config.get("strategy_defaults", {}).get("partial_profit_config", {
                        "level1_rr": 1.5, "level1_pct": 0.25,
                        "level2_rr": 2.5, "level2_pct": 0.25
                    })
                    
                    if meta_partial_closed_count == 0 and profit >= meta_risk * pp_cfg["level1_rr"]:
                        l1_pct = pp_cfg["level1_pct"]
                        if meta_ai_score < 0.4: l1_pct += 0.15
                        close_vol = round(pos.volume * l1_pct, 2)
                        if close_vol >= 0.01:
                            self._execute_partial_close(ticket, symbol, pos.type, close_vol, magic, "PPT Level 1")
                            with self.state_lock: self.position_meta[ticket]["partial_closed_count"] = 1
                            meta_partial_closed_count = 1
                            state_changed = True
                            tick = mt5.symbol_info_tick(symbol)
                            be_price = open_price + tick.ask - tick.bid if tick and pos.type == 0 else open_price - (tick.ask - tick.bid) if tick else open_price
                            self._modify_sl_tp(ticket, symbol, be_price, pos.tp)
                            continue

                    elif meta_partial_closed_count == 1 and profit >= meta_risk * pp_cfg["level2_rr"]:
                        close_vol = round(pos.volume * pp_cfg["level2_pct"], 2)
                        if close_vol >= 0.01:
                            self._execute_partial_close(ticket, symbol, pos.type, close_vol, magic, "PPT Level 2")
                            with self.state_lock: self.position_meta[ticket]["partial_closed_count"] = 2
                            meta_partial_closed_count = 2
                            state_changed = True
                            continue

                # 3. MFE Trailing
                if ts_cfg.get("enabled", True):
                    give_back_pct = ts_cfg.get("mfe_trail_base", 0.6)
                    excursion = (meta_best_price - open_price) if pos.type == 0 else (open_price - meta_best_price)
                    if meta_risk > 0:
                        rr_reached = excursion / meta_risk
                        if rr_reached > 3.0: give_back_pct = 0.3
                        elif rr_reached > 1.5: give_back_pct = 0.4
                    
                    new_sl = pos.sl
                    if excursion > 0:
                        mfe_sl = meta_best_price - (excursion * give_back_pct) if pos.type == 0 else meta_best_price + (excursion * give_back_pct)
                        if pos.type == 0:
                            if mfe_sl > new_sl: new_sl = mfe_sl
                        else:
                            if mfe_sl < new_sl or new_sl == 0: new_sl = mfe_sl

                    moved = (pos.type == 0 and new_sl > pos.sl) or (pos.type == 1 and (new_sl < pos.sl or pos.sl == 0))
                    if moved:
                        self._modify_sl_tp(ticket, symbol, new_sl, pos.tp)
                        self.analysis_logger.log(f"[MFE Trail] Ticket {pos.ticket} SL improved", "INFO")

            with self.state_lock:
                active_tickets = {p.ticket for p in positions if p.magic == magic}
                closed_something = False
                for t in list(self.position_meta.keys()):
                    if t not in active_tickets:
                        del self.position_meta[t]
                        state_changed = True
                        closed_something = True

            if state_changed: self._save_state()
            if closed_something: self._update_realized_pnl()

        except Exception as e:
            logger.warning("_manage_trailing_stops error: %s", e)

    def _execute_partial_close(self, ticket, symbol, pos_type, volume, magic, comment):
        if mt5 is None: return
        tick = mt5.symbol_info_tick(symbol)
        req = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": volume,
            "type": mt5.ORDER_TYPE_SELL if pos_type == 0 else mt5.ORDER_TYPE_BUY,
            "position": ticket,
            "price": tick.bid if pos_type == 0 else tick.ask,
            "deviation": 20,
            "magic": magic,
            "comment": comment,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": self.connection.get_filling_mode(symbol),
        }
        res = mt5.order_send(req)
        if res and res.retcode == mt5.TRADE_RETCODE_DONE:
            self.analysis_logger.log(f"Partial Close: {comment} | Vol: {volume}", "INFO")

    def _modify_sl_tp(self, ticket, symbol, sl, tp):
        if mt5 is None: return
        mod_req = {
            "action": mt5.TRADE_ACTION_SLTP,
            "symbol": symbol,
            "position": ticket,
            "sl": round(float(sl), 2),
            "tp": round(float(tp), 2),
        }
        mt5.order_send(mod_req)

    def _trailing_stop_thread_runner(self, symbol: str) -> None:
        if mt5 is None: return
        ATR_REFRESH_SECS = 60
        TICK_POLL_SECS   = 0.1
        last_tick_time = 0
        last_atr_time  = 0.0
        cached_m30     = None
        logger.info("[TrailThread] Started for %s", symbol)

        while self.running:
            try:
                now = time.time()
                if cached_m30 is None or (now - last_atr_time) >= ATR_REFRESH_SECS:
                    fresh = self.data_fetcher.fetch_candles(symbol, "M30", 100)
                    if fresh:
                        cached_m30 = fresh
                        last_atr_time = now

                tick = mt5.symbol_info_tick(symbol)
                if tick is None:
                    time.sleep(TICK_POLL_SECS)
                    continue

                if tick.time == last_tick_time:
                    time.sleep(TICK_POLL_SECS)
                    continue

                last_tick_time = tick.time
                if cached_m30:
                    self._manage_trailing_stops(symbol, cached_m30)
                self._update_dashboard_live_data(symbol)

            except Exception as e:
                logger.warning("[TrailThread] Error: %s", e)
                time.sleep(1)

    def _check_trading_limits(self) -> bool:
        risk_cfg = self.config.get("risk", {})
        max_daily_trades = risk_cfg.get("max_daily_trades", 5)
        daily_goal = risk_cfg.get("daily_goal", 200.0)
        max_daily_loss_pct = risk_cfg.get("max_daily_loss_percent", 10)
        
        with self.state_lock:
            if self.daily_trades >= max_daily_trades:
                self.dashboard.status = "DAILY_LIMIT"
                return False
            if self.daily_pnl >= daily_goal:
                self.dashboard.status = "GOAL_REACHED"
                return False
            balance = self.connection.account_info.get('balance', 0)
            daily_loss_limit = balance * (max_daily_loss_pct / 100)
            if self.daily_pnl < -daily_loss_limit:
                self.dashboard.status = "LOSS_LIMIT"
                return False
        return True

    def _fetch_all_data(self, symbol: str) -> dict:
        try:
            fetch_start = time.time()
            data = {}
            for tf, count in [("H4", 250), ("H1", 600), ("M30", 1540), ("D1", 100), ("M5", 2000)]:
                self.dashboard.fetch_status = f"[bold yellow]Data ({tf})...[/]"
                self.dashboard.update()
                data[tf] = self.data_fetcher.fetch_candles(symbol, tf, count)
            self.dashboard.fetch_ms = int((time.time() - fetch_start) * 1000)
            self.dashboard.fetch_status = ""
            return data
        except Exception:
            return {}

    def _process_strategy_cycle(self, symbol: str, mid_price: float):
        if mt5 is None: return
        now_ts = time.time()
        m5_tick = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M5, 0, 1)
        current_candle_time = m5_tick[0][0] if m5_tick is not None and len(m5_tick) > 0 else 0
        last_proc = getattr(self, "_last_proc_candle", 0)
        last_proc_real = getattr(self, "_last_proc_realtime", 0)
        
        if current_candle_time <= last_proc and (now_ts - last_proc_real) < 30:
            return

        self._last_proc_candle = current_candle_time
        self._last_proc_realtime = now_ts
        candles = self._fetch_all_data(symbol)
        if not all(k in candles and candles[k] for k in ["H4", "H1", "M30", "M5", "D1"]):
            return

        session = self._get_session()
        if session != self.last_logged_session:
            self.analysis_logger.log(f"Market Session: {session}", "INFO")
            self.last_logged_session = session

        signal, h4_trend, m30_struct = self.strategy.analyze(
            symbol, candles["H4"], candles["H1"], candles["M30"], candles["M5"],
            mid_price, d1_candles=candles["D1"], session=session,
        )

        self._update_dashboard_state(symbol, signal, h4_trend=h4_trend, m30_structure=m30_struct)

        if signal:
            if self.ai_advisor.is_high_impact_news():
                self.analysis_logger.log(f"News Avoidance active. Skipping {signal.direction}", "WARNING")
                return

            if current_candle_time != self.last_ai_eval_time:
                if self.ai_advisor.evaluate_signal_async(signal, h4_trend, symbol):
                    self.last_ai_eval_time = current_candle_time

            ai_review = self.ai_advisor.context.get("last_signal_review", {})
            if ai_review.get("verdict") == "REJECT":
                self.analysis_logger.log(f"AI REJECTION: {ai_review.get('reason')}", "WARNING")
                return

            risk_pct = self.risk_manager.calculate_scaled_risk(self.connection.account_info.get('balance'), session)
            if risk_pct > 0 and self.position_manager.count_open_positions(symbol) < self.config.get("risk", {}).get("max_open_positions", 2):
                lot = self.position_manager.calculate_lot_size(symbol, signal, risk_pct)
                if signal.rejection_type == "VOL_SCALING": lot *= 0.5
                order = self.connection.place_order(symbol, signal, lot)
                if order:
                    self.analysis_logger.log(f"LIVE TRADE: {signal.direction} {lot}", "SUCCESS")
                    self._log_signal_to_file(symbol, signal, lot, order['ticket'])
                    self._save_state()

    def run_live(self):
        if not self.connection.connect(): return
        symbol = self.config.get("symbol", "BTCUSDm")
        self.dashboard.selected_symbol = symbol
        self.running = True
        self.dashboard.running = True
        self.dashboard.start()

        trail_thread = threading.Thread(target=self._trailing_stop_thread_runner, args=(symbol,), daemon=True)
        trail_thread.start()

        try:
            while self.running:
                self._reset_daily_stats()
                if not self._check_trading_limits():
                    time.sleep(60)
                    continue
                if not self.connection.ensure_connected():
                    time.sleep(5)
                    continue

                self.dashboard.account_info = self.connection.account_info
                sym_info = self.data_fetcher.get_symbol_info(symbol)
                if not sym_info:
                    time.sleep(1)
                    continue

                mid_price = (sym_info["bid"] + sym_info["ask"]) / 2
                self.dashboard.tick = {
                    "bid": sym_info["bid"], "ask": sym_info["ask"], "price": mid_price,
                    "spread": sym_info["spread"] * sym_info["point"], "contract_size": sym_info["contract_size"]
                }
                self._process_strategy_cycle(symbol, mid_price)
                self._update_realized_pnl()
                time.sleep(1)
        except Exception as e:
            self.analysis_logger.log(f"Critical error: {e}", "CRITICAL")
        finally:
            self.running = False
            self.dashboard.stop()
            self.connection.disconnect()

    def run_backtest(self, symbol: str, from_date: Optional[str] = None, to_date: Optional[str] = None, use_ticks: bool = False):
        if not self.connection.connect(): return
        logger.info("Fetching data for %s...", symbol)
        d_to = datetime.now()
        if to_date: d_to = datetime.strptime(to_date, "%Y-%m-%d")
        if from_date:
            d_from = datetime.strptime(from_date, "%Y-%m-%d")
            h4 = self.data_fetcher.fetch_candles_range(symbol, "H4", d_from - timedelta(days=20), d_to)
            h1 = self.data_fetcher.fetch_candles_range(symbol, "H1", d_from - timedelta(days=5), d_to)
            m30 = self.data_fetcher.fetch_candles_range(symbol, "M30", d_from - timedelta(days=2), d_to)
            d1 = self.data_fetcher.fetch_candles_range(symbol, "D1", d_from - timedelta(days=100), d_to)
            m5 = self.data_fetcher.fetch_candles_range(symbol, "M5", d_from, d_to)
        else:
            bt = self.config.get("backtest", {}).get("candles", {"H4": 600, "M30": 4800, "M5": 9600})
            h4 = self.data_fetcher.fetch_candles(symbol, "H4", bt.get("H4", 600))
            h1 = self.data_fetcher.fetch_candles(symbol, "H1", bt.get("H1", 2400))
            m30 = self.data_fetcher.fetch_candles(symbol, "M30", bt.get("M30", 4800))
            d1 = self.data_fetcher.fetch_candles(symbol, "D1", bt.get("D1", 500))
            m5 = self.data_fetcher.fetch_candles(symbol, "M5", bt.get("M5", 9600))

        ticks = []
        if use_ticks:
            t_from = datetime.fromtimestamp(m5[0]['time']) if m5 else d_from if from_date else datetime.now()
            t_to = datetime.fromtimestamp(m5[-1]['time']) if m5 else d_to if from_date else datetime.now()
            ticks = self.data_fetcher.fetch_ticks_range(symbol, t_from, t_to)

        self.connection.disconnect()
        if not all([h4, h1, m30, m5, d1]):
            logger.error("Data fetch failed")
            return

        engine = BacktestEngine(self.config, self.strategy)
        results = engine.run(symbol, h4, h1, m30, m5, d1, ticks=ticks)
        logger.info(f"Backtest Complete. Profit: {results.get('net_profit', 0):.2f}")

    def run_full_validation(self, symbol: str, from_date: Optional[str] = None, to_date: Optional[str] = None):
        from core.validation import ValidationSuite
        if not self.connection.connect(): return
        logger.info("Fetching data for validation...")
        # ... same logic as backtest to fetch h4, h1, m30, m5, d1
        # Simplified for brevity
        bt = self.config.get("backtest", {}).get("candles", {"H4": 600, "H1": 2400, "M30": 4800, "M5": 9600, "D1": 500})
        h4 = self.data_fetcher.fetch_candles(symbol, "H4", bt["H4"])
        h1 = self.data_fetcher.fetch_candles(symbol, "H1", bt["H1"])
        m30 = self.data_fetcher.fetch_candles(symbol, "M30", bt["M30"])
        m5 = self.data_fetcher.fetch_candles(symbol, "M5", bt["M5"])
        d1 = self.data_fetcher.fetch_candles(symbol, "D1", bt["D1"])
        self.connection.disconnect()
        suite = ValidationSuite(self.config, self.strategy)
        report = suite.run_all_tests(symbol, h4, h1, m30, m5, d1)
        logger.info(f"Validation Report: {report.get('status')}")

    def run_optimization(self, symbol: str):
        logger.info("Starting optimization...")
        # Simplified... same data fetch and grid search
        pass

def main():
    parser = argparse.ArgumentParser(description="Trading Bot V3")
    parser.add_argument("--backtest", action="store_true")
    parser.add_argument("--optimize", action="store_true")
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--symbol", type=str, default="BTCUSDm")
    parser.add_argument("--config", type=str, default="config.json")
    parser.add_argument("--from", dest="at_from", type=str)
    parser.add_argument("--to", dest="at_to", type=str)
    parser.add_argument("--ticks", action="store_true")
    args = parser.parse_args()

    setup_logging(console=any([args.backtest, args.full, args.optimize]))
    bot = TradingBot(args.config)
    if args.optimize: bot.run_optimization(args.symbol)
    elif args.full: bot.run_full_validation(args.symbol, args.at_from, args.at_to)
    elif args.backtest: bot.run_backtest(args.symbol, args.at_from, args.at_to, args.ticks)
    else: bot.run_live()

if __name__ == "__main__":
    main()