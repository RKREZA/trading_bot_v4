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
from datetime import datetime, timezone, date, timedelta
from typing import Optional
import itertools
import copy
import pandas as pd
import MetaTrader5 as mt5

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
            if self.daily_trades > 0:
                # We'd ideally pass the actual trade objects here, but for now
                # we just trigger the end of day review placeholder.
                # In a full setup, you'd load yesterday's CSV and pass it.
                pass

            # 2. Reset counters
            self.daily_pnl = 0.0
            self.daily_trades = 0
            self.daily_loss = 0.0
            self.win_count = 0
            self.loss_count = 0
            self.last_reset_day = today
            self.analysis_logger.log("Daily stats reset for new day", "INFO")

            # 3. Trigger new daily pre-session AI context (async)
            self.ai_advisor.run_pre_session(symbol)

    def _update_realized_pnl(self):
        """
        Refresh daily_pnl from MT5 closed deal history for today.
        This ensures P/L reflects realized results, not just placed trades.
        """
        try:
            import MetaTrader5 as mt5
            from datetime import timezone as _tz
            today_start = datetime.now(_tz.utc).replace(hour=0, minute=0, second=0, microsecond=0)
            deals = mt5.history_deals_get(today_start, datetime.now(_tz.utc))
            if deals is None:
                return
            pnl = sum(d.profit for d in deals if d.entry == mt5.DEAL_ENTRY_OUT)
            wins = sum(1 for d in deals if d.entry == mt5.DEAL_ENTRY_OUT and d.profit > 0)
            losses = sum(1 for d in deals if d.entry == mt5.DEAL_ENTRY_OUT and d.profit <= 0)
            self.daily_pnl = pnl
            self.win_count = wins
            self.loss_count = losses

            # Notify for newly closed deals
            if self.notification_manager.enabled:
                for d in deals:
                    if d.entry == mt5.DEAL_ENTRY_OUT and d.ticket not in self.notified_deals:
                        res = "PROFIT" if d.profit > 0 else "LOSS"
                        self.notification_manager.notify_trade_close(d.symbol, "OUT", d.price, d.profit, res)
                        self.notified_deals.add(d.ticket)
        except Exception as e:
            logger.warning("Could not fetch realized P/L from MT5: %s", e)

    def _update_dashboard_state(self, signal=None, analysis=None):
        """Push latest state to the dashboard."""
        if signal:
            self.dashboard.signal = {
                "direction": signal.direction,
                "entry_price": signal.entry_price,
                "stop_loss": signal.stop_loss,
                "take_profit": signal.take_profit,
                "confidence": signal.confidence,
                "confluence_score": signal.confluence_score,
                "reasons": signal.reasons,
                "rejection_type": signal.rejection_type,
            }
        else:
            self.dashboard.signal = None
        if analysis:
            self.dashboard.h4_trend = analysis.get("h4_trend", "RANGING")
            self.dashboard.m30_structure = analysis.get("m30_structure", "NEUTRAL")
        
        # Pass AI context to dashboard
        self.dashboard.ai_context = self.ai_advisor.context
        
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

        Phase 1 (price moved 1R from entry)  : move SL to breakeven.
        Phase 2 (price moved 1.5R from entry): trail SL by 1.5× ATR.
        Phase 3 (price moved 2R from entry)  : trail SL tighter by 1.0× ATR.

        SL only ever moves in the favourable direction, never back.
        """
        try:
            import MetaTrader5 as mt5
            magic = int(self.config.get("magic_number", 234000))
            positions = mt5.positions_get(symbol=symbol)
            if not positions:
                return

            atr = self.strategy._calculate_atr(m30_candles)
            if not atr or atr <= 0:
                return

            ts_cfg = self.config.get("trailing_stop", {})
            if not ts_cfg.get("enabled", True):
                return

            be_rr    = ts_cfg.get("breakeven_at_rr",   1.0)
            p2_rr    = ts_cfg.get("trail_phase2_at_rr", 1.5)
            p3_rr    = ts_cfg.get("trail_phase3_at_rr", 2.0)
            p2_mult  = ts_cfg.get("trail_atr_multiplier",      1.5)
            p3_mult  = ts_cfg.get("trail_tight_atr_multiplier", 1.0)

            for pos in positions:
                if pos.magic != magic:
                    continue

                ticket = pos.ticket
                if ticket not in self.position_meta:
                    # Initialize metadata if missing (e.g. after restart)
                    risk = abs(pos.price_open - pos.sl) if pos.sl > 0 else 0
                    self.position_meta[ticket] = {
                        "best_price": pos.price_current,
                        "partial_closed_count": 0,
                        "risk": risk,
                        "ai_score": self.ai_advisor.context.get("last_signal_review", {}).get("score", 0.5)
                    }
                
                meta = self.position_meta[ticket]
                if "partial_closed_count" not in meta: meta["partial_closed_count"] = 0
                cur = pos.price_current
                open_price = pos.price_open

                # 1. Update MFE (Best Price)
                if pos.type == 0: # BUY
                    if cur > meta["best_price"]: meta["best_price"] = cur
                else: # SELL
                    if cur < meta["best_price"]: meta["best_price"] = cur

                # 2. Partial Profit Taking (1:1 and 2:1 RR)
                if meta["risk"] > 0 and meta["partial_closed_count"] < 2:
                    profit = (cur - open_price) if pos.type == 0 else (open_price - cur)
                    
                    pp_cfg = self.config.get("strategy", {}).get("partial_profit_config", {
                        "level1_rr": 1.0, "level1_pct": 0.25,
                        "level2_rr": 2.0, "level2_pct": 0.25
                    })
                    
                    # Step 1: 1:1 RR
                    if meta["partial_closed_count"] == 0 and profit >= meta["risk"] * pp_cfg["level1_rr"]:
                        # AI Adaptation: Take more if AI is cautious
                        l1_pct = pp_cfg["level1_pct"]
                        if meta.get("ai_score", 0.5) < 0.4: l1_pct += 0.15
                        
                        close_vol = round(pos.volume * l1_pct, 2)
                        if close_vol >= 0.01:
                            self._execute_partial_close(ticket, symbol, pos.type, close_vol, magic, "PPT Level 1 (1:1)")
                            meta["partial_closed_count"] = 1
                            # Move SL to BE
                            self._modify_sl_tp(ticket, symbol, open_price, pos.tp)
                            continue

                    # Step 2: 2:1 RR
                    elif meta["partial_closed_count"] == 1 and profit >= meta["risk"] * pp_cfg["level2_rr"]:
                        close_vol = round(pos.volume * pp_cfg["level2_pct"], 2) # Use current volume left
                        if close_vol >= 0.01:
                            self._execute_partial_close(ticket, symbol, pos.type, close_vol, magic, "PPT Level 2 (2:1)")
                            meta["partial_closed_count"] = 2
                            continue

                # 3. Optimized MFE Trailing
                # 1. Base Trail: 50% give-back
                give_back_pct = ts_cfg.get("mfe_trail_base", 0.5)
                
                # 2. Profit-Dependent (Tighten as profit increases)
                risk = meta.get("risk", 0)
                excursion = (meta["best_price"] - open_price) if pos.type == 0 else (open_price - meta["best_price"])
                
                if risk > 0:
                    rr_reached = excursion / risk
                    if rr_reached > 3.0: give_back_pct = 0.3
                    elif rr_reached > 1.5: give_back_pct = 0.4
                
                # 3. Time-Based (Tighten after many candles - not easily available here, using timestamp as proxy)
                # (Optional: fetch duration if needed)

                give_back_pct = max(0.1, min(give_back_pct, 0.9))
                
                new_sl = pos.sl
                if pos.type == 0: # BUY
                    if excursion > 0:
                        mfe_sl = meta["best_price"] - (excursion * give_back_pct)
                        if mfe_sl > new_sl: new_sl = mfe_sl
                else: # SELL
                    if excursion > 0:
                        mfe_sl = meta["best_price"] + (excursion * give_back_pct)
                        if mfe_sl < new_sl or new_sl == 0: new_sl = mfe_sl

                # Only send modification if SL actually improved
                moved = (pos.type == 0 and new_sl > pos.sl) or (pos.type == 1 and (new_sl < pos.sl or pos.sl == 0))
                if moved:
                    req = {
                        "action":   mt5.TRADE_ACTION_SLTP,
                        "symbol":   symbol,
                        "position": pos.ticket,
                        "sl":       round(float(new_sl), 2),
                        "tp":       pos.tp,
                    }
                    result = mt5.order_send(req)
                    if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                        self.analysis_logger.log(f"[MFE Trail] Ticket {pos.ticket} SL → {new_sl:.2f}", "INFO")

        except Exception as e:
            logger.warning("_manage_trailing_stops error: %s", e)

    def _execute_partial_close(self, ticket, symbol, pos_type, volume, magic, comment):
        req = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": volume,
            "type": mt5.ORDER_TYPE_SELL if pos_type == 0 else mt5.ORDER_TYPE_BUY,
            "position": ticket,
            "price": mt5.symbol_info_tick(symbol).bid if pos_type == 0 else mt5.symbol_info_tick(symbol).ask,
            "deviation": 20,
            "magic": magic,
            "comment": comment,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": self.connection.get_filling_mode(symbol),
        }
        res = mt5.order_send(req)
        if res and res.retcode == mt5.TRADE_RETCODE_DONE:
            self.analysis_logger.log(f"Partial Close executed: {comment} | Vol: {volume}", "INFO")
        else:
            logger.warning("Partial close failed for ticket %s: %s", ticket, res.comment if res else "N/A")

    def _modify_sl_tp(self, ticket, symbol, sl, tp):
        mod_req = {
            "action": mt5.TRADE_ACTION_SLTP,
            "symbol": symbol,
            "position": ticket,
            "sl": round(float(sl), 2),
            "tp": round(float(tp), 2),
        }
        mt5.order_send(mod_req)

    # ------------------------------------------------------------------
    # Real-Time Trailing Stop (background thread, tick-driven)
    # ------------------------------------------------------------------

    def _trailing_stop_thread_runner(self, symbol: str) -> None:
        """
        Runs in a dedicated background thread at tick speed.

        Design:
          - Polls MT5 for a new tick every ~100 ms.
          - When a new tick arrives it calls _manage_trailing_stops(),
            which sends TRADE_ACTION_SLTP only when the SL actually moves.
          - ATR is refreshed from M30 candles once per minute (cheap).
          - Completely independent of the 30-second strategy loop, so
            trailing reacts in real-time to price movement.
        """
        import MetaTrader5 as mt5

        ATR_REFRESH_SECS = 60          # re-fetch M30 candles every 60 s
        TICK_POLL_SECS   = 0.1         # check for new tick every 100 ms

        last_tick_time = 0
        last_atr_time  = 0.0
        cached_m30     = None

        logger.info("[TrailThread] Started for %s", symbol)

        while self.running:
            try:
                now = time.time()

                # Refresh M30 candles (and ATR) once per minute
                if cached_m30 is None or (now - last_atr_time) >= ATR_REFRESH_SECS:
                    fresh = self.data_fetcher.fetch_candles(symbol, "M30", 100)
                    if fresh:
                        cached_m30    = fresh
                        last_atr_time = now

                # Wait for a new tick
                tick = mt5.symbol_info_tick(symbol)
                if tick is None:
                    time.sleep(TICK_POLL_SECS)
                    continue

                if tick.time == last_tick_time:
                    # Same tick — no price movement, don't bother modifying
                    time.sleep(TICK_POLL_SECS)
                    continue

                last_tick_time = tick.time

                # New tick arrived — manage trailing stops immediately
                if cached_m30:
                    self._manage_trailing_stops(symbol, cached_m30)

            except Exception as e:
                logger.warning("[TrailThread] Error: %s", e)
                time.sleep(1)  # back-off on error

        logger.info("[TrailThread] Stopped for %s", symbol)

    # ------------------------------------------------------------------
    # Live Trading
    # ------------------------------------------------------------------

    def run_live(self):
        """Run the live trading loop with error handling and auto-reconnect."""
        if not self.connection.connect():
            return

        symbol = self.config.get("symbol", "BTCUSDm")
        self.dashboard.selected_symbol = symbol
        self.analysis_logger.log(f"Starting live trading for {symbol}")
        self.running = True
        self.dashboard.running = True
        self.dashboard.account_info = self.connection.account_info
        self.dashboard.start()

        # Start real-time trailing SL thread (tick-driven, independent of strategy loop)
        trail_thread = threading.Thread(
            target=self._trailing_stop_thread_runner,
            args=(symbol,),
            name="TrailingStopThread",
            daemon=True,   # dies automatically when main process exits
        )
        trail_thread.start()
        self.analysis_logger.log(f"[TrailThread] Real-time trailing SL active for {symbol}", "INFO")

        try:
            while self.running:
                start_time = time.time()
                self._reset_daily_stats()

                # Check daily limits
                max_daily_trades = self.config.get("max_daily_trades", 5)
                daily_goal = self.config.get("daily_goal", 200.0)
                if self.daily_trades >= max_daily_trades:
                    self.analysis_logger.log("Daily trade limit reached, skipping.", "WARNING")
                    time.sleep(60)
                    continue
                if self.daily_pnl >= daily_goal:
                    self.analysis_logger.log("Daily profit goal reached, stopping trades.", "WARNING")
                    time.sleep(60)
                    continue

                try:
                    # Health check with auto-reconnect
                    if not self.connection.ensure_connected():
                        self.analysis_logger.log("Connection lost — waiting to reconnect...", "ERROR")
                        time.sleep(5)
                        continue

                    # Update account info from connection
                    self.dashboard.account_info = self.connection.account_info

                    # Drawdown check
                    equity = self.connection.account_info.get('equity', 0)
                    if equity > self.peak_equity:
                        self.peak_equity = equity
                    drawdown = (self.peak_equity - equity) / self.peak_equity * 100 if self.peak_equity > 0 else 0
                    self.max_drawdown_reached = max(self.max_drawdown_reached, drawdown)
                    max_dd_allowed = self.config.get("risk", {}).get("max_drawdown_percent", 30)
                    if drawdown > max_dd_allowed:
                        self.analysis_logger.log(f"Max drawdown exceeded ({drawdown:.1f}% > {max_dd_allowed}%). Stopping trading.", "ERROR")
                        self.running = False
                        break

                    # Daily loss limit
                    max_daily_loss_pct = self.config.get("risk", {}).get("max_daily_loss_percent", 10)
                    daily_loss_limit = self.connection.account_info.get('balance', 0) * (max_daily_loss_pct / 100)
                    if self.daily_pnl < -daily_loss_limit:
                        self.analysis_logger.log(f"Daily loss limit reached ({self.daily_pnl:.2f} < -{daily_loss_limit:.2f}). Stopping trading.", "ERROR")
                        self.running = False
                        break

                    symbol_info = self.data_fetcher.get_symbol_info(symbol)
                    if not symbol_info:
                        time.sleep(1)
                        continue

                    mid_price = (symbol_info["bid"] + symbol_info["ask"]) / 2
                    self.dashboard.tick = {
                        "bid": symbol_info["bid"],
                        "ask": symbol_info["ask"],
                        "price": mid_price,
                        "spread": symbol_info["spread"] * symbol_info["point"],
                        "contract_size": symbol_info["contract_size"],
                    }

                    # Fetch candles (cached per timeframe)
                    h4_candles = self.data_fetcher.fetch_candles(symbol, "H4", 250)
                    h1_candles = self.data_fetcher.fetch_candles(symbol, "H1", 600) # Added H1 for MTF
                    m30_candles = self.data_fetcher.fetch_candles(symbol, "M30", 1540)
                    d1_candles = self.data_fetcher.fetch_candles(symbol, "D1", 100) # Added D1
                    m5_candles = self.data_fetcher.fetch_candles(symbol, "M5", 2000)

                    if h4_candles and h1_candles and m30_candles and m5_candles and d1_candles:
                        # Trailing SL is now handled by the real-time thread
                        # (no longer called here in the 30s strategy loop)

                        session = self._get_session()
                        if session != self.last_logged_session:
                            self.analysis_logger.log(f"Market Session: {session}", "INFO")
                            self.last_logged_session = session
                        signal, h4_trend = self.strategy.analyze(
                            symbol, h4_candles, h1_candles, m30_candles, m5_candles,
                            mid_price, d1_candles=d1_candles, session=session,
                        )

                        # --- AI Signal Integration ---
                        if signal:
                            # 0. News Avoidance
                            if self.ai_advisor.is_high_impact_news():
                                self.analysis_logger.log(f"News Avoidance: High-impact event near. Skipping {signal.direction} signal.", "WARNING")
                                signal = None

                        if signal:
                            # 1. Trigger AI Evaluation (Async)
                            # This will fire a background thread.
                            self.ai_advisor.evaluate_signal_async(signal, h4_trend, symbol)
                            
                            # 2. Check for latest Verdict from AI (Non-blocking)
                            ai_review = self.ai_advisor.context.get("last_signal_review", {})
                            ai_verdict = "VALID"
                            
                            # Check if verdict is fresh (within 60s) and matches signal direction
                            is_fresh = False
                            if ai_review:
                                try:
                                    updated_at = datetime.fromisoformat(ai_review.get("updated_at"))
                                    age = (datetime.now(timezone.utc) - updated_at).total_seconds()
                                    if age < 60 and ai_review.get("direction") == signal.direction:
                                        is_fresh = True
                                except Exception:
                                    pass
                            
                            if is_fresh:
                                ai_verdict = ai_review.get("verdict", "VALID")
                                if ai_verdict == "AVOID":
                                    self.analysis_logger.log(f"AI Verdict: AVOID signal for {symbol} - Skipping.", "WARNING")
                                    signal = None
                                elif ai_verdict == "CAUTION":
                                    signal.confidence *= 0.8
                                    self.analysis_logger.log(f"AI Verdict: CAUTION - Confidence reduced to {signal.confidence:.1f}%", "INFO")
                                
                                if signal:
                                    # Apply Confidence Adjustment
                                    adj = ai_review.get("confidence_adjustment", 0)
                                    signal.confidence += adj
                                    
                                    # Apply AI SL Buffer
                                    buffer_add = self.ai_advisor.sl_buffer_add
                                    if buffer_add > 0:
                                        if signal.direction == "BUY":
                                            signal.stop_loss -= buffer_add
                                        else:
                                            signal.stop_loss += buffer_add
                                        self.analysis_logger.log(f"AI SL Buffer Adjustment: {buffer_add:+.2f}", "INFO")
                                    
                                    self.analysis_logger.log(f"AI Reasoning: {ai_review.get('reasoning', '')}", "INFO")
                            else:
                                # Not fresh or not available - proceed with default (VALID)
                                self.analysis_logger.log("AI Verdict not ready - Proceeding with default VALID.", "INFO")
                            
                            # 3. Final Signal Check after AI (or default)
                            if signal:
                                if signal.confidence < self.strategy.min_confidence:
                                    self.analysis_logger.log(f"Signal confidence {signal.confidence:.1f}% below min {self.strategy.min_confidence}% after AI review.", "INFO")
                                    signal = None

                        if signal:
                            current_candle_time = m5_candles[-1]["time"]
                            last_trade = self.last_trade_time.get(symbol, 0)

                            # Prevent duplicate trades on the same candle
                            if current_candle_time > last_trade:
                                # Check open positions and pending orders
                                if self.position_manager.count_open_positions(symbol) > 0:
                                    self.analysis_logger.log(f"Skipping {signal.direction} – position already open for {symbol}")
                                    continue
                                if self.connection.get_pending_orders(symbol):
                                    self.analysis_logger.log(f"Skipping {signal.direction} – pending order exists for {symbol}")
                                    continue

                                # Calculate lot size based on risk
                                account_balance = self.connection.account_info.get("balance", 0)
                                risk_percent = self.risk_manager.calculate_scaled_risk(account_balance)
                                base_lot = self.position_manager.calculate_lot_size(symbol, signal, risk_percent, account_balance)

                                # Apply AI session risk multiplier
                                ai_multi = self.ai_advisor.lot_multiplier
                                lot_size = base_lot * ai_multi

                                # 1. Get detailed symbol limits
                                min_lot = symbol_info.get("min_lot", 0.01)
                                max_lot_broker = symbol_info.get("max_lot", 100.0)
                                lot_step = symbol_info.get("lot_step", 0.01)

                                # 2. Round to broker's volume step
                                if lot_step > 0:
                                    lot_size = round(lot_size / lot_step) * lot_step

                                # 3. Clamp to broker and config limits
                                max_lot_config = self.config.get("max_lot_size", 5.0)
                                max_lot = min(max_lot_broker, max_lot_config)
                                lot_size = max(min_lot, min(max_lot, lot_size))

                                # 4. Final safety check for precision (floating point artifacts)
                                lot_size = round(lot_size, 3) 

                                # (AI evaluation moved up to be synchronous)

                                if lot_size <= 0:
                                    self.analysis_logger.log(f"Invalid lot size {lot_size}, skipping trade", "ERROR")
                                    continue

                                self.analysis_logger.log(f"Attempting to place {signal.direction} order for {symbol} with lot {lot_size:.3f}", "INFO")

                                result = self.connection.place_order(symbol, signal, lot_size)
                                if result:
                                    self.last_trade_time[symbol] = current_candle_time
                                    self.daily_trades += 1
                                    self.dashboard.daily_trades = self.daily_trades
                                    
                                    # Notify via Telegram
                                    self.notification_manager.notify_trade_open(symbol, signal.direction, signal.entry_price, lot_size, signal.stop_loss, signal.take_profit)
                                    
                                    self._update_realized_pnl()  # refresh P/L from MT5 deal history
                                    self.analysis_logger.log(f"Trade executed: {result['ticket']}", "INFO")
                                else:
                                    self.analysis_logger.log(f"Failed to execute trade for {symbol}", "ERROR")

                except Exception as e:
                    logger.exception("Error in trading cycle: %s", e)
                    self.analysis_logger.log(f"Cycle error: {e}", "ERROR")

                self._update_realized_pnl()  # keep P/L in sync every cycle
                cycle_ms = (time.time() - start_time) * 1000
                self.dashboard.update(cycle_ms)
                time.sleep(1)

        except KeyboardInterrupt:
            self.analysis_logger.log("Stopping (Ctrl+C)...")
            self.running = False
        finally:
            self.dashboard.stop()
            self.connection.disconnect()

    # ------------------------------------------------------------------
    # Backtesting
    # ------------------------------------------------------------------

    def run_backtest(self, symbol: str, from_date: Optional[str] = None, to_date: Optional[str] = None, use_ticks: bool = False):
        """Fetch data from MT5 and run the backtest engine."""
        if not self.connection.connect():
            return

        logger.info("Fetching data for %s...", symbol)
        
        # Determine range
        d_to = datetime.now()
        if to_date:
            try:
                d_to = datetime.strptime(to_date, "%Y-%m-%d")
            except ValueError:
                logger.error("Invalid --to date format. Use YYYY-MM-DD")
                return

        if from_date:
            try:
                d_from = datetime.strptime(from_date, "%Y-%m-%d")
            except ValueError:
                logger.error("Invalid --from date format. Use YYYY-MM-DD")
                return
            
            # Use range fetch
            h4_candles = self.data_fetcher.fetch_candles_range(symbol, "H4", d_from - timedelta(days=20), d_to)
            h1_candles = self.data_fetcher.fetch_candles_range(symbol, "H1", d_from - timedelta(days=5), d_to)
            m30_candles = self.data_fetcher.fetch_candles_range(symbol, "M30", d_from - timedelta(days=2), d_to)
            d1_candles = self.data_fetcher.fetch_candles_range(symbol, "D1", d_from - timedelta(days=100), d_to)
            m5_candles = self.data_fetcher.fetch_candles_range(symbol, "M5", d_from, d_to)
        else:
            # Traditional N-candle fetch
            bt_candles = self.config.get("backtest", {}).get("candles", {"H4": 600, "M30": 4800, "M5": 9600})
            h4_candles = self.data_fetcher.fetch_candles(symbol, "H4", bt_candles.get("H4", 600))
            h1_candles = self.data_fetcher.fetch_candles(symbol, "H1", bt_candles.get("H1", 2400))
            m30_candles = self.data_fetcher.fetch_candles(symbol, "M30", bt_candles.get("M30", 4800))
            d1_candles = self.data_fetcher.fetch_candles(symbol, "D1", bt_candles.get("D1", 500))
            m5_candles = self.data_fetcher.fetch_candles(symbol, "M5", bt_candles.get("M5", 9600))

        ticks = []
        if use_ticks:
            # Only fetch ticks for the M5 range to save memory/time
            t_from = m5_candles[0]['time'] if m5_candles else d_from if from_date else 0
            t_to = m5_candles[-1]['time'] if m5_candles else d_to if from_date else 0
            
            if isinstance(t_from, int): t_from = datetime.fromtimestamp(t_from)
            if isinstance(t_to, int): t_to = datetime.fromtimestamp(t_to)
            
            ticks = self.data_fetcher.fetch_ticks_range(symbol, t_from, t_to)

        if not h1_candles or not h4_candles or not m30_candles or not m5_candles or not d1_candles:
            logger.error("Failed to fetch data for %s", symbol)
            self.connection.disconnect()
            return

        self.connection.disconnect()
        logger.info("MT5 connection closed — running backtest offline")

        engine = BacktestEngine(self.config, self.strategy)
        results = engine.run(symbol, h4_candles, h1_candles, m30_candles, m5_candles, d1_candles, ticks=ticks, quiet=True)

        # Save results to CSV
        if results.get("trades"):
            os.makedirs("backtest_results", exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"backtest_results/{symbol}_trades_{timestamp}.csv"
            df = pd.DataFrame(results["trades"])
            df.to_csv(filename, index=False)
            logger.info(f"Detailed trade history saved to: {filename}")

        # Print Detailed Results
        trades = results.get("trades", [])
        tp_count = len([t for t in trades if t.get("result") == "TP"])
        sl_count = len([t for t in trades if t.get("result") == "SL"])
        prof_sl = len([t for t in trades if t.get("result") == "SL" and t.get("pnl", 0) > 0])
        loss_sl = sl_count - prof_sl

        # Determine actual period processed
        start_date = datetime.fromtimestamp(m5_candles[0]['time']).strftime('%Y-%m-%d')
        end_date = datetime.fromtimestamp(m5_candles[-1]['time']).strftime('%Y-%m-%d')

        logger.info("-" * 50)
        logger.info(f"BACKTEST COMPLETE FOR {symbol}")
        logger.info(f"Period:          {start_date} to {end_date}")
        logger.info(f"Initial Balance: {results.get('initial_balance', 0):.2f}")
        logger.info(f"Final Balance:   {results.get('final_balance', 0):.2f}")
        logger.info(f"Net Profit:      {results.get('net_profit', 0):.2f} ({((results.get('final_balance', 0)-results.get('initial_balance', 0))/results.get('initial_balance', 1)*100):.1f}%)")
        logger.info(f"Win Rate:        {results.get('win_rate', 0):.1f}%")
        
        # Session Breakdown
        session_stats = results.get("session_stats", {})
        if session_stats:
            logger.info("Session Breakdown:")
            for s_name, stats in session_stats.items():
                count = stats["count"]
                wr = (stats["wins"] / count * 100) if count > 0 else 0
                logger.info(f"  - {s_name:<10}: {count:>3} trades | {wr:>5.1f}% WR | PnL: {stats['pnl']:>8.2f}")
        
        logger.info(f"TP Hits:         {tp_count}")
        logger.info(f"SL Hits:         {sl_count} (Profitable SL: {prof_sl}, Losing SL: {loss_sl})")
        logger.info(f"Profit Factor:   {results.get('profit_factor', 0):.2f}")
        logger.info(f"Sharpe Ratio:    {results.get('sharpe_ratio', 0):.2f}")
        logger.info(f"Max Drawdown:    {results.get('max_drawdown', 0):.1f}%")
        logger.info("-" * 50)

    # ------------------------------------------------------------------
    # Optimization
    # ------------------------------------------------------------------

    def run_optimization(self, symbol: str):
        """Grid search over strategy parameters to maximize Sharpe ratio."""
        logger.info("Starting optimization for %s...", symbol)

        if not self.connection.connect():
            return

        # Fetch data once
        bt_candles = self.config.get("backtest", {}).get("candles", {"H4": 600, "M30": 4800, "M5": 9600})
        h4_candles = self.data_fetcher.fetch_candles(symbol, "H4", bt_candles.get("H4", 600))
        h1_candles = self.data_fetcher.fetch_candles(symbol, "H1", bt_candles.get("H1", 2400))
        m30_candles = self.data_fetcher.fetch_candles(symbol, "M30", bt_candles.get("M30", 4800))
        m5_candles = self.data_fetcher.fetch_candles(symbol, "M5", bt_candles.get("M5", 9600))
        d1_candles = self.data_fetcher.fetch_candles(symbol, "D1", bt_candles.get("D1", 500))

        if not h4_candles or not h1_candles or not m30_candles or not m5_candles or not d1_candles:
            logger.error("Failed to fetch data for %s", symbol)
            self.connection.disconnect()
            return

        self.connection.disconnect()

        # Parameter ranges
        param_grid = {
            "min_confluence_score": [4, 5, 6],
            "min_confidence": [65, 75, 85],
            "sl_atr_buffer": [0.4, 0.6, 0.8],
            "pullback_distance_pct": [0.5, 0.7, 0.9],
            "atr_period": [10, 14, 20],
            "swing_lookback": [15, 20, 25],
        }

        best_sharpe = -999
        best_params = {}
        total_combos = 1
        for v in param_grid.values():
            total_combos *= len(v)

        combo = 0
        for min_conf_scr, min_conf_pct, sl_buf, pullback, atr, swing in itertools.product(
            param_grid["min_confluence_score"],
            param_grid["min_confidence"],
            param_grid["sl_atr_buffer"],
            param_grid["pullback_distance_pct"],
            param_grid["atr_period"],
            param_grid["swing_lookback"]
        ):
            combo += 1
            logger.info(f"Testing combo {combo}/{total_combos}: score={min_conf_scr}, conf={min_conf_pct}, sl_buf={sl_buf}, pullback={pullback}, atr={atr}, swing={swing}")

            # Create temporary config copy
            tmp_config = copy.deepcopy(self.config)
            tmp_config["strategy"]["min_confluence_score"] = min_conf_scr
            tmp_config["strategy"]["min_confidence"] = min_conf_pct
            tmp_config["strategy"]["sl_atr_buffer"] = sl_buf
            tmp_config["strategy"]["pullback_distance_pct"] = pullback
            tmp_config["strategy"]["atr_period"] = atr
            tmp_config["strategy"]["swing_lookback"] = swing

            tmp_strategy = StrategyEngine(tmp_config, self.analysis_logger)
            engine = BacktestEngine(tmp_config, tmp_strategy)
            results = engine.run(symbol, h4_candles, h1_candles, m30_candles, m5_candles, d1_candles, quiet=True)
            if results and results.get("sharpe_ratio", 0) > best_sharpe:
                best_sharpe = results["sharpe_ratio"]
                best_params = {
                    "min_confluence_score": min_conf_scr,
                    "min_confidence": min_conf_pct,
                    "sl_atr_buffer": sl_buf,
                    "pullback_distance_pct": pullback,
                    "atr_period": atr,
                    "swing_lookback": swing,
                }
                logger.info("New best Sharpe: %.2f with params: %s", best_sharpe, best_params)

        logger.info("=" * 50)
        logger.info("OPTIMIZATION COMPLETE")
        logger.info("Best Sharpe: %.2f", best_sharpe)
        logger.info("Best parameters: %s", best_params)
        logger.info("=" * 50)

        # Optionally update config file with best parameters
        # This is left as an exercise


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Trading Bot V3")
    parser.add_argument("--backtest", action="store_true", help="Run backtest")
    parser.add_argument("--optimize", action="store_true", help="Run parameter optimization")
    parser.add_argument("--full", action="store_true", help="Run full validation suite (backtest + stress tests)")
    parser.add_argument("--symbol", type=str, help="Symbol to trade/backtest")
    parser.add_argument("--config", type=str, default="config.json", help="Config file path")
    parser.add_argument("--from", dest="at_from", type=str, help="Backtest start date (YYYY-MM-DD)")
    parser.add_argument("--to", dest="at_to", type=str, help="Backtest end date (YYYY-MM-DD)")
    parser.add_argument("--ticks", action="store_true", help="Use tick-level data for backtest accuracy")
    args = parser.parse_args()

    # Initialize logging (console enabled for CLI modes)
    from core.logger import setup_logging
    is_cli = any([args.backtest, args.full, args.optimize])
    setup_logging(console=is_cli)

    bot = TradingBot(args.config)
    bot.config["symbol"] = args.symbol

    if args.optimize:
        # Suppress verbose strategy logs during optimization
        logging.getLogger("trading_bot.strategy").setLevel(logging.WARNING)
        bot.run_optimization(args.symbol)
    elif args.backtest:
        # Suppress verbose strategy logs during backtest
        logging.getLogger("trading_bot.strategy").setLevel(logging.WARNING)
        bot.run_backtest(args.symbol, from_date=args.at_from, to_date=args.at_to, use_ticks=args.ticks)
    else:
        bot.run_live()


if __name__ == "__main__":
    main()