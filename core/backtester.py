import os
import sys
import logging
import random
import numpy as np
import pandas as pd
import bisect
from datetime import datetime, timezone
from typing import List, Dict, Optional
from tqdm import tqdm
from tabulate import tabulate
import warnings
from tqdm import TqdmExperimentalWarning
warnings.filterwarnings("ignore", category=TqdmExperimentalWarning)

# Add the project root to sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from core.performance import PerformanceMetrics
from core.regime import MarketRegime
from core.strategy_engine import StrategyEngine, TradeSignal
from core.risk_manager import RiskManager
from core.lot_calculator import LotCalculator
from core.notifications import NotificationManager

logger = logging.getLogger("trading_bot.backtester")

class _OpenTrade:
    """Tracks the state of a single open trade during simulation."""
    __slots__ = (
        "signal", "entry_price", "lot", "sl", "tp",
        "entry_time", "regime", "ai_score", "spread", "slippage",
        "best_price", "trail_phase", "partial_closed_count", "tp_partial_1", "tp_partial_2",
        "comm_entry_paid", "comm_entry_amount", "session", "total_commission_paid"
    )

    def __init__(self, signal: TradeSignal, entry_price: float, lot: float,
                 entry_time: float, regime: str, ai_score: float, spread: float, slippage: float, session: str):
        self.signal = signal
        self.entry_price = entry_price
        self.lot = lot
        self.sl = signal.stop_loss
        self.tp = signal.take_profit
        self.entry_time = entry_time
        self.regime = regime
        self.ai_score = ai_score
        self.spread = spread
        self.slippage = slippage
        self.best_price = entry_price
        self.trail_phase = 0
        self.partial_closed_count = 0 
        self.comm_entry_paid = False
        self.session = session
        self.comm_entry_amount = 0.0
        self.total_commission_paid = 0.0
        
        # Calculate multi-level partial TPs (1:1 and 2:1 RR)
        risk = abs(entry_price - signal.stop_loss)
        if signal.direction == "BUY":
            self.tp_partial_1 = entry_price + risk
            self.tp_partial_2 = entry_price + (risk * 2)
        else:
            self.tp_partial_1 = entry_price - risk
            self.tp_partial_2 = entry_price - (risk * 2)

class BacktestEngine:
    """
    Production-ready, research-grade backtesting system.
    Optimized for realism, accuracy, and performance.
    """

    def __init__(self, config: dict, strategy: StrategyEngine):
        self.config = config
        self.strategy = strategy
        self.ai_filter = None
        self.initial_balance = config.get("backtest", {}).get("initial_balance", 1000)
        self.balance = self.initial_balance
        self.risk_manager = RiskManager(config)
        self.notification_manager = NotificationManager(config)

    def _get_spread(self, symbol: str, current_atr: float, avg_atr: float, point: float, session: str = "LONDON") -> float:
        """Session-aware, volatility-correlated spread model (returns points)."""
        base_spread = self.config.get("backtest", {}).get("spread_pips", {}).get(symbol, 25)
        # 1. Session-aware widening (Tokyo/off-hours have wider spreads)
        session_mult = {"LONDON": 1.0, "LONDON/NY": 1.2, "NEW_YORK": 1.1, "TOKYO": 1.6}.get(session, 1.3)
        # 2. Volatility scaling: higher ATR -> wider spreads
        vol_ratio = max(0.5, min(5.0, current_atr / avg_atr)) if avg_atr > 0 else 1.0
        vol_mult = 1.0 + (vol_ratio - 1.0) * 1.5
        vol_mult = max(0.8, min(vol_mult, 3.5))
        # 3. Random jitter +/-15% (real spreads fluctuate tick-to-tick)
        jitter = random.uniform(0.85, 1.15)
        return base_spread * session_mult * vol_mult * jitter

    def _get_commission(self, symbol: str, lot: float) -> float:
        """Per-symbol commission from config (round-turn per lot)."""
        comm_per_lot = self.config.get("backtest", {}).get("commission_usd", {}).get(symbol, 7.0)
        return lot * comm_per_lot

    def _get_slippage(self, symbol: str, current_atr: float, avg_atr: float, point: float) -> float:
        """Correlated slippage: scales with volatility ratio. Returns price units."""
        base_slip = self.config.get("backtest", {}).get("slippage_points", {}).get(symbol, 1.0)
        vol_ratio = max(0.5, min(5.0, current_atr / avg_atr)) if avg_atr > 0 else 1.0
        return base_slip * vol_ratio * random.uniform(0.5, 1.5) * point

    def _calc_lot_size(self, balance: float, entry: float, sl: float, point: float, contract_size: float, risk_pct: Optional[float] = None) -> float:
        if risk_pct is None:
            risk_pct = self.config.get("risk", {}).get("risk_per_trade", 1.0)
        risk_amount = balance * (risk_pct / 100.0)
        risk_dist_price = abs(entry - sl)
        
        # Use unified LotCalculator
        # In backtest mode, tick_size = point, and tick_value = contract_size * point
        tick_size = point
        tick_value = contract_size * point # Dollar value of 1 lot moving 1 point
        
        lot = LotCalculator.calculate(
            risk_amount=risk_amount,
            sl_distance=risk_dist_price,
            tick_size=tick_size,
            tick_value=tick_value,
            volume_min=0.01, # Default for backtest unless symbol_info is added
            volume_max=self.config.get("risk", {}).get("max_lot_size", 5.0),
            volume_step=0.01
        )
        
        return lot

    @staticmethod
    def _find_slice_index(times: List[datetime], time_threshold: datetime) -> int:
        """Find index of last candle with time <= time_threshold using binary search."""
        idx = bisect.bisect_right(times, time_threshold) - 1
        return idx

    def run(self, symbol: str, h4_candles: List[dict], h1_candles: List[dict], m30_candles: List[dict], 
            m5_candles: List[dict], d1_candles: List[dict], ticks: Optional[List[dict]] = None, 
            quiet: bool = False, tick_entry: bool = False):
        """
        Main simulation loop.
        :param tick_entry: If True, enters trades within the same candle a signal is generated (Realistic Breakout).
        """
        
        # --- FIX 1: Silence background modules so they don't break the progress bar ---
        self.strategy.silent = True
        self.risk_manager.silent = True
        if self.ai_filter: self.ai_filter.silent = True
        self.notification_manager.enabled = False # Prevent Telegram spam/logs during BT
        
        # Reset balance for each run (prevents cross-run contamination)
        self.balance = self.initial_balance
        
        symbol_cfg = self.config.get("symbols_config", {}).get(symbol, {})
        point = symbol_cfg.get("point", 0.01)
        contract_size = symbol_cfg.get("contract_size", 100)
        lot_step = symbol_cfg.get("lot_step", 0.01)
        min_lot = symbol_cfg.get("min_lot", 0.01)
        
        trades = []
        session_stats = {} # session -> {trades, wins, pnl}
        utc_offset = self.config.get("strategy_defaults", {}).get("utc_offset_hours", 0)
        open_trade: Optional[_OpenTrade] = None
        
        logger.debug("BacktestEngine.run started for %s. M5 candles: %d", symbol, len(m5_candles))
        if not m5_candles:
            return {"net_profit": 0, "trades": [], "sharpe_ratio": 0, "profit_factor": 0}

        m5_closes = np.array([c['close'] for c in m5_candles])
        
        m5_returns = np.zeros_like(m5_closes)
        if len(m5_closes) > 1:
            m5_returns[1:] = np.diff(np.log(m5_closes))
        
        # Performance: Pre-calculate ATR series
        m5_highs = np.array([c['high'] for c in m5_candles])
        m5_lows = np.array([c['low'] for c in m5_candles])
        m5_tr = np.zeros_like(m5_closes)
        if len(m5_closes) > 1:
            m5_tr[1:] = np.maximum(m5_highs[1:] - m5_lows[1:], 
                            np.maximum(np.abs(m5_highs[1:] - m5_closes[:-1]), 
                                       np.abs(m5_lows[1:] - m5_closes[:-1])))
        m5_atr_series = np.zeros_like(m5_closes)
        if len(m5_tr) > 14:
            m5_atr_series[14] = np.mean(m5_tr[1:15])
            for x in range(15, len(m5_tr)):
                m5_atr_series[x] = (m5_atr_series[x-1] * 13 + m5_tr[x]) / 14
        else:
            m5_atr_series[:] = 0.1
        
        h4_times = [datetime.fromtimestamp(c['time'], tz=timezone.utc) for c in h4_candles]
        h1_times = [datetime.fromtimestamp(c['time'], tz=timezone.utc) for c in h1_candles]
        m30_times = [datetime.fromtimestamp(c['time'], tz=timezone.utc) for c in m30_candles]
        m5_times = [datetime.fromtimestamp(c['time'], tz=timezone.utc) for c in m5_candles]
        logger.debug("All times extracted.")
        
        # --- Optimization: Convert ticks to structured Numpy array ---
        tick_times = None
        tick_arr = None
        if ticks:
            # Using structured arrays for memory efficiency and vectorized access
            tick_dtype = [('time', 'i8'), ('bid', 'f8'), ('ask', 'f8')]
            tick_arr = np.array([(t['time'], t['bid'], t['ask']) for t in ticks], dtype=tick_dtype)
            tick_times = tick_arr['time']
            logger.info("[Backtest] Processed %d ticks into structured Numpy array.", len(ticks))
            
        # --- Optimization: Pre-calculate all Strategy/Regime Data (Vectorized) ---
        pre_ctx = self.strategy.preprocess_history(h4_candles, h1_candles, m30_candles, m5_candles)
        m5_precomputed = pre_ctx["m5"]
        
        pending_signal = None  # Next-candle entry: stores signal for delayed execution

        
        # Use standard tqdm with custom configuration for better visibility
        with tqdm(range(200, len(m5_candles)), desc=f"BT:{symbol}", unit="c", 
                  dynamic_ncols=True, leave=True, colour="cyan") as pbar:
            for i in pbar:
                current_candle = m5_candles[i]
                candle_time = m5_times[i]
                
                # --- Gap Risk: Weekend/Holiday Detection ---
                if open_trade and i > 200:
                    time_gap = m5_candles[i]['time'] - m5_candles[i-1]['time']
                    if time_gap > 7200:  # 2+ hour gap (weekend/holiday)
                        gap_open = current_candle['open']
                        gap_atr = m5_atr_series[i-1]
                        gap_slippage = gap_atr * 0.3  # Worse fill across gaps
                        if open_trade.signal.direction == "BUY":
                            exit_price = gap_open - gap_slippage
                        else:
                            exit_price = gap_open + gap_slippage
                        pnl = (exit_price - open_trade.entry_price) * (1 if open_trade.signal.direction == "BUY" else -1) * contract_size * open_trade.lot
                        comm_exit = self._get_commission(symbol, open_trade.lot) * 0.5
                        final_pnl = pnl - comm_exit
                        self.strategy.report_trade_result("SL", candle_time, session=open_trade.session)
                        entry_dt = open_trade.entry_time if isinstance(open_trade.entry_time, datetime) else datetime.fromtimestamp(open_trade.entry_time, tz=timezone.utc)
                        exit_dt = candle_time if isinstance(candle_time, datetime) else datetime.fromtimestamp(candle_time, tz=timezone.utc)
                        trade_record = {
                            "time": entry_dt, "exit_time": exit_dt,
                            "direction": open_trade.signal.direction,
                            "entry": round(float(open_trade.entry_price), 5),
                            "exit": round(float(exit_price), 5),
                            "lot": open_trade.lot, "pnl": round(float(final_pnl), 2),
                            "commission": round(float(comm_exit), 2), "result": "GAP",
                            "regime": open_trade.regime,
                            "ai_score": round(float(open_trade.ai_score), 4),
                            "spread": 0, "slippage": round(float(gap_slippage / point), 2),
                            "session": open_trade.session
                        }
                        trades.append(trade_record)
                        s_name = open_trade.session
                        if s_name not in session_stats:
                            session_stats[s_name] = {"count": 0, "wins": 0, "pnl": 0.0}
                        session_stats[s_name]["count"] += 1
                        if final_pnl > 0: session_stats[s_name]["wins"] += 1
                        session_stats[s_name]["pnl"] += final_pnl
                        self.balance += final_pnl
                        self.risk_manager.update_history(trade_record)
                        # pbar.write(f"[{exit_dt}] GAP CLOSE {open_trade.signal.direction} | P&L: ${final_pnl:>8.2f}")
                        open_trade = None
                        pending_signal = None
                        continue
                
                # --- Execute Pending Signal at this candle's OPEN (next-candle mode) ---
                if not open_trade and pending_signal is not None and not tick_entry:
                    ps = pending_signal
                    pending_signal = None
                    
                    open_trade = self._handle_entry(
                        symbol, ps["signal"], current_candle, candle_time,
                        point, contract_size, ps["session"], ps["regime"],
                        ps["ai_score"], m5_avg_atr[i-1], m5_atr_series[i-1]
                    )
                    self.notification_manager.notify_trade_open(symbol, open_trade.signal.direction, open_trade.entry_price, open_trade.lot, open_trade.sl, open_trade.tp)

                if open_trade:
                    if not hasattr(open_trade, 'comm_entry_paid') or not open_trade.comm_entry_paid:
                        comm_entry = self._get_commission(symbol, open_trade.lot) * 0.5
                        self.balance -= comm_entry
                        open_trade.comm_entry_paid = True
                        open_trade.comm_entry_amount = comm_entry
                        open_trade.total_commission_paid += comm_entry
                    
                    # Fix Look-ahead Bias: use ATR from [i-1] for decisions at start of candle i
                    current_atr = m5_atr_series[i-1]
                    avg_atr_here = m5_avg_atr[i-1]
                    spread_val = self._get_spread(symbol, current_atr, avg_atr_here, point, open_trade.session) * point
                    
                    bid_h, bid_l, bid_c = current_candle['high'], current_candle['low'], current_candle['close']
                    ask_h, ask_l, ask_c = bid_h + spread_val, bid_l + spread_val, bid_c + spread_val
                    
                    # --- Exit Evaluation (Tick-Level if available) ---
                    closed = False
                    exit_price = 0
                    result_type = ""
                    
                    # Optimized Tick-Level Slicing
                    candle_ticks = None
                    if tick_arr is not None:
                        start_time = current_candle['time']
                        next_candle_time = m5_candles[i+1]['time'] if i+1 < len(m5_candles) else float('inf')
                        
                        start_idx = np.searchsorted(tick_times, start_time, side='left')
                        end_idx = np.searchsorted(tick_times, next_candle_time, side='left')
                        
                        if start_idx < end_idx:
                            candle_ticks = tick_arr[start_idx:end_idx]
                    
                    if candle_ticks is not None:
                        # Vectorized Exit Check
                        bids = candle_ticks['bid']
                        asks = candle_ticks['ask']
                        
                        if open_trade.signal.direction == "BUY":
                            sl_hits = bids <= open_trade.sl
                            tp_hits = bids >= open_trade.tp
                            if np.any(sl_hits | tp_hits):
                                # Find first event
                                idx_sl = np.where(sl_hits)[0][0] if np.any(sl_hits) else len(bids)
                                idx_tp = np.where(tp_hits)[0][0] if np.any(tp_hits) else len(bids)
                                if idx_sl < idx_tp:
                                    exit_price, result_type, closed = open_trade.sl, "SL", True
                                else:
                                    exit_price, result_type, closed = open_trade.tp, "TP", True
                        else: # SELL
                            sl_hits = asks >= open_trade.sl
                            tp_hits = asks <= open_trade.tp
                            if np.any(sl_hits | tp_hits):
                                idx_sl = np.where(sl_hits)[0][0] if np.any(sl_hits) else len(asks)
                                idx_tp = np.where(tp_hits)[0][0] if np.any(tp_hits) else len(asks)
                                if idx_sl < idx_tp:
                                    exit_price, result_type, closed = open_trade.sl, "SL", True
                                else:
                                    exit_price, result_type, closed = open_trade.tp, "TP", True
                    else:
                        # Fallback to Candle-Level (OHLC) with spread adjustment
                        bid_h, bid_l = current_candle['high'], current_candle['low']
                        ask_h, ask_l = bid_h + spread_val, bid_l + spread_val
                        
                        if open_trade.signal.direction == "BUY":
                            if bid_l <= open_trade.sl:
                                exit_price, result_type, closed = open_trade.sl, "SL", True
                            elif bid_h >= open_trade.tp:
                                exit_price, result_type, closed = open_trade.tp, "TP", True
                        else:
                            if ask_h >= open_trade.sl:
                                exit_price, result_type, closed = open_trade.sl, "SL", True
                            elif ask_l <= open_trade.tp:
                                exit_price, result_type, closed = open_trade.tp, "TP", True
                    
                        # --- Advanced Partial Profit Taking (PPT) ---
                        # Only run if enabled in strategy_defaults and trade is not closed
                        if not closed and self.config.get("strategy_defaults", {}).get("partial_profit_enabled", True):
                            ppt_cfg = self.config.get("strategy_defaults", {}).get("partial_profit_config", {"level1_pct": 0.25, "level2_pct": 0.33})
                            level1_pct = ppt_cfg.get("level1_pct", 0.25)
                            level2_pct = ppt_cfg.get("level2_pct", 0.33)
                            
                            if open_trade.signal.direction == "BUY":
                                # Level 1
                                if open_trade.partial_closed_count == 0 and bid_h >= open_trade.tp_partial_1:
                                    success, pnl, comm = self._handle_partial_tp(open_trade, level1_pct, open_trade.tp_partial_1, symbol, contract_size, lot_step, min_lot)
                                    open_trade.partial_closed_count = 1
        
                                # Level 2
                                elif open_trade.partial_closed_count == 1 and bid_h >= open_trade.tp_partial_2:
                                    success, pnl, comm = self._handle_partial_tp(open_trade, level2_pct, open_trade.tp_partial_2, symbol, contract_size, lot_step, min_lot)
                                    open_trade.partial_closed_count = 2
        
                            else: # SELL
                                # Level 1
                                if open_trade.partial_closed_count == 0 and ask_l <= open_trade.tp_partial_1:
                                    success, pnl, comm = self._handle_partial_tp(open_trade, level1_pct, open_trade.tp_partial_1, symbol, contract_size, lot_step, min_lot)
                                    open_trade.partial_closed_count = 1
        
                                # Level 2
                                elif open_trade.partial_closed_count == 1 and ask_l <= open_trade.tp_partial_2:
                                    success, pnl, comm = self._handle_partial_tp(open_trade, level2_pct, open_trade.tp_partial_2, symbol, contract_size, lot_step, min_lot)
                                    open_trade.partial_closed_count = 2
    
                        if not closed:
                            # Save SL before MFE/trail updates for intra-candle bias check
                            sl_before_trail = open_trade.sl
                        
                            # MFE Tracking
                            if open_trade.signal.direction == "BUY":
                                if bid_h > open_trade.best_price:
                                    open_trade.best_price = bid_h
                            else:
                                if ask_l < open_trade.best_price:
                                    open_trade.best_price = ask_l
    
                            # --- Optimized MFE Trailing ---
                            ts_cfg = self.config.get("trailing_stop", {})
                            if ts_cfg.get("enabled", True):
                                # 1. Base Trail: 50% give-back
                                give_back_pct = ts_cfg.get("mfe_trail_base", 0.5)
                            
                                # 2. Volatility Adjustment
                                # If current ATR is much higher than average, widen trail (give more back)
                                if i > 20:
                                    prev_atr_slice = m5_atr_series[i-21:i-1]
                                    if len(prev_atr_slice) > 0:
                                        avg_atr = np.mean(prev_atr_slice)
                                        if avg_atr > 0:
                                            atr_ratio = m5_atr_series[i-1] / avg_atr
                                            if atr_ratio > 1.3: give_back_pct += 0.2
                                            elif atr_ratio < 0.7: give_back_pct -= 0.1
                            
                                # 3. Profit-Dependent (Tighten as profit increases)
                                excursion = (open_trade.best_price - open_trade.entry_price) if open_trade.signal.direction == "BUY" else (open_trade.entry_price - open_trade.best_price)
                                risk = abs(open_trade.entry_price - open_trade.signal.stop_loss)
                                if risk > 0:
                                    rr_reached = excursion / risk
                                    if rr_reached >= 3.0:
                                        give_back_pct = 0.3 # Tighten to 30% give-back
                                    elif rr_reached >= 2.0:
                                        give_back_pct = 0.4
                            
                                # 4. Time-Based (Tighten after many candles)
                                candles_in_trade = i - bisect.bisect_left(m5_times, open_trade.entry_time)
                                if candles_in_trade > 100: # ~8 hours on M5
                                    give_back_pct = min(give_back_pct, 0.4)
        
                                give_back_pct = max(0.1, min(give_back_pct, 0.9))
                            
                                # 5. Apply Trail
                                if open_trade.signal.direction == "BUY":
                                    new_sl = open_trade.best_price - (excursion * give_back_pct)
                                    if new_sl > open_trade.sl:
                                        open_trade.sl = new_sl
                                else:
                                    new_sl = open_trade.best_price + (excursion * give_back_pct)
                                    if new_sl < open_trade.sl:
                                        open_trade.sl = new_sl
                            
                                # 6. CRITICAL: Intra-candle look-ahead bias protection
                                # The candle's HIGH updated MFE and moved SL tighter, but the LOW
                                # might have come FIRST. If the candle's adverse extremum violated
                                # the ORIGINAL SL (before trail), we must assume worst case: SL hit first.
                                if not closed:
                                    if open_trade.signal.direction == "BUY" and bid_l <= sl_before_trail and open_trade.sl != sl_before_trail:
                                        # Candle low hit original SL AND trail moved SL — ambiguous, assume SL hit
                                        exit_price = sl_before_trail
                                        result_type = "SL"
                                        closed = True
                                        open_trade.sl = sl_before_trail  # Restore for accurate record
                                    elif open_trade.signal.direction == "SELL" and ask_h >= sl_before_trail and open_trade.sl != sl_before_trail:
                                        exit_price = sl_before_trail
                                        result_type = "SL"
                                        closed = True
                                        open_trade.sl = sl_before_trail
                                    # Also check if candle violated the NEW tighter SL
                                    elif open_trade.signal.direction == "BUY" and bid_l <= open_trade.sl:
                                        exit_price = open_trade.sl
                                        result_type = "SL"
                                        closed = True
                                    elif open_trade.signal.direction == "SELL" and ask_h >= open_trade.sl:
                                        exit_price = open_trade.sl
                                        result_type = "SL"
                                        closed = True
                    
                    if closed:
                        # Add trailing stop slippage penalty if not using ticks
                        if result_type == "SL" and not candle_ticks:
                            # if SL was moved (trailing), add extra slippage
                            if open_trade.sl != open_trade.signal.stop_loss:
                                ts_slippage = random.uniform(0.0, 0.3) * point
                                exit_price = exit_price - ts_slippage if open_trade.signal.direction == "BUY" else exit_price + ts_slippage
                        
                        exit_slippage = self._get_slippage(symbol, current_atr, avg_atr_here, point)
                        final_exit = exit_price - exit_slippage if open_trade.signal.direction == "BUY" else exit_price + exit_slippage
                        
                        # Notify strategy with session context
                        self.strategy.report_trade_result(result_type, candle_time, session=open_trade.session)
                        
                        pnl = (final_exit - open_trade.entry_price) * (1 if open_trade.signal.direction == "BUY" else -1) * contract_size * open_trade.lot
                        # Exit commission (remaining 50% of round-turn for the current lot)
                        comm_exit = self._get_commission(symbol, open_trade.lot) * 0.5
                        open_trade.total_commission_paid += comm_exit
                        
                        final_pnl = pnl - comm_exit 
                        total_comm = round(open_trade.total_commission_paid, 2)
                        
                        # Ensure datetime objects for pandas/metrics
                        entry_dt = open_trade.entry_time if isinstance(open_trade.entry_time, datetime) else datetime.fromtimestamp(open_trade.entry_time, tz=timezone.utc)
                        exit_dt = candle_time if isinstance(candle_time, datetime) else datetime.fromtimestamp(candle_time, tz=timezone.utc)
    
                        trade_record = {
                            "time": entry_dt,
                            "exit_time": exit_dt,
                            "direction": open_trade.signal.direction,
                            "entry": round(float(open_trade.entry_price), 5),
                            "exit": round(float(final_exit), 5),
                            "lot": open_trade.lot,
                            "pnl": round(float(final_pnl), 2),
                            "commission": round(float(total_comm), 2),
                            "result": result_type,
                            "regime": open_trade.regime,
                            "ai_score": round(float(open_trade.ai_score), 4),
                            "spread": round(float(spread_val / point), 2),
                            "slippage": round(float(open_trade.slippage / point), 2),
                            "session": open_trade.session
                        }
                        trades.append(trade_record)
                        
                        # Update session stats
                        s_name = open_trade.session
                        if s_name not in session_stats:
                            session_stats[s_name] = {"count": 0, "wins": 0, "pnl": 0.0}
                        session_stats[s_name]["count"] += 1
                        if final_pnl > 0:
                            session_stats[s_name]["wins"] += 1
                        session_stats[s_name]["pnl"] += final_pnl

                        self.balance += final_pnl
                        
                        # Update Risk Manager History for Kelly
                        self.risk_manager.update_history(trade_record)
                        
                        # Notify via Telegram
                        self.notification_manager.notify_trade_close(symbol, trade_record['direction'], trade_record['exit'], final_pnl, result_type)
                        
                        # if not quiet:
                        #     pbar.write(f"[{trade_record['exit_time']}] CLOSED {trade_record['direction']} | P&L: ${pnl:>8.2f} | Result: {result_type}")
                        open_trade = None
                        continue
                
                if i % 25 == 0:
                    sl_count = sum(1 for t in trades if t['result'] == 'SL')
                    tp_count = sum(1 for t in trades if t['result'] == 'TP')
                    postfix = {
                        "date": candle_time.strftime('%d-%m-%y'),
                        "balance": f"${self.balance:.0f}",
                        "trade": f"{len(trades)} [sl: {sl_count}, tp: {tp_count}]",
                        "status": "OPEN" if open_trade else "IDLE"
                    }
                    pbar.set_postfix(postfix, refresh=False)
    
                # --- Signal Generation (store as pending for next-candle entry) ---
                if not open_trade and pending_signal is None:
                    # High-Performance Signal Check
                    pre_at_i = m5_precomputed[i]
                    signal, h4_trend, regime_str = self.strategy.analyze(
                        symbol, h4_slice, h1_slice, m30_slice, m5_slice, bid, 
                        d1_candles=d1_candles, session=current_session,
                        preprocessed=pre_at_i
                    )
                    
                    if signal:
                        ai_features = {
                            "direction": signal.direction,
                            "confidence": signal.confidence,
                            "atr": current_atr,
                            "regime": regime_str,
                            "timestamp": candle_time
                        }
                        ai_decision, ai_score, ai_sl_buffer = self.ai_filter.filter_signal(ai_features)
                        
                        if ai_decision:
                            # Apply AI SL Buffer at signal time
                            if ai_sl_buffer > 0:
                                if signal.direction == "BUY":
                                    signal.stop_loss -= (ai_sl_buffer * current_atr)
                                else:
                                    signal.stop_loss += (ai_sl_buffer * current_atr)
                            
                            # Execute
                            if tick_entry and not open_trade:
                                # Breakout mode: Entry at the breakout level within current candle
                                entry_price = signal.entry_price
                                if candle_ticks is not None:
                                    # Find first tick crossing the level
                                    if signal.direction == "BUY":
                                        matches = np.where(candle_ticks['ask'] >= entry_price)[0]
                                    else:
                                        matches = np.where(candle_ticks['bid'] <= entry_price)[0]
                                    
                                    if len(matches) > 0:
                                        entry_price = candle_ticks[matches[0]]['ask' if signal.direction == "BUY" else 'bid']
                                
                                open_trade = self._handle_entry(
                                    symbol, signal, current_candle, candle_time,
                                    point, contract_size, current_session, ai_features["regime"],
                                    ai_score, avg_atr_here, current_atr, force_price=entry_price
                                )
                                self.notification_manager.notify_trade_open(symbol, signal.direction, open_trade.entry_price, open_trade.lot, signal.stop_loss, signal.take_profit)
                            else:
                                # Next-candle mode: Store as pending
                                pending_signal = {
                                    "signal": signal,
                                    "ai_score": ai_score,
                                    "regime": ai_features["regime"],
                                    "session": current_session,
                                }

        performance = PerformanceMetrics.calculate_metrics(trades, self.initial_balance)
        performance['trades'] = trades
        performance['session_stats'] = session_stats

        # --- Post-Backtest Sanity Checks ---
        if trades:
            mc_drawdown = self._run_monte_carlo_drawdown(trades)
            performance['mc_max_drawdown'] = mc_drawdown
            if mc_drawdown > 30:
                logger.warning(f"CAUTION: Monte Carlo Max Drawdown ({mc_drawdown:.1f}%) exceeds 30%!")

            # OOS Consistency (if data is split)
            oos_score = self._check_oos_consistency(trades)
            if oos_score is not None:
                performance['oos_consistency_score'] = oos_score
                if oos_score < 0.5:
                    logger.warning(f"CAUTION: Out-of-Sample consistency score low ({oos_score:.2f}). Possible over-fitting.")

        # --- Export to CSV ---
        if trades:
            os.makedirs("backtest_results", exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            csv_filename = f"backtest_results/{symbol}_trades_{timestamp}.csv"
            pd.DataFrame(trades).to_csv(csv_filename, index=False)
            print(f"\n[SUCCESS] Trade history exported to: {csv_filename}")

        # --- Terminal Summary Table ---
        sl_hits = sum(1 for t in trades if t['result'] == 'SL')
        tp_hits = sum(1 for t in trades if t['result'] == 'TP')
        
        # Calculate trailing stop hits (SL where pnl > 0 or sl move detected)
        # Note: In our backtester 'SL' result is used for any stop hit.
        trailing_sl_hits = sum(1 for t in trades if t['result'] == 'SL' and t['pnl'] > 0)
        hard_sl_hits = sl_hits - trailing_sl_hits

        summary_data = [
            ["Metric", "Value"],
            ["Initial Balance", f"${performance.get('initial_balance', 0):,.2f}"],
            ["Final Balance", f"${performance.get('final_balance', 0):,.2f}"],
            ["Net Profit", f"${performance.get('net_profit', 0):,.2f}"],
            ["Win Rate", f"{performance.get('win_rate', 0):.2f}%"],
            ["Total Trades", performance.get('total_trades', 0)],
            ["TP Hits", f"{tp_hits} (Full Target)"],
            ["SL Hits (Hard)", f"{hard_sl_hits} (Initial SL)"],
            ["SL Hits (Trail)", f"{trailing_sl_hits} (Profit Protected)"],
            ["Max Drawdown", f"{performance.get('max_drawdown', 0):.2f}%"],
            ["Profit Factor", f"{performance.get('profit_factor', 0):.2f}"],
            ["Sharpe Ratio", f"{performance.get('sharpe_ratio', 0):.2f}"],
            ["MC Max DD", f"{performance.get('mc_max_drawdown', 0):.2f}%"]
        ]
        
        print("\n" + tabulate(summary_data, headers="firstrow", tablefmt="fancy_grid"))

        return performance

    def _handle_entry(self, symbol: str, signal: TradeSignal, current_candle: dict, candle_time: datetime,
                      point: float, contract_size: float, session: str, regime: str, ai_score: float,
                      avg_atr_here: float, current_atr: float, force_price: Optional[float] = None) -> _OpenTrade:
        """Centralized helper for trade execution (Slippage, Spread, Lot calculation)."""
        spread_val = self._get_spread(symbol, current_atr, avg_atr_here, point, session)
        slippage = self._get_slippage(symbol, current_atr, avg_atr_here, point)
        
        # Entry Price: Either a forced tick/breakout level or the candle OHLC
        ref_price = force_price if force_price is not None else current_candle['open']
        
        if signal.direction == "BUY":
            entry = ref_price + (spread_val * point) + slippage
        else:
            entry = ref_price - slippage
            
        risk_pct = self.risk_manager.calculate_scaled_risk(self.balance, session=session)
        lot = self._calc_lot_size(self.balance, entry, signal.stop_loss, point, contract_size, risk_pct)
        
        if signal.rejection_type == "VOL_SCALING":
            lot = max(0.01, round(lot * 0.5, 2))
            
        trade = _OpenTrade(
            signal, entry, lot, candle_time,
            regime, ai_score, spread_val, slippage, session
        )
        
        # Initial commission
        comm_entry = self._get_commission(symbol, lot) * 0.5
        self.balance -= comm_entry
        trade.comm_entry_paid = True
        trade.comm_entry_amount = comm_entry
        trade.total_commission_paid = comm_entry
        
        return trade

    def _handle_partial_tp(self, trade: _OpenTrade, close_factor: float, tp_price: float, 
                          symbol: str, contract_size: float, lot_step: float, min_lot: float):
        """Helper to process partial profit taking."""
        raw_close_lot = trade.lot * close_factor
        close_lot = round(raw_close_lot / lot_step) * lot_step
        close_lot = round(close_lot, 2)
        
        if close_lot < min_lot:
            return False, 0.0, 0.0 # Skip PPT
        elif round(trade.lot - close_lot, 2) < min_lot:
            close_lot = round(trade.lot - min_lot, 2)
            
        if close_lot < min_lot:
            return False, 0.0, 0.0 # Skip
            
        pnl = (tp_price - trade.entry_price) if trade.signal.direction == "BUY" else (trade.entry_price - tp_price)
        pnl = pnl * contract_size * close_lot
        comm = self._get_commission(symbol, close_lot) * 0.5
        trade.total_commission_paid += comm
        self.balance += (pnl - comm)
        trade.lot = round(trade.lot - close_lot, 2)
        if self.config.get("trailing_stop", {}).get("enabled", True):
            trade.sl = trade.entry_price
        return True, pnl, comm

    def _run_monte_carlo_drawdown(self, trades: List[Dict], iterations: int = 100) -> float:
        """Randomly shuffle trade order 100 times and find the worst-case max drawdown."""
        pnl_list = [t['pnl'] for t in trades]
        max_drawdowns = []

        for _ in range(iterations):
            shuffled = pnl_list[:]
            random.shuffle(shuffled)
            
            balance = self.initial_balance
            equity_curve = [balance]
            for pnl in shuffled:
                balance += pnl
                equity_curve.append(balance)
            
            equity_series = pd.Series(equity_curve)
            rolling_max = equity_series.cummax()
            drawdown = (rolling_max - equity_series) / rolling_max * 100
            max_drawdowns.append(drawdown.max())
            
        return float(np.mean(max_drawdowns))

    def _check_oos_consistency(self, trades: List[Dict]) -> Optional[float]:
        """
        Compare Sharpe Ratio of first 70% (In-Sample) vs last 30% (Out-of-Sample).
        Returns the ratio of OOS Sharpe / IS Sharpe.
        """
        if len(trades) < 20:
            return None
        
        split_idx = int(len(trades) * 0.7)
        is_trades = trades[:split_idx]
        oos_trades = trades[split_idx:]
        
        is_perf = PerformanceMetrics.calculate_metrics(is_trades, self.initial_balance)
        # For OOS, use the ending balance of IS as initial
        oos_perf = PerformanceMetrics.calculate_metrics(oos_trades, is_perf['final_balance'])
        
        is_sharpe = is_perf.get('sharpe_ratio', 0)
        oos_sharpe = oos_perf.get('sharpe_ratio', 0)
        
        if is_sharpe > 0:
            return float(oos_sharpe / is_sharpe)
        return 0.0
