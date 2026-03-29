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

# Add the project root to sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from core.performance import PerformanceMetrics
from core.regime import MarketRegime
from core.ai_filter import AIFilter
from core.strategy_engine import StrategyEngine, TradeSignal
from core.risk_manager import RiskManager
from core.notifications import NotificationManager

logger = logging.getLogger("trading_bot.backtester")

class _OpenTrade:
    """Tracks the state of a single open trade during simulation."""
    __slots__ = (
        "signal", "entry_price", "lot", "sl", "tp",
        "entry_time", "regime", "ai_score", "spread", "slippage",
        "best_price", "trail_phase", "partial_closed_count", "tp_partial_1", "tp_partial_2",
        "comm_entry_paid", "comm_entry_amount", "session"
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
        self.ai_filter = AIFilter(config)
        self.ai_filter.backtest_mode = True
        self.initial_balance = config.get("backtest", {}).get("initial_balance", 1000)
        self.balance = self.initial_balance
        self.risk_manager = RiskManager(config)
        self.notification_manager = NotificationManager(config)

    def _get_spread(self, symbol: str, current_atr: float, point: float) -> float:
        symbol_cfg = self.config.get("symbol_defaults", {}).get(symbol, {})
        base_spread = symbol_cfg.get("base_spread", 25) # in points (e.g. 25 for XAUUSDm)
        # If ATR is high, widen spread. 
        # Example: if ATR > 100 points, spread = base + (ATR * 0.2)
        atr_points = current_atr / point if point > 0 else 0
        dynamic_spread = base_spread
        if atr_points > 100:
            dynamic_spread += (atr_points - 100) * 0.2
        return max(base_spread, dynamic_spread)

    def _get_commission(self, lot: float) -> float:
        # $7 per lot round-turn for XAUUSDm
        return lot * 7.0

    def _get_slippage(self, symbol: str) -> float:
        # Return random value in points (0.0 to 1.0)
        return random.uniform(0.0, 1.0)

    def _calc_lot_size(self, balance: float, entry: float, sl: float, point: float, contract_size: float, risk_pct: Optional[float] = None) -> float:
        if risk_pct is None:
            risk_pct = self.config.get("risk", {}).get("risk_per_trade_pct", 1.0)
        risk_amount = balance * (risk_pct / 100.0)
        risk_dist_price = abs(entry - sl)
        if risk_dist_price < point:
            risk_dist_price = point * 10 

        # Correct Lot Size Calculation for Gold
        point_value = contract_size * point   # dollars per point (e.g. 100 * 0.01 = $1)
        risk_points = risk_dist_price / point
        lot = risk_amount / (risk_points * point_value)
        
        lot = round(float(max(0.01, lot)), 2)
        max_lot = self.config.get("risk", {}).get("max_lot_size", 5.0)
        return min(lot, max_lot)

    @staticmethod
    def _find_slice_index(times: List[datetime], time_threshold: datetime) -> int:
        """Find index of last candle with time <= time_threshold using binary search."""
        idx = bisect.bisect_right(times, time_threshold) - 1
        return idx

    def run(self, symbol: str, h4_candles: List[dict], h1_candles: List[dict], m30_candles: List[dict], m5_candles: List[dict], d1_candles: List[dict], ticks: Optional[List[dict]] = None, quiet: bool = False):
        if quiet:
            self.strategy.silent = True
            self.risk_manager.silent = True
            self.ai_filter.silent = True
        
        symbol_cfg = self.config.get("symbol_defaults", {}).get(symbol, {})
        point = symbol_cfg.get("point", 0.01)
        contract_size = symbol_cfg.get("contract_size", 100)
        
        trades = []
        session_stats = {} # session -> {trades, wins, pnl}
        utc_offset = self.config.get("strategy_defaults", {}).get("utc_offset_hours", 0)
        open_trade: Optional[_OpenTrade] = None
        
        print(f"DEBUG: BacktestEngine.run started for {symbol}. M5 candles: {len(m5_candles)}")
        if not m5_candles:
            return {"net_profit": 0, "trades": [], "sharpe_ratio": 0, "profit_factor": 0}

        m5_closes = np.array([c['close'] for c in m5_candles])
        # print(f"DEBUG: m5_closes shape: {m5_closes.shape}")
        
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
        print("DEBUG: All times extracted.")
        
        comm_entry = 0
        tick_idx = 0
        tick_count = len(ticks) if ticks else 0
        
        with tqdm(range(200, len(m5_candles)), desc=f"Backtesting {symbol}", unit=" candle") as pbar:
            for i in pbar:
                current_candle = m5_candles[i]
                candle_time = m5_times[i]
                
                if open_trade:
                    if not hasattr(open_trade, 'comm_entry_paid') or not open_trade.comm_entry_paid:
                        comm_entry = self._get_commission(open_trade.lot) * 0.5
                        self.balance -= comm_entry
                        open_trade.comm_entry_paid = True
                        open_trade.comm_entry_amount = comm_entry
                    
                    # Fix Look-ahead Bias: use ATR from [i-1] for decisions at start of candle i
                    current_atr = m5_atr_series[i-1]
                    spread_val = self._get_spread(symbol, current_atr, point) * point
                    
                    bid_h, bid_l, bid_c = current_candle['high'], current_candle['low'], current_candle['close']
                    ask_h, ask_l, ask_c = bid_h + spread_val, bid_l + spread_val, bid_c + spread_val
                    
                    # --- Exit Evaluation (Tick-Level if available) ---
                    closed = False
                    exit_price = 0
                    result_type = ""
                    
                    # If we have ticks for this candle, check them sequentially
                    candle_ticks = []
                    if ticks:
                        next_candle_time = m5_candles[i+1]['time'] if i+1 < len(m5_candles) else float('inf')
                        while tick_idx < tick_count and ticks[tick_idx]['time'] < next_candle_time:
                            candle_ticks.append(ticks[tick_idx])
                            tick_idx += 1
                    
                    if candle_ticks:
                        for t in candle_ticks:
                            bid, ask = t['bid'], t['ask']
                            if open_trade.signal.direction == "BUY":
                                if bid <= open_trade.sl:
                                    exit_price, result_type, closed = open_trade.sl, "SL", True
                                    break
                                elif bid >= open_trade.tp:
                                    exit_price, result_type, closed = open_trade.tp, "TP", True
                                    break
                            else: # SELL
                                if ask >= open_trade.sl:
                                    exit_price, result_type, closed = open_trade.sl, "SL", True
                                    break
                                elif ask <= open_trade.tp:
                                    exit_price, result_type, closed = open_trade.tp, "TP", True
                                    break
                    else:
                        # Fallback to Candle-Level (OHLC)
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
                        # Only run if enabled in strategy_defaults
                        if self.config.get("strategy_defaults", {}).get("partial_profit_enabled", True):
                            if open_trade.signal.direction == "BUY":
                                # Level 1
                                if open_trade.partial_closed_count == 0 and bid_h >= open_trade.tp_partial_1:
                                    close_lot = open_trade.lot * 0.25
                                    pnl = (open_trade.tp_partial_1 - open_trade.entry_price) * contract_size * close_lot
                                    comm = self._get_commission(close_lot)
                                    self.balance += (pnl - comm)
                                    open_trade.lot -= close_lot
                                    
                                    # Move SL to BE only if trailing_stop is enabled
                                    if self.config.get("trailing_stop", {}).get("enabled", True):
                                        open_trade.sl = open_trade.entry_price # BE
                                        
                                    open_trade.partial_closed_count = 1
                                    if not quiet: pbar.write(f"[{candle_time}] Buy PPT1 @ {open_trade.tp_partial_1:.2f} | PnL: ${pnl:.2f} | Comm: ${comm:.2f}")
        
                                # Level 2
                                elif open_trade.partial_closed_count == 1 and bid_h >= open_trade.tp_partial_2:
                                    close_lot = open_trade.lot * 0.33 # ~25% of original (1/3 of remaining 0.75)
                                    pnl = (open_trade.tp_partial_2 - open_trade.entry_price) * contract_size * close_lot
                                    comm = self._get_commission(close_lot)
                                    self.balance += (pnl - comm)
                                    open_trade.lot -= close_lot
                                    open_trade.partial_closed_count = 2
                                    if not quiet: pbar.write(f"[{candle_time}] Buy PPT2 @ {open_trade.tp_partial_2:.2f} | PnL: ${pnl:.2f} | Comm: ${comm:.2f}")
        
                            else: # SELL
                                # Level 1
                                if open_trade.partial_closed_count == 0 and ask_l <= open_trade.tp_partial_1:
                                    close_lot = open_trade.lot * 0.25
                                    pnl = (open_trade.entry_price - open_trade.tp_partial_1) * contract_size * close_lot
                                    # PPT exit commission (remaining 50% for the closed portion)
                                    comm_ppt = self._get_commission(close_lot) * 0.5
                                    self.balance += (pnl - comm_ppt)
                                    open_trade.lot -= close_lot
                                    
                                    # Move SL to BE only if trailing_stop is enabled
                                    if self.config.get("trailing_stop", {}).get("enabled", True):
                                        open_trade.sl = open_trade.entry_price # BE
                                        
                                    open_trade.partial_closed_count = 1
                                    if not quiet: pbar.write(f"[{candle_time}] Sell PPT1 @ {open_trade.tp_partial_1:.2f} | PnL: ${pnl:.2f} | Comm_Exit: ${comm_ppt:.2f}")
        
                                # Level 2
                                elif open_trade.partial_closed_count == 1 and ask_l <= open_trade.tp_partial_2:
                                    close_lot = open_trade.lot * 0.33
                                    pnl = (open_trade.entry_price - open_trade.tp_partial_2) * contract_size * close_lot
                                    # PPT exit commission (remaining 50% for the closed portion)
                                    comm_ppt = self._get_commission(close_lot) * 0.5
                                    self.balance += (pnl - comm_ppt)
                                    open_trade.lot -= close_lot
                                    open_trade.partial_closed_count = 2
                                    if not quiet: pbar.write(f"[{candle_time}] Sell PPT2 @ {open_trade.tp_partial_2:.2f} | PnL: ${pnl:.2f} | Comm_Exit: ${comm_ppt:.2f}")
    
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
                    
                    if closed:
                        # Add trailing stop slippage penalty if not using ticks
                        if result_type == "SL" and not candle_ticks:
                            # if SL was moved (trailing), add extra slippage
                            if open_trade.sl != open_trade.signal.stop_loss:
                                ts_slippage = random.uniform(0.0, 0.3) * point
                                exit_price = exit_price - ts_slippage if open_trade.signal.direction == "BUY" else exit_price + ts_slippage
                        
                        exit_slippage = self._get_slippage(symbol) * point
                        final_exit = exit_price - exit_slippage if open_trade.signal.direction == "BUY" else exit_price + exit_slippage
                        
                        # Notify strategy with session context
                        self.strategy.report_trade_result(result_type, candle_time, session=open_trade.session)
                        
                        pnl = (final_exit - open_trade.entry_price) * (1 if open_trade.signal.direction == "BUY" else -1) * contract_size * open_trade.lot
                        # Exit commission (remaining 50% of round-turn for the current lot)
                        comm_exit = self._get_commission(open_trade.lot) * 0.5
                        
                        final_pnl = pnl - comm_exit 
                        # Total for logging is 50% of original lot (entry) + 50% of exit lot + 50% of PPT lots
                        # Simpler: Total comm paid is trackable if we wanted, but for records we just need this trade's total.
                        total_comm = comm_exit + (self._get_commission(open_trade.lot) * 0.5) # Approximate
                        
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
                        
                        if not quiet:
                            pbar.write(f"[{trade_record['exit_time']}] CLOSED {trade_record['direction']} | P&L: ${pnl:>8.2f} | Result: {result_type}")
                        open_trade = None
                        continue
                
                if i % 10 == 0:
                    postfix = {
                        "date": candle_time.strftime('%Y-%m-%d'),
                        "balance": f"${self.balance:.2f}",
                        "trades": len(trades),
                        "status": "OPEN" if open_trade else "SEARCH"
                    }
                    if open_trade:
                        postfix["sl"] = f"{open_trade.sl:.2f}"
                        postfix["tp"] = f"{open_trade.tp:.2f}"
                    pbar.set_postfix(postfix, refresh=False)
    
                if not open_trade:
                    h4_idx = self._find_slice_index(h4_times, candle_time)
                    h1_idx = self._find_slice_index(h1_times, candle_time)
                    m30_idx = self._find_slice_index(m30_times, candle_time)
                    
                    h4_slice = h4_candles[:h4_idx + 1]
                    h1_slice = h1_candles[:h1_idx + 1]
                    m30_slice = m30_candles[:m30_idx + 1]
                    m5_slice = m5_candles[:i + 1]
                    
                    # Fix Look-ahead Bias: use ATR from [i-1] for decisions at start of candle i
                    current_atr = m5_atr_series[i-1]
                    spread_val = self._get_spread(symbol, current_atr, point)
                    
                    bid = current_candle['close']
                    ask = bid + (spread_val * point)
                    
                    # Session context for overrides
                    current_hour = candle_time.hour if hasattr(candle_time, 'hour') else 0
                    current_session = self.strategy.get_session_from_hour(current_hour, utc_offset)
                    
                    signal, rejection_reason = self.strategy.analyze(symbol, h4_slice, h1_slice, m30_slice, m5_slice, bid, d1_candles=d1_candles, session=current_session)
                    
                    if signal:
                        ai_features = {
                            "direction": signal.direction,
                            "confidence": signal.confidence,
                            "atr": current_atr,
                            "regime": MarketRegime.classify(m30_slice),
                            "timestamp": candle_time
                        }
                        ai_decision, ai_score, ai_sl_buffer = self.ai_filter.filter_signal(ai_features)
                        
                        if ai_decision:
                            slippage = self._get_slippage(symbol) * point
                            
                            # Apply AI SL Buffer
                            if ai_sl_buffer > 0:
                                if signal.direction == "BUY":
                                    signal.stop_loss -= (ai_sl_buffer * current_atr)
                                else:
                                    signal.stop_loss += (ai_sl_buffer * current_atr)
    
                            entry = ask + slippage if signal.direction == "BUY" else bid - slippage
                            
                            # Dynamic Risk Scaling
                            risk_pct = self.risk_manager.calculate_scaled_risk(self.balance, session=current_session)
                            lot = self._calc_lot_size(self.balance, entry, signal.stop_loss, point, contract_size, risk_pct)
                            
                            # Volatility Scaling: 50% lot reduction if flag set
                            if signal.rejection_type == "VOL_SCALING":
                                lot *= 0.5
                                lot = max(0.01, round(lot, 2))
                                signal.reasons.append("Vol Scaling (50% Lot)")
                            
                            open_trade = _OpenTrade(
                                signal, entry, lot, candle_time, 
                                ai_features['regime'], ai_score, spread_val, slippage,
                                current_session
                            )
                            
                            # Notify via Telegram
                            self.notification_manager.notify_trade_open(symbol, signal.direction, entry, lot, signal.stop_loss, signal.take_profit)
                            if not quiet:
                                t_str = candle_time.strftime('%Y-%m-%d %H:%M') if isinstance(candle_time, datetime) else str(candle_time)
                                pbar.write(f"[{t_str}] OPENED {signal.direction} @ {entry:.5f} | Lot: {lot}")

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

        return performance

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
