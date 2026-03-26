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

logger = logging.getLogger("trading_bot.backtester")

class _OpenTrade:
    """Tracks the state of a single open trade during simulation."""
    __slots__ = (
        "signal", "entry_price", "lot", "sl", "tp",
        "entry_time", "regime", "ai_score", "spread", "slippage",
        "best_price", "trail_phase", "partial_closed", "tp_partial"
    )

    def __init__(self, signal: TradeSignal, entry_price: float, lot: float,
                 entry_time: int, regime: str, ai_score: float, spread: float, slippage: float):
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
        self.partial_closed = False
        
        # Calculate partial TP (1:1 RR)
        risk = abs(entry_price - signal.stop_loss)
        if signal.direction == "BUY":
            self.tp_partial = entry_price + risk
        else:
            self.tp_partial = entry_price - risk

class BacktestEngine:
    """
    Production-ready, research-grade backtesting system.
    Optimized for realism, accuracy, and performance.
    """

    def __init__(self, config: dict, strategy: StrategyEngine):
        self.config = config
        self.strategy = strategy
        self.ai_filter = AIFilter(threshold=config.get("ai_filter", {}).get("threshold", 0.75))
        self.initial_balance = config.get("backtest", {}).get("initial_balance", 10000)
        self.balance = self.initial_balance

    def _get_spread(self, symbol: str, current_volatility: float) -> float:
        symbol_cfg = self.config.get("symbol_defaults", {}).get(symbol, {})
        base_spread = symbol_cfg.get("base_spread", 0.5)
        vol_mult = self.config.get("execution", {}).get("volatility_multiplier", 1.5)
        # Avoid negative spread, cap at reasonable level
        return max(base_spread, base_spread * (1 + current_volatility * vol_mult))

    def _get_slippage(self, symbol: str) -> float:
        max_slip = self.config.get("symbols_config", {}).get(symbol, {}).get("max_slippage", 0.2)
        return random.uniform(0, max_slip)

    def _calc_lot_size(self, balance: float, entry: float, sl: float, point: float, contract_size: float) -> float:
        risk_pct = self.config.get("risk", {}).get("risk_per_trade_pct", 1.0)
        risk_amount = balance * (risk_pct / 100.0)
        
        # Avoid division by zero
        risk_dist_price = abs(entry - sl)
        if risk_dist_price < point:
            risk_dist_price = point * 10 # Default to 10 points if too tight

        lot = risk_amount / (risk_dist_price * contract_size)
        
        # Round to 0.01 and clamp to [0.01, max_lot]
        lot = round(max(0.01, lot), 2)
        max_lot = self.config.get("risk", {}).get("max_lot_size", 5.0)
        return min(lot, max_lot)

    @staticmethod
    def _find_slice_index(times: List[int], time_threshold: int) -> int:
        """Find index of last candle with time <= time_threshold using binary search."""
        idx = bisect.bisect_right(times, time_threshold) - 1
        return idx

    def run(self, symbol: str, h4_candles: List[dict], m30_candles: List[dict], m5_candles: List[dict], d1_candles: List[dict], quiet: bool = False):
        if quiet:
            self.strategy.silent = True
        
        symbol_cfg = self.config.get("symbol_defaults", {}).get(symbol, {})
        point = symbol_cfg.get("point", 0.01)
        contract_size = symbol_cfg.get("contract_size", 100)
        
        trades = []
        open_trade: Optional[_OpenTrade] = None
        
        # Pre-calculate log returns and time lists for efficiency
        m5_closes = np.array([c['close'] for c in m5_candles])
        m5_returns = np.zeros_like(m5_closes)
        m5_returns[1:] = np.diff(np.log(m5_closes))
        
        h4_times = [c['time'] for c in h4_candles]
        m30_times = [c['time'] for c in m30_candles]
        
        pbar = tqdm(range(100, len(m5_candles)), desc=f"Backtesting {symbol}", unit=" candle")
        for i in pbar:
            current_candle = m5_candles[i]
            candle_time = current_candle['time']
            
            # ANTI-LOOKAHEAD: Strictly use data available AT or BEFORE current_candle[i]
            
            # 1. RESOLVE OPEN TRADE (BID/ASK + REAL-TIME SL/TP)
            if open_trade:
                # Volatility at the moment of resolution (using window up to current M5 candle)
                vol_window = m5_returns[max(0, i-60):i+1] # approx same time as 10 M30 candles
                volatility = np.std(vol_window) if len(vol_window) > 0 else 0
                spread_val = self._get_spread(symbol, volatility) * point
                
                bid_h, bid_l, bid_c = current_candle['high'], current_candle['low'], current_candle['close']
                ask_h, ask_l, ask_c = bid_h + spread_val, bid_l + spread_val, bid_c + spread_val
                
                closed = False
                exit_price = 0
                result_type = ""
                
                if open_trade.signal.direction == "BUY":
                    # BUY: Exit at BID. SL/TP triggered by BID.
                    if bid_l <= open_trade.sl:
                        exit_price, result_type, closed = open_trade.sl, "SL", True
                    elif bid_h >= open_trade.tp:
                        exit_price, result_type, closed = open_trade.tp, "TP", True
                else:
                    # SELL: Exit at ASK. SL/TP triggered by ASK.
                    if ask_h >= open_trade.sl:
                        exit_price, result_type, closed = open_trade.sl, "SL", True
                    elif ask_l <= open_trade.tp:
                        exit_price, result_type, closed = open_trade.tp, "TP", True
                
                # Dynamic Trailing Stop (using config settings)
                ts_cfg = self.config.get("trailing_stop", {})
                if ts_cfg.get("enabled", True):
                    risk = abs(open_trade.entry_price - open_trade.signal.stop_loss)
                    move_to_be_rr = ts_cfg.get("breakeven_at_rr", 1.0)
                    
                    if open_trade.signal.direction == "BUY":
                        dist = bid_h - open_trade.entry_price
                        if not open_trade.partial_closed and dist >= risk * move_to_be_rr:
                            open_trade.sl = open_trade.entry_price
                            open_trade.partial_closed = True # Using this as BE flag
                            if not quiet: pbar.write(f"[{current_candle['time']}] SL moved to BE (+ spread/slippage)")
                        
                        # Phase 2 Trailing (ATR based or step)
                        trail_activation = ts_cfg.get("trail_phase2_at_rr", 1.5)
                        if dist >= risk * trail_activation:
                            # Move SL to half of profit
                            new_sl = open_trade.entry_price + (dist * 0.5)
                            if new_sl > open_trade.sl:
                                open_trade.sl = new_sl
                    else:
                        dist = open_trade.entry_price - ask_l
                        if not open_trade.partial_closed and dist >= risk * move_to_be_rr:
                            open_trade.sl = open_trade.entry_price
                            open_trade.partial_closed = True
                            if not quiet: pbar.write(f"[{current_candle['time']}] SL moved to BE (+ spread/slippage)")
                            
                        trail_activation = ts_cfg.get("trail_phase2_at_rr", 1.5)
                        if dist >= risk * trail_activation:
                            new_sl = open_trade.entry_price - (dist * 0.5)
                            if new_sl < open_trade.sl:
                                open_trade.sl = new_sl
                
                if closed:
                    exit_slippage = self._get_slippage(symbol) * point
                    final_exit = exit_price - exit_slippage if open_trade.signal.direction == "BUY" else exit_price + exit_slippage
                    
                    pnl = (final_exit - open_trade.entry_price) * (1 if open_trade.signal.direction == "BUY" else -1) * contract_size * open_trade.lot
                    
                    trade_record = {
                        "time": datetime.fromtimestamp(open_trade.entry_time, tz=timezone.utc).strftime("%Y-%m-%d %H:%M"),
                        "exit_time": datetime.fromtimestamp(candle_time, tz=timezone.utc).strftime("%Y-%m-%d %H:%M"),
                        "direction": open_trade.signal.direction,
                        "entry": round(open_trade.entry_price, 5),
                        "exit": round(final_exit, 5),
                        "lot": open_trade.lot,
                        "pnl": round(pnl, 2),
                        "result": result_type,
                        "regime": open_trade.regime,
                        "ai_score": round(open_trade.ai_score, 4),
                        "spread": round(open_trade.spread, 2),
                        "slippage": round(open_trade.slippage / point, 2)
                    }
                    trades.append(trade_record)
                    self.balance += pnl
                    # Real-time console feedback (using pbar.write to avoid breaking the progress bar)
                    pbar.write(f"[{trade_record['exit_time']}] CLOSED {trade_record['direction']} | P&L: ${pnl:>8.2f} | Result: {result_type}")
                    pbar.set_postfix(balance=f"${self.balance:.2f}", trades=len(trades))
                    open_trade = None
                    continue

            # 2. SIGNAL GENERATION (ANTI-LOOKAHEAD)
            if not open_trade:
                h4_idx = self._find_slice_index(h4_times, candle_time)
                m30_idx = self._find_slice_index(m30_times, candle_time)
                
                # Slices (inclusive of binary-searched indices)
                h4_slice = h4_candles[:h4_idx + 1]
                m30_slice = m30_candles[:m30_idx + 1]
                m5_slice = m5_candles[:i + 1]
                
                # Volatility for spread/AI features
                vol_window = m5_returns[max(0, i-60):i+1]
                volatility = np.std(vol_window) if len(vol_window) > 0 else 0
                spread_val = self._get_spread(symbol, volatility)
                
                bid = current_candle['close']
                ask = bid + (spread_val * point)
                
                # Analyze strategy (only data safe slices)
                signal, _ = self.strategy.analyze(symbol, h4_slice, m30_slice, m5_slice, bid, d1_candles=d1_candles)
                
                if signal:
                    # Pass minimal data to AI filter (prevent leakage)
                    ai_features = {
                        "direction": signal.direction,
                        "confidence": signal.confidence,
                        "volatility": volatility,
                        "regime": MarketRegime.classify(m30_slice)
                    }
                    ai_decision, ai_score = self.ai_filter.filter_signal(ai_features)
                    
                    if ai_decision:
                        slippage = self._get_slippage(symbol) * point
                        entry = ask + slippage if signal.direction == "BUY" else bid - slippage
                        lot = self._calc_lot_size(self.balance, entry, signal.stop_loss, point, contract_size)
                        
                        open_trade = _OpenTrade(
                            signal, entry, lot, candle_time, 
                            ai_features['regime'], ai_score, spread_val, slippage
                        )
                        pbar.write(f"[{datetime.fromtimestamp(candle_time, tz=timezone.utc).strftime('%Y-%m-%d %H:%M')}] OPENED {signal.direction} @ {entry:.5f} | Lot: {lot}")

        performance = PerformanceMetrics.calculate_metrics(trades, self.initial_balance)
        performance['trades'] = trades
        return performance
