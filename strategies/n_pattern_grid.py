import numpy as np
import logging
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List
from core.base_strategy import BaseStrategy, MarketData
from core.common.types import TradeSignal

logger = logging.getLogger("trading_bot.strategy.n_pattern_grid")

class NPatternGridStrategy(BaseStrategy):
    """
    M1 N-Pattern Grid Strategy.
    Detects "Big Candle" + "Retrace" setup.
    """

    def __init__(self, strategy_id: str, config: dict):
        super().__init__(strategy_id, config)
        self.fixed_lot = 0.05
        self.retrace_pct = 0.85
        self.grid_pips = 1.50 # 15 pips for XAUUSDm
        self.active_grids = {}

    def _is_big_candle(self, open_p, high, low, close, avg_body, session="GLOBAL"):
        
        body_ratio_thresh = 0.90
        size_mult = 1
            
        body_size = abs(close - open_p)
        candle_range = high - low
        if candle_range == 0: return False
        body_ratio = body_size / candle_range
        is_large = body_size > (avg_body * size_mult)
        return body_ratio > body_ratio_thresh and is_large

    def generate_signal(self, market_data: MarketData) -> Optional[TradeSignal]:
        self.last_rejection_reason = "No Pattern"
        # Institutional Adaptation: Use M1 as the primary signal engine
        m1 = market_data.m1_candles
        if m1 is None or len(m1) < 50: return None
        current_price = market_data.current_price
        
        # 1. CLEANUP
        for pid, grid in list(self.active_grids.items()):
            if grid['direction'] == 'BUY' and (current_price >= grid['tp'] or current_price <= grid['sl']):
                del self.active_grids[pid]
            elif grid['direction'] == 'SELL' and (current_price <= grid['tp'] or current_price >= grid['sl']):
                del self.active_grids[pid]

        # 2. PATTERN DETECTION (with Impulse Candle Fusion)
        avg_body = np.mean(np.abs(m1.o[-50:-1] - m1.c[-50:-1]))
        current_session = market_data.session
        
        # We look for the most recent pattern ending at Bar -1 or -2
        for i in range(len(m1) - 1, len(m1) - 15, -1):
            # Check if Bar i is a big candle
            if not self._is_big_candle(m1.o[i], m1.h[i], m1.l[i], m1.c[i], avg_body, session=current_session):
                continue
                
            is_bullish = m1.c[i] > m1.o[i]
            
            # --- IMPULSE CANDLE FUSION ---
            # Group consecutive candles in the same direction
            first_idx = i
            while first_idx > 0:
                prev = first_idx - 1
                if (m1.c[prev] > m1.o[prev]) == is_bullish and \
                   self._is_big_candle(m1.o[prev], m1.h[prev], m1.l[prev], m1.c[prev], avg_body, session=current_session):
                    first_idx = prev
                else:
                    break
            
            # Virtual Candle Construction
            v_open = m1.o[first_idx]
            v_close = m1.c[i]
            v_high = np.max(m1.h[first_idx:i+1])
            v_low = np.min(m1.l[first_idx:i+1])
            v_range = v_high - v_low
            
            pid = f"{'bull' if is_bullish else 'bear'}_{first_idx}_{str(m1.time[first_idx])}"
            if pid in self.active_grids: continue
            
            # Institutional Gating: Dynamic Retracement Analyzer
            r_pct = self._calculate_historical_retrace(m1)
            # Cap it between 50% and 85% to avoid extreme edge cases
            r_pct = max(0.50, min(0.85, r_pct))
            
            # 3. ATR VOLATILITY BUFFER (Disabled for No-SL experiment)
            atr = self._calculate_atr(m1)
            
            if is_bullish:
                retrace_level = v_high - (v_range * r_pct)
                history_since = m1.l[i+1:]
                if len(history_since) > 0 and m1.l[-1] <= retrace_level and not np.any(history_since[:-1] <= retrace_level):
                    # No-SL Approach
                    sl = 0.0 
                    tp = v_high 
                    
                    self.active_grids[pid] = {
                        'direction': 'BUY', 'tp': tp, 'sl': sl, 'last_entry': current_price, 'count': 1, 'symbol': market_data.symbol
                    }
                    logger.info(f"Signal: Bullish N-Fusion ({current_session}) at {current_price}, TP: {tp}, SL: {sl}")
                    return TradeSignal(direction="BUY", confidence=1.0, price=current_price, stop_loss=sl, take_profit=tp, volume=self.fixed_lot)
            else:
                retrace_level = v_low + (v_range * r_pct)
                history_since = m1.h[i+1:]
                if len(history_since) > 0 and m1.h[-1] >= retrace_level and not np.any(history_since[:-1] >= retrace_level):
                    # No-SL Approach
                    sl = 0.0
                    tp = v_low
                    
                    self.active_grids[pid] = {
                        'direction': 'SELL', 'tp': tp, 'sl': sl, 'last_entry': current_price, 'count': 1, 'symbol': market_data.symbol
                    }
                    logger.info(f"Signal: Bearish N-Fusion ({current_session}) at {current_price}, TP: {tp}, SL: {sl}")
                    return TradeSignal(direction="SELL", confidence=1.0, price=current_price, stop_loss=sl, take_profit=tp, volume=self.fixed_lot)




        # 3. GRID (with Safety SL)
        for pid, grid in self.active_grids.items():
            if grid['direction'] == 'BUY' and current_price <= grid['last_entry'] - 1.50:
                grid['last_entry'] = current_price
                grid['count'] += 1
                tp_dist = grid['tp'] - current_price
                sl = 0.0 # No SL
                return TradeSignal(direction="BUY", confidence=1.0, price=current_price, stop_loss=sl, take_profit=grid['tp'], volume=self.fixed_lot)
            elif grid['direction'] == 'SELL' and current_price >= grid['last_entry'] + 1.50:
                grid['last_entry'] = current_price
                grid['count'] += 1
                tp_dist = current_price - grid['tp']
                sl = 0.0 # No SL
                return TradeSignal(direction="SELL", confidence=1.0, price=current_price, stop_loss=sl, take_profit=grid['tp'], volume=self.fixed_lot)

        return None

    def _calculate_historical_retrace(self, m1) -> float:
        """
        Institutional Forensic: Scans the last 300 bars for Big Candle impulses 
        and calculates the average retracement depth before the trend continued.
        """
        try:
            depths = []
            lookback = 300
            if len(m1) < lookback: lookback = len(m1)
            
            # Use same body logic as signal engine
            avg_body = np.mean(np.abs(m1.o[-50:-1] - m1.c[-50:-1]))
            
            for i in range(len(m1) - 50, len(m1) - lookback, -1):
                # Is it a big candle?
                body = abs(m1.c[i] - m1.o[i])
                if body < avg_body * 1.5: continue
                
                is_bull = m1.c[i] > m1.o[i]
                v_high = m1.h[i]
                v_low = m1.l[i]
                v_range = v_high - v_low
                
                # Look at the next 20 bars to see the max retracement before a NEW high/low was made
                # or before the candle was invalidated (price broke below/above candle)
                future = m1[i+1:i+21]
                if len(future) == 0: continue
                
                if is_bull:
                    # How deep did it go relative to the impulse range?
                    retrace_min = np.min(future.l)
                    retrace_depth = (v_high - retrace_min) / v_range if v_range > 0 else 0
                    if 0.2 < retrace_depth < 1.1: # Reasonable retracement
                        depths.append(retrace_depth)
                else:
                    retrace_max = np.max(future.h)
                    retrace_depth = (retrace_max - v_low) / v_range if v_range > 0 else 0
                    if 0.2 < retrace_depth < 1.1:
                        depths.append(retrace_depth)
                        
            if len(depths) > 0:
                return float(np.mean(depths))
        except Exception as e:
            logger.debug(f"Retrace Analyzer failed: {e}")
            
        return 0.618 # Fallback to Golden Ratio

    def get_metrics(self, market_data: MarketData) -> Dict[str, Any]:
        return {"active_grids": len(self.active_grids), "total_positions": sum(g['count'] for g in self.active_grids.values())}

    def on_trade_closed(self, trade_record: dict) -> None:
        pass

    def _calculate_atr(self, m1, period=14) -> float:
        """Calculates Average True Range for volatility buffering."""
        try:
            highs = m1.h[-period-1:-1]
            lows = m1.l[-period-1:-1]
            closes = m1.c[-period-2:-2]
            
            tr1 = highs - lows
            tr2 = np.abs(highs - closes)
            tr3 = np.abs(lows - closes)
            
            tr = np.maximum(tr1, np.maximum(tr2, tr3))
            return float(np.mean(tr))
        except:
            return 0.50 # Fallback for XAUUSDm
