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
        self.body_ratio_threshold = 0.70
        self.body_size_mult = 1.5

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
        m1 = market_data.m1_candles
        if m1 is None or len(m1) < 50: return None
        current_price = market_data.current_price
        
        # 1. CLEANUP
        for pid, grid in list(self.active_grids.items()):
            if grid['direction'] == 'BUY' and current_price >= grid['tp']:
                del self.active_grids[pid]
            elif grid['direction'] == 'SELL' and current_price <= grid['tp']:
                del self.active_grids[pid]

        # 2. PATTERN DETECTION
        avg_body = np.mean(np.abs(m1.o[-50:-1] - m1.c[-50:-1]))
        current_session = market_data.session
        
        for i in range(len(m1) - 30, len(m1) - 1):
            o, h, l_val, c_val = m1.o[i], m1.h[i], m1.l[i], m1.c[i]
            if self._is_big_candle(o, h, l_val, c_val, avg_body, session=current_session):
                is_bullish = c_val > o
                c_range = h - l_val
                pid = f"{'bull' if is_bullish else 'bear'}_{i}_{m1.time[i]}"
                if pid in self.active_grids: continue
                
                # V5-INSIGNIA PROFESSIONAL STANDARDS
                if current_session in ["TOKYO", "ASIA", "ROLLOVER"]:
                    r_pct = 0.80 # Deep snipe
                    sl_div = 3.0 # High precision
                else:
                    r_pct = 0.60 # Momentum capture
                    sl_div = 1.5 # Volatility buffer
                
                if is_bullish:
                    retrace_level = h - (c_range * r_pct)
                    history_since = m1.l[i+1:-1]
                    if m1.l[-1] <= retrace_level and (len(history_since) == 0 or not np.any(history_since <= retrace_level)):
                        tp_dist = h - current_price
                        sl = current_price - (tp_dist / sl_div)
                        self.active_grids[pid] = {
                            'direction': 'BUY', 'tp': h, 'sl': sl, 'last_entry': current_price, 'count': 1, 'symbol': market_data.symbol
                        }
                        logger.info(f"Signal: Bullish N ({current_session}) at {current_price}, TP: {h}, SL: {sl}")
                        return TradeSignal(direction="BUY", confidence=1.0, price=current_price, stop_loss=sl, take_profit=h, volume=self.fixed_lot)
                else:
                    retrace_level = l_val + (c_range * r_pct)
                    history_since = m1.h[i+1:-1]
                    if m1.h[-1] >= retrace_level and (len(history_since) == 0 or not np.any(history_since >= retrace_level)):
                        tp_dist = current_price - l_val
                        sl = current_price + (tp_dist / sl_div)
                        self.active_grids[pid] = {
                            'direction': 'SELL', 'tp': l_val, 'sl': sl, 'last_entry': current_price, 'count': 1, 'symbol': market_data.symbol
                        }
                        logger.info(f"Signal: Bearish N ({current_session}) at {current_price}, TP: {l_val}, SL: {sl}")
                        return TradeSignal(direction="SELL", confidence=1.0, price=current_price, stop_loss=sl, take_profit=l_val, volume=self.fixed_lot)

        # 3. GRID
        for pid, grid in self.active_grids.items():
            if grid['direction'] == 'BUY' and current_price <= grid['last_entry'] - 1.50:
                grid['last_entry'] = current_price
                grid['count'] += 1
                return TradeSignal(direction="BUY", confidence=1.0, price=current_price, stop_loss=0, take_profit=grid['tp'], volume=self.fixed_lot)
            elif grid['direction'] == 'SELL' and current_price >= grid['last_entry'] + 1.50:
                grid['last_entry'] = current_price
                grid['count'] += 1
                return TradeSignal(direction="SELL", confidence=1.0, price=current_price, stop_loss=0, take_profit=grid['tp'], volume=self.fixed_lot)

        return None

    def get_metrics(self, market_data: MarketData) -> Dict[str, Any]:
        return {"active_grids": len(self.active_grids), "total_positions": sum(g['count'] for g in self.active_grids.values())}

    def on_trade_closed(self, trade_record: dict) -> None:
        pass
