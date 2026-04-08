import numpy as np
import logging
from typing import Optional
from core.base_strategy import BaseStrategy, MarketData
from core.common.types import TradeSignal

logger = logging.getLogger("trading_bot.strategy.breakout")

class BreakoutStrategy(BaseStrategy):
    """
    V4 Institutional Breakout.
    Targeting High-Momentum Range Breaks with Candle Strength filters.
    Rule: Enter on Breakout High/Low + Body Size > 70% of total candle range.
    """

    def __init__(self, strategy_id: str, config: dict):
        super().__init__(strategy_id, config)
        self.lookback = 20
        self.body_thresh = 0.65 # Relaxed from 0.75
        self.h1_strength_thresh = 0.50 # Relaxed from 0.60
        self.min_confidence = 0.70 # Relaxed from 0.75

    def generate_signal(self, market_data: MarketData) -> Optional[TradeSignal]:
        """
        Institutional MTF Breakout (Step 11).
        Requires H1 Momentum and M15 Trend alignment.
        """
        # 1. MTF Confirmation (H1 / M15) / Volume Consensus
        h1_candles = market_data.htf_candles
        if len(h1_candles) < 22: 
            self.last_rejection_reason = "Breakout: H1 Insufficient data"
            return None
            
        h1 = h1_candles[-1]
        m15_trend = self.get_ema_trend(market_data.m15_candles)
        
        # H1 Candle Body Strength
        h1_candle_range = h1.high - h1.low
        h1_body = abs(h1.close - h1.open)
        h1_strength = (h1_body / h1_candle_range) if h1_candle_range > 0 else 0
        h1_dir = 1 if h1.close > h1.open else -1
        
        # H1 Volume Confirmation with Time-Weighted Scaling (Audit Fix)
        # Prevents rejection during the first minutes of a new hour
        h1_v = h1_candles.v
        h1_vol_sma = np.mean(h1_v[-21:-1]) 
        
        # Calculate completion percentage of the current H1 candle
        minutes_into_hour = market_data.timestamp.minute
        completion_pct = max(0.05, (minutes_into_hour + 1) / 60.0) # Min 5% to avoid zero-vol pass
        
        # Scale the threshold by completion percentage
        dynamic_threshold = h1_vol_sma * completion_pct
        vol_confirmed = h1.tick_volume > dynamic_threshold
        
        if not vol_confirmed:
            self.last_rejection_reason = f"Breakout: Vol {h1.tick_volume:.0f} < Dynamic {dynamic_threshold:.0f} ({completion_pct:.1%})"
        
        m5 = market_data.m5_candles
        if len(m5) < self.lookback + 1:
            self.last_rejection_reason = "Breakout: M5 Insufficient data"
            return None

        # 2. Local Range Calculation
        prev_range = m5[-self.lookback-1:-1]
        r_high = np.max(prev_range.high)
        r_low = np.min(prev_range.low)

        # 3. M5 Strength Assessment
        last = m5[-1]
        price = market_data.current_price
        m5_range = last.high - last.low
        m5_strength = (abs(last.close - last.open) / m5_range) if m5_range > 0 else 0

        # 4. Integrated Decision Logic
        reasons = []
        if h1_strength > 0.6: reasons.append(f"H1: Strong {'Bullish' if h1_dir == 1 else 'Bearish'} ({h1_strength:.2f})")
        if m5_strength > self.body_thresh: reasons.append(f"M5: Institutional Break ({m5_strength:.2f})")
        if vol_confirmed: reasons.append("Volume: Above Average (Institutional Presence)")
        
        if price > r_high: reasons.append("Price: Above Range High")
        elif price < r_low: reasons.append("Price: Below Range Low")
        else: reasons.append("Price: Inside Range")

        # BUY: Price > High AND Strength > 75% AND H1 Bullish + Strong AND M15 NOT Bearish AND Volume OK
        if price > r_high and m5_strength >= self.body_thresh:
            if h1_dir == 1 and h1_strength >= self.h1_strength_thresh and m15_trend != -1 and vol_confirmed:
                return TradeSignal(direction="BUY", price=price, confidence=0.95, timestamp=market_data.timestamp, reasons=reasons)
        
        # SELL: Price < Low AND Strength > 75% AND H1 Bearish + Strong AND M15 NOT Bullish AND Volume OK
        if price < r_low and m5_strength >= self.body_thresh:
            if h1_dir == -1 and h1_strength >= self.h1_strength_thresh and m15_trend != 1 and vol_confirmed:
                return TradeSignal(direction="SELL", price=price, confidence=0.95, timestamp=market_data.timestamp, reasons=reasons)
        
        # Diagnostics
        if price > r_high or price < r_low:
             if m5_strength < self.body_thresh: self.last_rejection_reason = f"Breakout: M5 Strength too low ({m5_strength:.2f})"
             elif h1_strength < self.h1_strength_thresh: self.last_rejection_reason = f"Breakout: H1 Strength too low ({h1_strength:.2f})"
             elif not vol_confirmed: self.last_rejection_reason = "Breakout: Volume not confirmed"
             else: self.last_rejection_reason = "Breakout: MTF/Dir Mismatch"
        else:
             self.last_rejection_reason = "Breakout: Price inside range"

        # Fallback: Report Bias
        return TradeSignal(direction="NONE", price=price, confidence=0.5, reasons=reasons)

    def get_stop_loss(self, signal: TradeSignal, market_data: MarketData) -> float:
        last = market_data.m5_candles[-1]
        atr_vals = market_data.m5_candles.atr(14)
        atr = atr_vals[-1] if len(atr_vals) > 0 and not np.isnan(atr_vals[-1]) else 1.0
        
        if signal.direction == "BUY":
            return min(last.low, market_data.current_price - (atr * 2.5))
        return max(last.high, market_data.current_price + (atr * 2.5))

    def get_take_profit(self, signal: TradeSignal, market_data: MarketData) -> float:
        sl_price = self.get_stop_loss(signal, market_data)
        risk = abs(market_data.current_price - sl_price)
        # Breakouts play for explosive moves
        if signal.direction == "BUY":
            return market_data.current_price + (risk * 3.5)
        return market_data.current_price - (risk * 3.5)
