import numpy as np
import logging
from datetime import time, datetime
from typing import Optional, Tuple
from core.base_strategy import BaseStrategy, MarketData
from core.common.types import TradeSignal

logger = logging.getLogger("trading_bot.strategy.liquidity")

class LiquiditySessionStrategy(BaseStrategy):
    """
    V4 Institutional Liquidity / Session.
    Focus: London and NY open volatility breaking the Asian Range.
    
    Improved Version 4 (Aggressive Calibration):
    - Balanced Asian Range Maturity (2.5x ATR).
    - Reduced Volatility Trigger for earlier session entry.
    - Institutional SL/TP with ATR-based volatility targets.
    """

    def __init__(self, strategy_id: str, config: dict):
        super().__init__(strategy_id, config)
        self.asian_high = 0.0
        self.asian_low = 0.0
        self.range_set = False
        self.vol_trigger_mult = 0.5  # Relaxed for higher entry frequency
        self.london_trade_taken = False
        self.ny_trade_taken = False
        
        # Optimization Parameters (Iteration 4)
        self.range_maturity_limit = 4.0  # Relaxed from 2.5 to 4.0 for Gold
        self.tp_mult = 3.0              
        self.min_confidence = 0.70 # Relaxed from 0.85

    def generate_signal(self, market_data: MarketData) -> Optional[TradeSignal]:
        m5 = market_data.m5_candles
        price = market_data.current_price
        dt = market_data.timestamp
        
        # 1. Indicator Context
        atr_vals = m5.atr(14)
        vol_sma_vals = m5.get_indicator('vol_sma_20')
        
        if len(atr_vals) == 0 or len(vol_sma_vals) == 0:
            return None
            
        atr = atr_vals[-1]
        vol_sma = vol_sma_vals[-1]
        
        if np.isnan(atr) or atr <= 0 or np.isnan(vol_sma) or vol_sma <= 0:
            self.last_rejection_reason = "Liquidity: Missing Indicators (ATR/VOL)"
            return None

        # 2. Asian Range Calculation (00:00 - 08:00 UTC)
        current_time = dt.time()
        if time(0, 0) <= current_time < time(8, 0):
            if not self.range_set:
                self.asian_high = price
                self.asian_low = price
                self.range_set = True
            else:
                self.asian_high = max(self.asian_high, price)
                self.asian_low = min(self.asian_low, price)
            return None

        # 3. Active Windows & Session Gating
        is_london = time(8, 0) <= current_time < time(11, 0)
        is_ny = time(13, 0) <= current_time < time(17, 0)
        
        if is_london and self.london_trade_taken: 
            self.last_rejection_reason = "Liquidity: London trade already taken"
            return None
        if is_ny and self.ny_trade_taken: 
            self.last_rejection_reason = "Liquidity: NY trade already taken"
            return None
        if not (is_london or is_ny): 
            self.last_rejection_reason = "Liquidity: Outside session windows"
            return None
        if not self.range_set or self.asian_high == self.asian_low: 
            self.last_rejection_reason = "Liquidity: Asian range not set"
            return None

        # 4. Asian Range Maturity Filter (Relaxed)
        range_height = self.asian_high - self.asian_low
        if range_height > (atr * self.range_maturity_limit):
            self.last_rejection_reason = f"Liquidity: Asian Range too mature ({range_height/atr:.1f}x ATR)"
            return None

        # 5. Breakout Validation with Momentum (Relaxed)
        # Using a minimal buffer to allow the session open momentum to speak for itself
        breakout_buffer = atr * 0.1
        last_candle_body = abs(m5.close[-1] - m5.open[-1])
        last_volume = m5.tick_volume[-1]
        
        is_volatile = last_candle_body > (atr * self.vol_trigger_mult)
        is_high_volume = last_volume > (vol_sma * 1.05)
        
        if not (is_volatile or is_high_volume):
            self.last_rejection_reason = f"Liquidity: Insufficient Breakout Momentum (Vol={last_candle_body/atr:.2f}x ATR)"
            return None

        # 6. Decision Logic
        if price > (self.asian_high + breakout_buffer):
            if is_london: self.london_trade_taken = True
            if is_ny: self.ny_trade_taken = True
            return TradeSignal(direction="BUY", price=price, confidence=0.85, timestamp=dt)
            
        if price < (self.asian_low - breakout_buffer):
            if is_london: self.london_trade_taken = True
            if is_ny: self.ny_trade_taken = True
            return TradeSignal(direction="SELL", price=price, confidence=0.85, timestamp=dt)

        self.last_rejection_reason = "Liquidity: Price inside breakout buffer"
        return None

    def get_metrics(self, market_data: MarketData) -> dict:
        return {
            "asian_high": self.asian_high,
            "asian_low": self.asian_low,
            "range_set": self.range_set,
            "london_trade_taken": self.london_trade_taken,
            "ny_trade_taken": self.ny_trade_taken
        }

    def get_stop_loss(self, signal: TradeSignal, market_data: MarketData) -> float:
        m5 = market_data.m5_candles
        atr_vals = m5.atr(14)
        atr = atr_vals[-1] if len(atr_vals) > 0 else 1.0
        
        if signal.direction == "BUY":
            sl_price = self.asian_low
            min_sl = market_data.current_price - (atr * 1.5)
            # Use 1.5 ATR as a firm floor to survive Monte Carlo Jitter
            return min(sl_price, min_sl)
        else:
            sl_price = self.asian_high
            min_sl = market_data.current_price + (atr * 1.5)
            return max(sl_price, min_sl)

    def get_take_profit(self, signal: TradeSignal, market_data: MarketData) -> float:
        m5 = market_data.m5_candles
        atr_vals = m5.atr(14)
        atr = atr_vals[-1] if len(atr_vals) > 0 else 1.0
        
        if signal.direction == "BUY":
            return market_data.current_price + (atr * self.tp_mult)
        return market_data.current_price - (atr * self.tp_mult)

    def reset_daily_stats(self) -> None:
        self.asian_high = 0.0
        self.asian_low = 0.0
        self.range_set = False
        self.london_trade_taken = False
        self.ny_trade_taken = False
