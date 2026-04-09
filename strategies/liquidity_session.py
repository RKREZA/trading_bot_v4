import numpy as np
import logging
from datetime import time, datetime
from typing import Optional, Tuple
from core.base_strategy import BaseStrategy, MarketData
from core.common.types import TradeSignal

logger = logging.getLogger("trading_bot.strategy.liquidity")

class LiquiditySessionStrategy(BaseStrategy):
    """
    V4 Institutional Liquidity / Session (Dynamic Edition).
    Focus: London and NY open volatility breaking the Asian Range.
    """

    def __init__(self, strategy_id: str, config: dict):
        super().__init__(strategy_id, config)
        self.asian_high = 0.0
        self.asian_low = 0.0
        self.range_set = False
        self.london_trade_taken = False
        self.ny_trade_taken = False
        
        # Pull parameters dynamically from config
        # Priority: strategy_id -> "LiquiditySession"
        strat_config = self.config.get(strategy_id, self.config.get("LiquiditySession", {}))
        self.enabled = strat_config.get("enabled", True)
        self.range_maturity_limit = strat_config.get("range_maturity_limit", 4.0)  
        self.vol_trigger_mult = strat_config.get("vol_trigger_mult", 0.5)
        self.sl_atr = strat_config.get("sl_atr", 1.5)
        self.rr_target = strat_config.get("rr_target", 8.0) 

    def generate_signal(self, market_data: MarketData) -> Optional[TradeSignal]:
        m5 = market_data.m5_candles
        price = market_data.current_price
        dt = market_data.timestamp
        
        atr_vals = m5.atr(14)
        if len(atr_vals) == 0 or len(m5.v) < 20:
            return None
            
        atr = atr_vals[-1]
        vol_sma = np.mean(m5.v[-20:])
        
        if np.isnan(atr) or atr <= 0 or np.isnan(vol_sma) or vol_sma <= 0:
            return None

        # Asian Range Calculation (00:00 - 08:00 UTC)
        current_time = dt.time()
        if time(0, 0) <= current_time < time(8, 0):
            if not self.range_set:
                self.asian_high = price
                self.asian_low = price
                self.range_set = True
            else:
                self.asian_high = max(self.asian_high, price)
                self.asian_low = min(self.asian_low, price)
            self.last_rejection_reason = "Liquidity: Building Asian Range"
            return None

        # Active Windows Gating
        is_london = time(8, 0) <= current_time < time(11, 0)
        is_ny = time(13, 0) <= current_time < time(17, 0)
        
        if is_london and self.london_trade_taken: return None
        if is_ny and self.ny_trade_taken: return None
        if not (is_london or is_ny): return None
        if not self.range_set or self.asian_high == self.asian_low: return None

        # Asian Range Maturity Filter
        range_height = self.asian_high - self.asian_low
        if range_height > (atr * self.range_maturity_limit):
            self.last_rejection_reason = f"Liquidity: Range too mature ({range_height/atr:.1f}x ATR)"
            return None

        # Breakout Validation with Momentum
        breakout_buffer = atr * 0.1
        last_candle = m5[-1]
        last_candle_body = abs(last_candle.close - last_candle.open)
        
        is_volatile = last_candle_body > (atr * self.vol_trigger_mult)
        is_high_volume = last_candle.tick_volume > (vol_sma * 1.05)
        
        if not (is_volatile or is_high_volume):
            self.last_rejection_reason = f"Liquidity: Insufficient Momentum"
            return None

        # Decision Logic
        if price > (self.asian_high + breakout_buffer):
            if is_london: self.london_trade_taken = True
            if is_ny: self.ny_trade_taken = True
            return TradeSignal(direction="BUY", price=price, confidence=0.85, timestamp=dt)
            
        if price < (self.asian_low - breakout_buffer):
            if is_london: self.london_trade_taken = True
            if is_ny: self.ny_trade_taken = True
            return TradeSignal(direction="SELL", price=price, confidence=0.85, timestamp=dt)

        return None

    def get_metrics(self, market_data: MarketData) -> dict:
        return {
            "asian_high": self.asian_high,
            "asian_low": self.asian_low,
            "range_set": self.range_set,
            "london_trade_taken": self.london_trade_taken,
            "ny_trade_taken": self.ny_trade_taken
        }

    def get_thresholds(self) -> dict:
        return {
            "Range Limit": f"< {self.range_maturity_limit} ATR",
            "Breakout Vol": f"> {self.vol_trigger_mult} ATR"
        }

    def get_stop_loss(self, signal: TradeSignal, market_data: MarketData) -> float:
        m5 = market_data.m5_candles
        atr_vals = m5.atr(14)
        atr = atr_vals[-1] if len(atr_vals) > 0 else 1.0
        
        # Dynamic, strict ATR Stop Loss
        if signal.direction == "BUY":
            return market_data.current_price - (atr * self.sl_atr)
        else:
            return market_data.current_price + (atr * self.sl_atr)

    def get_take_profit(self, signal: TradeSignal, market_data: MarketData) -> float:
        m5 = market_data.m5_candles
        atr_vals = m5.atr(14)
        atr = atr_vals[-1] if len(atr_vals) > 0 else 1.0
        
        # Dynamic RR Target (High for Trailing)
        risk = atr * self.sl_atr
        if signal.direction == "BUY":
            return market_data.current_price + (risk * self.rr_target)
        else:
            return market_data.current_price - (risk * self.rr_target)

    def reset_daily_stats(self) -> None:
        self.asian_high = 0.0
        self.asian_low = 0.0
        self.range_set = False
        self.london_trade_taken = False
        self.ny_trade_taken = False