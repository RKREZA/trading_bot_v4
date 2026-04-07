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
    Asian Range: 00:00 - 08:00 UTC.
    London Open: 08:00 UTC. NY Open: 13:00 UTC.
    
    Hardened for Certification:
    - ATR-based Volatility Buffer for breakout validation.
    - Institutional SL Floor (1.5x ATR) to prevent 'Infinite Leverage' crashes.
    - ATR-Based Take Profit (3.5x ATR) for realistic targets.
    - Session-limited execution (1 trade per open).
    """

    def __init__(self, strategy_id: str, config: dict):
        super().__init__(strategy_id, config)
        self.asian_high = 0.0
        self.asian_low = 0.0
        self.range_set = False
        self.vol_trigger_mult = 1.0 # Calibrated to allow high-conviction trades
        self.london_trade_taken = False
        self.ny_trade_taken = False

    def generate_signal(self, market_data: MarketData) -> Optional[TradeSignal]:
        m5 = market_data.m5_candles
        price = market_data.current_price
        dt = market_data.timestamp
        
        # 1. Indicator Context (Institutional IPC - Limit Aware)
        # BUG FIX: Use helper methods for limit-aware indicator access
        atr_vals = m5.atr(14)
        vol_sma_vals = m5.get_indicator('vol_sma_20')
        
        if len(atr_vals) == 0 or len(vol_sma_vals) == 0:
            return None
            
        atr = atr_vals[-1]
        vol_sma = vol_sma_vals[-1]
        
        if np.isnan(atr) or atr <= 0 or np.isnan(vol_sma) or vol_sma <= 0:
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
        
        if is_london and self.london_trade_taken: return None
        if is_ny and self.ny_trade_taken: return None
        
        if not (is_london or is_ny):
            return None

        if not self.range_set or self.asian_high == self.asian_low:
            return None

        # 4. Breakout Validation with ATR Buffer (0.2 ATR)
        breakout_buffer = atr * 0.2
        
        # 5. Institutional Liquidity Filter (Price OR Volume Expansion)
        last_candle_body = abs(m5.close[-1] - m5.open[-1])
        last_volume = m5.tick_volume[-1]
        
        is_volatile = last_candle_body > (atr * self.vol_trigger_mult)
        is_high_volume = last_volume > (vol_sma * 1.2) # 20% volume expansion
        
        if not (is_volatile or is_high_volume):
            return None

        # 6. Decision Logic: High-Fidelity Signal
        if price > (self.asian_high + breakout_buffer):
            if is_london: self.london_trade_taken = True
            if is_ny: self.ny_trade_taken = True
            return TradeSignal(direction="BUY", price=price, confidence=0.88, timestamp=dt)
            
        if price < (self.asian_low - breakout_buffer):
            if is_london: self.london_trade_taken = True
            if is_ny: self.ny_trade_taken = True
            return TradeSignal(direction="SELL", price=price, confidence=0.88, timestamp=dt)

        return None

    def get_stop_loss(self, signal: TradeSignal, market_data: MarketData) -> float:
        """Institutional SL: Boundary or 1.5x ATR floor."""
        m5 = market_data.m5_candles
        atr_vals = m5.atr(14)
        atr = atr_vals[-1] if len(atr_vals) > 0 else 1.0
        
        if signal.direction == "BUY":
            sl_price = self.asian_low
            min_sl = market_data.current_price - (atr * 1.5)
            return min(sl_price, min_sl)
        else:
            sl_price = self.asian_high
            min_sl = market_data.current_price + (atr * 1.5)
            return max(sl_price, min_sl)

    def get_take_profit(self, signal: TradeSignal, market_data: MarketData) -> float:
        """Institutional TP: Decoupled 3.5x ATR target."""
        m5 = market_data.m5_candles
        atr_vals = m5.atr(14)
        atr = atr_vals[-1] if len(atr_vals) > 0 else 1.0
        
        if signal.direction == "BUY":
            return market_data.current_price + (atr * 3.5)
        return market_data.current_price - (atr * 3.5)

    def reset_daily_stats(self) -> None:
        """Reset Asian Range and Session Flags for the new day."""
        self.asian_high = 0.0
        self.asian_low = 0.0
        self.range_set = False
        self.london_trade_taken = False
        self.ny_trade_taken = False
