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
    """

    def __init__(self, strategy_id: str, config: dict):
        super().__init__(strategy_id, config)
        self.asian_high = 0.0
        self.asian_low = 0.0
        self.range_set = False

    def generate_signal(self, market_data: MarketData) -> Optional[TradeSignal]:
        m5 = market_data.m5_candles
        price = market_data.current_price
        dt = market_data.timestamp
        
        # 1. Asian Range Calculation (00:00 - 08:00 UTC)
        # Assuming broker time is UTC or adjusted for this logic
        current_time = dt.time()
        
        if time(0, 0) <= current_time < time(8, 0):
            # Building range
            if not self.range_set:
                self.asian_high = price
                self.asian_low = price
                self.range_set = True
            else:
                self.asian_high = max(self.asian_high, price)
                self.asian_low = min(self.asian_low, price)
            return None

        # 2. Breakout Windows (London/NY Open)
        # London: 08:00 - 10:00 | NY: 13:00 - 15:00
        is_london = time(8, 0) <= current_time < time(10, 0)
        is_ny = time(13, 0) <= current_time < time(15, 0)
        
        if not (is_london or is_ny):
            return None

        if not self.range_set or self.asian_high == self.asian_low:
            return None

        # 3. Decision Logic: Break Asian Range with Volume/Momentum
        if price > self.asian_high:
            return TradeSignal(direction="BUY", confidence=0.80, timestamp=dt)
            
        if price < self.asian_low:
            return TradeSignal(direction="SELL", confidence=0.80, timestamp=dt)

        return None

    def get_stop_loss(self, signal: TradeSignal, market_data: MarketData) -> float:
        # Stop at the other side of the Asian Range
        if signal.direction == "BUY":
            return self.asian_low
        return self.asian_high

    def get_take_profit(self, signal: TradeSignal, market_data: MarketData) -> float:
        # Standard Institutional R:R target
        sl_price = self.get_stop_loss(signal, market_data)
        risk = abs(market_data.current_price - sl_price)
        if signal.direction == "BUY":
            return market_data.current_price + (risk * 2.5)
        return market_data.current_price - (risk * 2.5)

    def reset_daily_stats(self) -> None:
        """Reset Asian Range for the new day."""
        self.asian_high = 0.0
        self.asian_low = 0.0
        self.range_set = False
