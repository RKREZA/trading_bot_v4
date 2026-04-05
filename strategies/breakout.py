import numpy as np
import logging
from core.base_strategy import BaseStrategy, MarketData
from core.types import TradeSignal

logger = logging.getLogger("trading_bot.strategy.breakout")

class BreakoutStrategy(BaseStrategy):
    """
    Institutional Breakout Strategy.
    Uses previous N-bar range and candle body strength filters for entry.
    Targeting explosive, high-momentum moves.
    """

    def __init__(self, strategy_id: str, config: dict):
        super().__init__(strategy_id, config)
        p = config.get("params", {})
        self.lookback = int(p.get("lookback_period", 20))
        self.body_strength_threshold = float(p.get("body_strength_threshold", 0.70)) # 70% body size relative to candle range
        self.tp_rr = float(p.get("tp_rr", 4.0)) # High RR for breakouts
        self.atr_period = 14

    def generate_signal(self, market_data: MarketData) -> TradeSignal:
        m5 = market_data.m5_candles
        price = market_data.current_price

        if len(m5) < self.lookback + 1:
            return TradeSignal(direction="NONE")

        # 1. Range Calculation (excluding the current candle)
        prev_range = m5[-self.lookback-1:-1]
        range_high = np.max(prev_range.high)
        range_low = np.min(prev_range.low)

        # 2. Breakout detection
        last_close = m5.close[-1]
        last_open = m5.open[-1]
        last_high = m5.high[-1]
        last_low = m5.low[-1]

        # 3. Candle Body Strength Filter
        candle_range = abs(last_high - last_low)
        body_size = abs(last_close - last_open)
        body_strength = (body_size / candle_range) if candle_range > 0 else 0

        # Entry logic: price must break range and the candle must be strong
        if last_close > range_high and body_strength >= self.body_strength_threshold:
            return TradeSignal(direction="BUY", confidence=0.85, timestamp=market_data.timestamp)

        if last_close < range_low and body_strength >= self.body_strength_threshold:
            return TradeSignal(direction="SELL", confidence=0.85, timestamp=market_data.timestamp)

        return TradeSignal(direction="NONE")

    def get_stop_loss(self, signal: TradeSignal, market_data: MarketData) -> float:
        m5 = market_data.m5_candles
        # SL usually goes at the other end of the breakout candle or middle of the range
        if signal.direction == "BUY":
            return m5.low[-1] # Tight stop at breakout candle low
        return m5.high[-1]

    def get_take_profit(self, signal: TradeSignal, market_data: MarketData) -> float:
        sl_dist = abs(market_data.current_price - self.get_stop_loss(signal, market_data))
        if signal.direction == "BUY":
            return market_data.current_price + (sl_dist * self.tp_rr) 
        return market_data.current_price - (sl_dist * self.tp_rr)

    def _calculate_atr(self, candles) -> float:
        h, l, c = candles.high, candles.low, candles.close
        tr = np.maximum(h[1:] - l[1:], np.maximum(np.abs(h[1:] - c[:-1]), np.abs(l[1:] - c[:-1])))
        return float(np.mean(tr[-self.atr_period:]))
