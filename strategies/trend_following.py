import numpy as np
import logging
from typing import Optional
from core.base_strategy import BaseStrategy, MarketData
from core.types import TradeSignal

logger = logging.getLogger("trading_bot.strategy.trend")

class TrendFollowingStrategy(BaseStrategy):
    """
    Institutional Trend Following Strategy.
    Uses M5 EMA 50/200 alignment with H1 Trend confirmation.
    """

    def __init__(self, strategy_id: str, config: dict):
        super().__init__(strategy_id, config)
        p = config.get("params", {})
        self.ema_fast = int(p.get("ema_fast", 50))
        self.ema_slow = int(p.get("ema_slow", 200))
        self.atr_period = 14
        self.sl_atr_mult = float(p.get("sl_atr_mult", 2.5))
        self.tp_rr = float(p.get("tp_rr", 4.0))

    def generate_signal(self, market_data: MarketData) -> TradeSignal:
        m5 = market_data.m5_candles
        h1 = market_data.htf_candles
        price = market_data.current_price

        if len(m5) < self.ema_slow or len(h1) < 20:
            return TradeSignal(direction="NONE")

        # 1. H1 Trend Confirmation (Close > SMA 20)
        h1_ema = np.mean(h1.close[-20:])
        h1_bull = h1.close[-1] > h1_ema
        h1_bear = h1.close[-1] < h1_ema

        # 2. M5 EMA Alignment
        ema50 = np.mean(m5.close[-self.ema_fast:])
        ema200 = np.mean(m5.close[-self.ema_slow:])
        
        m5_bull = ema50 > ema200 and price > ema50
        m5_bear = ema50 < ema200 and price < ema50

        # 3. Pullback check (Price must be close to EMA50 to avoid overextended entries)
        atr = self._calculate_atr(m5)
        dist_to_ema = abs(price - ema50)
        pullback_ok = dist_to_ema < (atr * 1.5)

        if h1_bull and m5_bull and pullback_ok:
            return TradeSignal(direction="BUY", confidence=0.8, timestamp=market_data.timestamp)
        
        if h1_bear and m5_bear and pullback_ok:
            return TradeSignal(direction="SELL", confidence=0.8, timestamp=market_data.timestamp)

        return TradeSignal(direction="NONE")

    def get_stop_loss(self, signal: TradeSignal, market_data: MarketData) -> float:
        base_sl = getattr(signal, "stop_loss", 0.0)
        entry_price = getattr(signal, "price", market_data.current_price)
        atr = self._calculate_atr(market_data.m5_candles)
        
        if signal.direction == "BUY":
            # Trailing Stop: 2.0x ATR from current price
            trail_sl = market_data.current_price - (atr * 2.0)
            return max(base_sl, trail_sl)
        else:
            trail_sl = market_data.current_price + (atr * 2.0)
            return min(base_sl, trail_sl)

    def get_take_profit(self, signal: TradeSignal, market_data: MarketData) -> float:
        sl_dist = abs(market_data.current_price - self.get_stop_loss(signal, market_data))
        if signal.direction == "BUY":
            return market_data.current_price + (sl_dist * self.tp_rr)
        return market_data.current_price - (sl_dist * self.tp_rr)

    def _calculate_atr(self, candles) -> float:
        h, l, c = candles.high, candles.low, candles.close
        tr = np.maximum(h[1:] - l[1:], np.maximum(np.abs(h[1:] - c[:-1]), np.abs(l[1:] - c[:-1])))
        return float(np.mean(tr[-self.atr_period:]))
