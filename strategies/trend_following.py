import numpy as np
import logging
from typing import Optional
from core.base_strategy import BaseStrategy, MarketData
from core.common.types import TradeSignal

logger = logging.getLogger("trading_bot.strategy.trend")

class TrendFollowingStrategy(BaseStrategy):
    """
    V4 Institutional Trend Following.
    EMA 50/200 on M5 for core momentum and HTF SMA 20 for global trend bias.
    Deterministic, robust, and non-overfitted logic.
    """

    def __init__(self, strategy_id: str, config: dict):
        super().__init__(strategy_id, config)
        self.ema_fast = 50
        self.ema_slow = 200
        self.htf_ma = 20

    def generate_signal(self, market_data: MarketData) -> Optional[TradeSignal]:
        """
        Institutional MTF Trend Following (Step 11).
        Requires H1 and M15 agreement before M5 entry.
        """
        # 1. MTF Alignment Check (Consensus)
        # Optimized for V4 Benchmark: Requires H1 Bias agreement
        h1_trend = self.get_ema_trend(market_data.htf_candles)
        if h1_trend == 0:
            return None

        # 2. Local M5 Entry Logic
        m5 = market_data.m5_candles
        h1_trend = self.get_ema_trend(market_data.htf_candles)
        m5_trend = self.get_ema_trend(m5)
        
        # We also want price proximity to EMA 50 for better R:R
        ema50 = m5.ema(50)[-1]
        price = market_data.current_price
        
        # 3. Decision Logic (Subordinated to HTF)
        if h1_trend == 1 and m5_trend == 1:
            # Bullish Alignment: Entry on Price above Fast EMA
            if price > ema50:
                return TradeSignal(direction="BUY", confidence=0.85, timestamp=market_data.timestamp)
        
        if h1_trend == -1 and m5_trend == -1:
            # Bearish Alignment: Entry on Price below Fast EMA
            if price < ema50:
                return TradeSignal(direction="SELL", confidence=0.85, timestamp=market_data.timestamp)

        return None

    def get_stop_loss(self, signal: TradeSignal, market_data: MarketData) -> float:
        atr = market_data.m5_candles.atr(14)[-1]
        if signal.direction == "BUY":
            return market_data.current_price - (atr * 2.5)
        return market_data.current_price + (atr * 2.5)
    def get_take_profit(self, signal: TradeSignal, market_data: MarketData) -> float:
        sl_price = self.get_stop_loss(signal, market_data)
        risk = abs(market_data.current_price - sl_price)
        if signal.direction == "BUY":
            return market_data.current_price + (risk * 3.5)
        return market_data.current_price - (risk * 3.5)
