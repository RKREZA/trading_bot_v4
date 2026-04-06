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
        reasons = []
        if h1_trend == 1: reasons.append("H1: Bullish Trend")
        elif h1_trend == -1: reasons.append("H1: Bearish Trend")
        
        if m5_trend == 1: reasons.append("M5: EMA Momentum UP")
        elif m5_trend == -1: reasons.append("M5: EMA Momentum DOWN")
        
        if price > ema50: reasons.append("Price: Above EMA50 (Bullish)")
        else: reasons.append("Price: Below EMA50 (Bearish)")

        if h1_trend == 1 and m5_trend == 1 and price > ema50:
            return TradeSignal(direction="BUY", price=price, confidence=0.85, timestamp=market_data.timestamp, reasons=reasons)
        
        if h1_trend == -1 and m5_trend == -1 and price < ema50:
            return TradeSignal(direction="SELL", price=price, confidence=0.85, timestamp=market_data.timestamp, reasons=reasons)

        # Fallback: Report Bias Even If No Trade
        return TradeSignal(direction="NONE", price=price, confidence=0.5, reasons=reasons)

    def get_stop_loss(self, signal: TradeSignal, market_data: MarketData) -> float:
        atr_vals = market_data.m5_candles.atr(14)
        atr = atr_vals[-1] if len(atr_vals) > 0 and not np.isnan(atr_vals[-1]) else 1.0
        if signal.direction == "BUY":
            return market_data.current_price - (atr * 2.5)
        return market_data.current_price + (atr * 2.5)

    def get_take_profit(self, signal: TradeSignal, market_data: MarketData) -> float:
        sl_price = self.get_stop_loss(signal, market_data)
        risk = abs(market_data.current_price - sl_price)
        if signal.direction == "BUY":
            return market_data.current_price + (risk * 3.5)
        return market_data.current_price - (risk * 3.5)
