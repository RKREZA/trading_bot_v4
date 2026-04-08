import numpy as np
import logging
from typing import Optional, Dict, Any
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
        self.adx_period = 14
        self.adx_threshold = 25 # Relaxed from 32
        self.vol_exclusion_mult = 3.0 # Relaxed from 2.0
        self.proximity_atr_mult = 2.0 # Relaxed from 1.5

    def generate_signal(self, market_data: MarketData) -> Optional[TradeSignal]:
        """
        Institutional MTF Trend Following (Step 11).
        Requires H1 and M15 agreement before M5 entry.
        """
        # 1. MTF Alignment Check (Consensus)
        h1_trend = self.get_ema_trend(market_data.htf_candles)
        m15_trend = self.get_ema_trend(market_data.m15_candles)
        if h1_trend == 0:
            self.last_rejection_reason = "Trend: H1 Neutral"
            return None 
        if m15_trend != h1_trend:
            self.last_rejection_reason = "Trend: M15 non-consensus"
            return None

        # 2. Local M5 Entry Logic
        m5 = market_data.m5_candles
        h1_trend = self.get_ema_trend(market_data.htf_candles)
        m5_trend = self.get_ema_trend(m5)
        
        # 3. ADX Intensity Filter (Phase 2 Hardening)
        adx_vals = m5.adx(self.adx_period)
        if len(adx_vals) == 0:
            self.last_rejection_reason = "Trend: No ADX data"
            return None
        adx = adx_vals[-1]
        if np.isnan(adx) or adx < self.adx_threshold:
            self.last_rejection_reason = f"Trend: ADX too weak ({adx:.1f} < {self.adx_threshold})"
            return None 
        
        # 4. Price Proximity & Volatility Guard (Phase 2 Hardening)
        ema50 = m5.ema(self.ema_fast)[-1]
        atr_vals = m5.atr(14)
        atr = atr_vals[-1]
        avg_atr = np.mean(atr_vals[-14:])
        price = market_data.current_price
        
        # Volatility Spike Check: Don't enter if market is "exploding" (Mean Reversion Risk)
        if atr > (avg_atr * self.vol_exclusion_mult):
            self.last_rejection_reason = f"Trend: Volatility Spike ({atr/avg_atr:.1f}x)"
            return None

        # Distance from EMA: Don't chase
        dist_from_ema = abs(price - ema50)
        if dist_from_ema > (atr * self.proximity_atr_mult):
            self.last_rejection_reason = f"Trend: Overextended ({dist_from_ema/atr:.1f} ATR)"
            return None 
            
        # 5. Decision Logic (Subordinated to HTF)
        reasons = [f"ADX: {adx:.1f}"]
        if h1_trend == 1 and m5_trend == 1 and price > ema50:
            return TradeSignal(direction="BUY", price=price, confidence=0.85, timestamp=market_data.timestamp, reasons=reasons)
        
        if h1_trend == -1 and m5_trend == -1 and price < ema50:
            return TradeSignal(direction="SELL", price=price, confidence=0.85, timestamp=market_data.timestamp, reasons=reasons)
        
        self.last_rejection_reason = "Trend: M5 non-alignment"
        return None

    def get_metrics(self, market_data: MarketData) -> Dict[str, Any]:
        m5 = market_data.m5_candles
        adx_vals = m5.adx(self.adx_period)
        adx = adx_vals[-1] if len(adx_vals) > 0 else 0
        ema_vals = m5.ema(self.ema_fast)
        ema50 = ema_vals[-1] if len(ema_vals) > 0 else 0
        atr_vals = m5.atr(14)
        if len(atr_vals) < 14 or ema50 == 0: return {}
        
        atr = atr_vals[-1]
        avg_atr = np.mean(atr_vals[-14:])
        price = market_data.current_price
        dist_from_ema = abs(price - ema50)
        
        return {
            "ADX": adx,
            "Vol Spike": atr/avg_atr if avg_atr > 0 else 0,
            "EMA Dist": dist_from_ema/atr if atr > 0 else 0
        }

    def get_thresholds(self) -> Dict[str, Any]:
        return {
            "ADX": f"> {self.adx_threshold}",
            "Vol Spike": f"< {self.vol_exclusion_mult}x",
            "EMA Dist": f"< {self.proximity_atr_mult}x"
        }

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
