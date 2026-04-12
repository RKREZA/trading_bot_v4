import numpy as np
import logging
from typing import Optional, Dict, Any
from core.base_strategy import BaseStrategy, MarketData
from core.common.types import TradeSignal
from core.session_detector import SessionDetector

logger = logging.getLogger("trading_bot.strategy.trend_v5")

class TrendFollowingStrategy(BaseStrategy):
    """
    V5-MODERN Institutional Trend Following.
    Logic: 
    1. Triple EMA Cloud Alignment (50 > 100 > 200 for Long).
    2. SuperTrend Confirmation.
    3. ADX Momentum Gating (> 25).
    4. ATR Trailing Stop (Chandelier Exit) management.
    """

    def __init__(self, strategy_id: str, config: dict):
        super().__init__(strategy_id, config)
        
        strat_config = self.get_strat_config()
        
        # Trend Parameters
        self.ema_fast_period = 50
        self.ema_mid_period = 100
        self.ema_slow_period = 200
        
        self.adx_threshold = strat_config.get("adx_threshold", 25)
        self.min_confidence = float(strat_config.get("min_confidence", 0.65))
        
        # Trailing Stop Parameters
        self.sl_atr_mult = strat_config.get("sl_atr", 3.0) # Standard Chandelier
        self.tp_atr_mult = strat_config.get("tp_atr", 6.0) # "Blue Sky" Target
        
        self.min_bars_between_signals = strat_config.get("min_bars_between_signals", 20)
        self._last_signal_bar = 0

        # Session-Specific Parameter Overrides (NY Alpha Optimization)
        self.session_adjustments = {
            "LONDON": {"adx_offset": 0, "sl_mult": 0, "conf_floor": 0.65},
            "NEW_YORK": {"adx_offset": -5, "sl_mult": 0.7, "conf_floor": 0.60},
            "LONDON/NY": {"adx_offset": -2, "sl_mult": 0.3, "conf_floor": 0.62}
        }

    def generate_signal(self, market_data: MarketData) -> Optional[TradeSignal]:
        strat_config = self.get_strat_config()
        allowed_sessions = strat_config.get("allowed_sessions", ["LONDON", "NEW_YORK", "LONDON/NY"])
        
        # 1. Session Gating
        if not SessionDetector.is_session_active(market_data.timestamp, allowed_sessions=allowed_sessions):
            self.last_rejection_reason = f"Out of Session ({market_data.session})"
            return None

        m15 = market_data.m15_candles
        if m15 is None or len(m15) < 200: # Need enough for EMA 200 warmup
            return None
        
        # 2. Cooldown Gating
        bars_since_last = len(m15) - self._last_signal_bar
        if bars_since_last < self.min_bars_between_signals:
            self.last_rejection_reason = "Signal cooldown active"
            return None

        # 3. Spread & Liquidity Gates
        if not self.is_spread_safe(market_data):
            return None
            
        # 4. Triple EMA Cloud Alignment
        ema50 = m15.get_indicator("ema_50")
        ema100 = m15.get_indicator("ema_100")
        ema200 = m15.get_indicator("ema_200")
        
        if len(ema200) < 5: return None
        
        is_long_cloud = ema50[-1] > ema100[-1] > ema200[-1]
        is_short_cloud = ema50[-1] < ema100[-1] < ema200[-1]
        
        # 5. SuperTrend Detection
        st_val = m15.get_indicator("supertrend_val")
        st_dir = m15.get_indicator("supertrend_dir")
        
        if len(st_dir) < 5: return None
        
        # 6. Session Adjustment Resolution
        adj = self.session_adjustments.get(market_data.session, {"adx_offset": 0, "sl_mult": 0, "conf_floor": self.min_confidence})
        effective_adx_threshold = self.adx_threshold + adj["adx_offset"]
        
        # 7. ADX Momentum Filter
        adx_vals = m15.get_indicator("adx_14")
        adx = adx_vals[-1]
        adx_slope = adx - adx_vals[-5] if len(adx_vals) >= 5 else 0
        
        if adx < effective_adx_threshold:
            self.last_rejection_reason = f"ADX too low ({adx:.1f} < {effective_adx_threshold})"
            return None
        
        if adx_slope < -2.0: # Momentum is dying
            self.last_rejection_reason = "ADX momentum stalling"
            return None

        # 7. Directional Synthesis
        direction = None
        if is_long_cloud and st_dir[-1] == 1:
            direction = "BUY"
        elif is_short_cloud and st_dir[-1] == -1:
            direction = "SELL"
            
        if not direction:
            self.last_rejection_reason = "Indicators mismatch (Cloud vs ST)"
            return None
            
        # 8. RSI Overextension Protection
        rsi_vals = m15.get_indicator("rsi_14")
        if len(rsi_vals) > 0:
            rsi = rsi_vals[-1]
            if direction == "BUY" and rsi > 75:
                self.last_rejection_reason = f"Overbought (RSI {rsi:.1f})"
                return None
            if direction == "SELL" and rsi < 25:
                self.last_rejection_reason = f"Oversold (RSI {rsi:.1f})"
                return None

        # 9. SIGNAL EMISSION
        confidence = 0.70 + (min(20, (adx - 20)) / 100.0) # Boost confidence for high ADX
        
        conf_floor = adj["conf_floor"]
        if confidence < conf_floor:
            self.last_rejection_reason = f"Confidence {confidence:.2f} < {conf_floor:.2f} (Session Gated)"
            return None

        self._last_signal_bar = len(m15)
        return TradeSignal(direction=direction, confidence=confidence, price=market_data.current_price)

    def get_stop_loss(self, signal: TradeSignal, market_data: MarketData) -> float:
        """
        Institutional Trailing Stop: Uses the SuperTrend line or ATR offset.
        """
        m15 = market_data.m15_candles
        st_val = m15.get_indicator("supertrend_val")[-1]
        atr = m15.get_indicator("atr_14")[-1]
        
        adj = self.session_adjustments.get(market_data.session, {"adx_offset": 0, "sl_mult": 0, "conf_floor": self.min_confidence})
        effective_sl_mult = self.sl_atr_mult + adj["sl_mult"]
        
        # Use SuperTrend as primary, fallback to ATR buffer
        if signal.direction == "BUY":
            return max(st_val, market_data.current_price - (atr * effective_sl_mult))
        else:
            return min(st_val, market_data.current_price + (atr * effective_sl_mult))

    def get_take_profit(self, signal: TradeSignal, market_data: MarketData) -> float:
        """
        "Blue Sky" Trend Target: 6x ATR or ride the trail.
        """
        atr = market_data.m15_candles.get_indicator("atr_14")[-1]
        if signal.direction == "BUY":
            return market_data.current_price + (atr * self.tp_atr_mult)
        else:
            return market_data.current_price - (atr * self.tp_atr_mult)

    def get_metrics(self, market_data: MarketData) -> Dict[str, Any]:
        m15 = market_data.m15_candles
        if m15 is None or len(m15) < 14: return {}
        return {
            "ADX": m15.get_indicator("adx_14")[-1],
            "ST": "Long" if m15.get_indicator("supertrend_dir")[-1] == 1 else "Short"
        }