import numpy as np
import logging
from typing import Optional, Dict, Any, List, Tuple
from core.base_strategy import BaseStrategy, MarketData
from core.common.types import TradeSignal
from core.session_detector import SessionDetector

logger = logging.getLogger("trading_bot.strategy.trend_smc")

class TrendFollowingStrategy(BaseStrategy):
    """
    V5-SMC Institutional Trend Following (Orderflow Edition).
    Pipeline:
    1. Session Kill Zone Filter
    2. Value Alignment (VWAP + Rolling POC)
    3. Structural & Liquidity Gating
    4. Displacement & Validated FVG Detection
    5. Mitigation Entry
    """

    def __init__(self, strategy_id: str, config: dict):
        super().__init__(strategy_id, config)
        
        strat_config = self.get_strat_config()
        
        # Kill Zones
        self.allowed_sessions = strat_config.get("allowed_sessions", ["LONDON", "NEW_YORK"])
        
        # SMC / WFO Exposed Parameters
        self.fvg_min_size_atr = float(strat_config.get("fvg_min_size", 0.5))
        self.poc_window = int(strat_config.get("poc_window", 100))
        self.displacement_threshold = float(strat_config.get("displacement_threshold", 1.5))
        self.volume_spike_factor = float(strat_config.get("volume_spike_factor", 1.5))
        
        self.min_confidence = float(strat_config.get("min_confidence", 0.65))
        self._last_signal_bar = 0

    def get_parameter_grid(self) -> Dict[str, List[Any]]:
        return {
            "fvg_min_size": [0.3, 0.5, 1.0],
            "poc_window": [50, 100, 150],
            "displacement_threshold": [1.2, 1.5, 2.0],
            "volume_spike_factor": [1.2, 1.5, 2.0]
        }

    def _get_swings(self, highs: np.ndarray, lows: np.ndarray, window: int = 5) -> Tuple[List[float], List[float]]:
        swing_highs = []
        swing_lows = []
        if len(highs) < window * 2 + 1:
            return swing_highs, swing_lows
            
        for i in range(window, len(highs) - window):
            # Swing High
            if all(highs[i] > highs[i-window:i]) and all(highs[i] > highs[i+1:i+window+1]):
                swing_highs.append(highs[i])
            # Swing Low
            if all(lows[i] < lows[i-window:i]) and all(lows[i] < lows[i+1:i+window+1]):
                swing_lows.append(lows[i])
                
        return swing_highs, swing_lows

    def _calculate_poc(self, data: np.ndarray, vols: np.ndarray, bins: int = 50) -> float:
        if len(data) == 0: return 0.0
        min_p, max_p = np.min(data), np.max(data)
        if min_p == max_p: return min_p
        
        step = (max_p - min_p) / bins
        profile = np.zeros(bins)
        
        for i in range(len(data)):
            idx = int((data[i] - min_p) / step)
            if idx >= bins: idx = bins - 1
            if idx < 0: idx = 0
            profile[idx] += vols[i]
            
        poc_idx = np.argmax(profile)
        return min_p + (poc_idx * step) + (step / 2)

    def generate_signal(self, market_data: MarketData) -> Optional[TradeSignal]:
        # 1. Session Kill Zone Filter
        if not SessionDetector.is_session_active(market_data.timestamp, allowed_sessions=self.allowed_sessions):
            self.last_rejection_reason = f"Out of Kill Zone ({market_data.session})"
            return None

        m15 = market_data.m15_candles
        if m15 is None or len(m15) < max(self.poc_window, 50):
            return None

        # Cooldown
        if len(m15) - self._last_signal_bar < 5:
            return None

        # Data extraction
        current_price = market_data.current_price
        o = m15.o
        c = m15.c
        h = m15.h
        l = m15.l
        v = m15.v
        
        atr_14 = m15.atr(14)
        if len(atr_14) == 0 or np.isnan(atr_14[-1]): return None
        current_atr = atr_14[-1]

        # 2. Value Alignment (Rolling VWAP & POC)
        window = self.poc_window
        recent_c, recent_h, recent_l, recent_v = c[-window:], h[-window:], l[-window:], v[-window:]
        typical_price = (recent_h + recent_l + recent_c) / 3.0
        
        vwap = np.sum(typical_price * recent_v) / (np.sum(recent_v) + 1e-9)
        poc = self._calculate_poc(typical_price, recent_v, bins=50)

        # Macro Bias Setup (Relaxed for Pullback Entry)
        # We check if VWAP is pointing up or Price is generally above VWAP.
        # Strict dual-alignment (VWAP + POC) often gates valid deep pullbacks.
        is_bullish_macro = current_price > vwap 
        is_bearish_macro = current_price < vwap 

        if not is_bullish_macro and not is_bearish_macro:
            self.last_rejection_reason = "No Value Alignment (Price inside VWAP)"
            return None

        # 3. Institutional Imbalance Candle (IC) Detection
        # Instead of strict missing wicks, we search for Massive Body Displacement.
        ic_lookback = 15
        if len(c) < ic_lookback + 2: return None
        
        valid_ic_idx = -1
        ic_type = 0 # 1 Bullish, -1 Bearish
        target_discount_low = 0.0
        target_discount_high = 0.0
        
        for i in range(len(c) - ic_lookback, len(c) - 1): # i is the displacement candle
            body_size = abs(o[i] - c[i])
            candle_size = h[i] - l[i]
            avg_vol = np.mean(v[i-10:i]) if i >= 10 else np.mean(v[:i])
            
            # Using configured FVG params to control Institutional displacement scale
            is_large_body = body_size > current_atr * self.fvg_min_size_atr
            is_large_candle = candle_size > current_atr * self.displacement_threshold
            has_volume_spike = v[i] > avg_vol * self.volume_spike_factor
            
            if is_large_body and is_large_candle and has_volume_spike:
                # Discovered an Institutional Footprint
                valid_ic_idx = i
                
                # Identify if Bullish or Bearish Displacement
                if c[i] > o[i]:
                    ic_type = 1
                    # Bullish Fiber Zone (Discount overlap within the candle and slightly below origin)
                    target_discount_low = l[i] - (current_atr * 0.2)
                    target_discount_high = l[i] + (body_size * 0.6) # From origin up to 60% of the body
                else:
                    ic_type = -1
                    # Bearish Premium Zone
                    target_discount_low = h[i] - (body_size * 0.6)
                    target_discount_high = h[i] + (current_atr * 0.2)
                
        if valid_ic_idx == -1:
            self.last_rejection_reason = "No Institutional Displacement (IC) found"
            return None

        # 4. Mitigation Entry & Context Validations
        signal = None
        
        if ic_type == 1 and is_bullish_macro:
            # We want price to pull back INTO the Discount Fiber Zone
            if current_price >= target_discount_low and current_price <= target_discount_high:
                signal = "BUY"
                
        elif ic_type == -1 and is_bearish_macro:
            if current_price >= target_discount_low and current_price <= target_discount_high:
                signal = "SELL"

        if not signal:
            self.last_rejection_reason = "IC exists but price not in Discount/Premium entry zone"
            return None

        # 5. Structure & Swing Extraction for TP/SL mapping
        swing_highs, swing_lows = self._get_swings(recent_h, recent_l, window=5)
        
        # Stop loss logic (Below FVG Origin or Recent Swing)
        origin_candle_low = l[valid_ic_idx]
        origin_candle_high = h[valid_ic_idx]
        
        if signal == "BUY":
            # Primary: Below displacement candle low
            sl = origin_candle_low - (current_atr * 0.2)
            # Secondary check: Nearest swing low
            if swing_lows:
                nearest_sl = min([sl for sl in swing_lows if sl < current_price], default=sl)
                sl = min(sl, nearest_sl - (current_atr * 0.1))
                
            # TP: Nearest major swing high
            tp = max([sh for sh in swing_highs if sh > current_price], default=current_price + (current_atr * 3.0))
            
        else:
            sl = origin_candle_high + (current_atr * 0.2)
            if swing_highs:
                nearest_sh = max([sh for sh in swing_highs if sh > current_price], default=sl)
                sl = max(sl, nearest_sh + (current_atr * 0.1))
                
            tp = min([sl_ for sl_ in swing_lows if sl_ < current_price], default=current_price - (current_atr * 3.0))

        # Risk-to-Reward Safety Hatch
        risk = abs(current_price - sl)
        reward = abs(tp - current_price)
        # Use a more tolerant R:R for high-probability SMC setups if needed, 
        # but defaulting to BaseStrategy.min_rr (usually 2.0)
        actual_rr = reward / risk if risk > 0 else 0
        if risk == 0 or actual_rr < self.min_rr:
            self.last_rejection_reason = f"Poor R:R structural mapping ({actual_rr:.2f} < {self.min_rr})"
            return None

        ts = TradeSignal(
            direction=signal, 
            confidence=0.85, 
            price=current_price,
            stop_loss=sl,
            take_profit=tp
        )
        return ts

    def get_stop_loss(self, signal: TradeSignal, market_data: MarketData) -> float:
        # Pre-calculated in generate_signal and passed via TradeSignal object in the ecosystem
        # Providing a fallback just in case
        return signal.stop_loss if signal.stop_loss > 0 else (market_data.current_price * 0.99)

    def get_take_profit(self, signal: TradeSignal, market_data: MarketData) -> float:
        return signal.take_profit if signal.take_profit > 0 else (market_data.current_price * 1.01)

    def get_metrics(self, market_data: MarketData) -> Dict[str, Any]:
        m15 = market_data.m15_candles
        if m15 is None or len(m15) < self.poc_window: return {}
        
        # Re-calc for dashboard live telemetry
        window = self.poc_window
        recent_c, recent_h, recent_l, recent_v = m15.c[-window:], m15.h[-window:], m15.l[-window:], m15.v[-window:]
        typical_price = (recent_h + recent_l + recent_c) / 3.0
        
        vwap = np.sum(typical_price * recent_v) / (np.sum(recent_v) + 1e-9)
        poc = self._calculate_poc(typical_price, recent_v, bins=50)
        
        return {
            "SMC_VWAP": round(vwap, 3),
            "SMC_POC": round(poc, 3),
            "Bias": "Bullish" if market_data.current_price > vwap and market_data.current_price > poc else ("Bearish" if market_data.current_price < vwap and market_data.current_price < poc else "Neutral")
        }