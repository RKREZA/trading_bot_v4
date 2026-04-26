import numpy as np
import logging
from typing import Optional, Dict, Any, List, Tuple
from core.base_strategy import BaseStrategy, MarketData
from core.common.types import TradeSignal
from core.session_detector import SessionDetector

logger = logging.getLogger("trading_bot.strategy.trend_smc")

class TrendFollowingStrategy(BaseStrategy):
    """
    V6-INSIGNIA Institutional Trend Following (Hardened Edition).
    
    Key fixes over V5:
    - Lowered ADX threshold (25→18) to capture Gold's structural trends
    - Added Multi-Timeframe EMA consensus (H1 EMA50/200 alignment)
    - Reduced displacement requirements for more trade opportunities
    - Dynamic confidence scoring based on confluence factors
    - Fibonacci retracement entry zone (50-78.6% of IC body)
    - Adaptive SL/TP using swing structure + ATR scaling
    
    Pipeline:
    1. Session Kill Zone Filter
    2. Multi-Timeframe Trend Consensus (H1 EMA + VWAP)
    3. ADX Trend Strength Gate (relaxed)
    4. Institutional Displacement Detection
    5. Fibonacci Mitigation Entry
    6. Dynamic Confidence Scoring
    """

    def __init__(self, strategy_id: str, config: dict):
        super().__init__(strategy_id, config)
        
        strat_config = self.get_strat_config()
        
        self.allowed_sessions = strat_config.get("allowed_sessions", ["LONDON", "NEW_YORK", "LONDON/NY"])
        
        # SMC / WFO Exposed Parameters (V6 Hardened Values)
        self.fvg_min_size_atr = float(strat_config.get("fvg_min_size", 0.3))       # Reduced from 0.5
        self.poc_window = int(strat_config.get("poc_window", 100))
        self.displacement_threshold = float(strat_config.get("displacement_threshold", 0.9))  # Catch real institutional moves
        self.volume_spike_factor = float(strat_config.get("volume_spike_factor", 1.5))  # Demand high-quality volume
        self.ic_lookback = int(strat_config.get("ic_lookback", 20))       
        
        self.min_confidence = float(strat_config.get("min_confidence", 0.70))  
        self.adx_min_threshold = float(strat_config.get("adx_min_threshold", 25.0))  # High-quality trends only
        self.min_bars_between_signals = int(strat_config.get("min_bars_between_signals", 10))  # Reduced from 20
        self._last_signal_bar = 0

    def get_parameter_grid(self) -> Dict[str, List[Any]]:
        return {
            "fvg_min_size": [0.2, 0.3, 0.5],
            "poc_window": [50, 100, 150],
            "displacement_threshold": [0.5, 0.7, 1.0],
            "adx_min_threshold": [15.0, 18.0, 22.0]
        }

    def _get_swings(self, highs: np.ndarray, lows: np.ndarray, window: int = 5) -> Tuple[List[float], List[float]]:
        swing_highs = []
        swing_lows = []
        if len(highs) < window * 2 + 1:
            return swing_highs, swing_lows
            
        for i in range(window, len(highs) - window):
            if all(highs[i] > highs[i-window:i]) and all(highs[i] > highs[i+1:i+window+1]):
                swing_highs.append(highs[i])
            if all(lows[i] < lows[i-window:i]) and all(lows[i] < lows[i+1:i+window+1]):
                swing_lows.append(lows[i])
                
        return swing_highs, swing_lows

    def _calculate_poc(self, data: np.ndarray, vols: np.ndarray, bins: int = 50) -> float:
        """Vectorized POC calculation using NumPy histograms."""
        if len(data) == 0: return 0.0
        min_p, max_p = np.min(data), np.max(data)
        if min_p == max_p: return min_p
        
        counts, bin_edges = np.histogram(data, bins=bins, weights=vols, range=(min_p, max_p))
        poc_idx = np.argmax(counts)
        return (bin_edges[poc_idx] + bin_edges[poc_idx+1]) / 2

    def _get_mtf_trend(self, market_data: MarketData) -> int:
        """
        V6 Multi-Timeframe Trend Consensus.
        Returns: 1 (Bullish), -1 (Bearish), 0 (No consensus)
        
        Requires H1 EMA(50) > EMA(200) for bullish bias, and vice versa.
        """
        h1 = market_data.htf_candles
        ema_50 = h1.get_indicator("ema_50")
        ema_200 = h1.get_indicator("ema_200")
        
        if len(ema_50) < 2 or len(ema_200) < 2:
            return 0
        if np.isnan(ema_50[-1]) or np.isnan(ema_200[-1]):
            return 0
            
        # Primary: EMA stack alignment
        if ema_50[-1] > ema_200[-1]:
            return 1  # Bullish
        elif ema_50[-1] < ema_200[-1]:
            return -1  # Bearish
        return 0

    def _calculate_dynamic_confidence(self, adx_val: float, mtf_trend: int, 
                                       ic_type: int, has_vwap_align: bool,
                                       volume_ratio: float) -> float:
        """
        V6 Dynamic Confidence Scoring.
        Replaces hardcoded 0.85 with a multi-factor score.
        """
        score = 0.55  # Base
        
        # ADX strength contribution (0 to +0.15)
        if adx_val >= 30:
            score += 0.15
        elif adx_val >= 22:
            score += 0.10
        elif adx_val >= 18:
            score += 0.05
            
        # MTF consensus (+0.10)
        if (mtf_trend == 1 and ic_type == 1) or (mtf_trend == -1 and ic_type == -1):
            score += 0.10
            
        # VWAP alignment (+0.05)
        if has_vwap_align:
            score += 0.05
            
        # Volume confirmation (+0.05)
        if volume_ratio > 1.5:
            score += 0.05
        elif volume_ratio > 1.0:
            score += 0.02
            
        return min(0.95, score)

    def generate_signal(self, market_data: MarketData) -> Optional[TradeSignal]:
        # 0. Cooldown
        m15 = market_data.m15_candles
        if m15 is None or len(m15) < max(self.poc_window, 50):
            return None
            
        if len(m15) - self._last_signal_bar < self.min_bars_between_signals:
            self.last_rejection_reason = "Signal cooldown active"
            return None

        # 1. Session Kill Zone Filter
        if not SessionDetector.is_session_active(market_data.timestamp, allowed_sessions=self.allowed_sessions):
            self.last_rejection_reason = f"Out of Kill Zone ({market_data.session})"
            return None

        # 2. Multi-Timeframe Trend Consensus (V6 NEW)
        mtf_trend = self._get_mtf_trend(market_data)
        if mtf_trend == 0:
            self.last_rejection_reason = "No MTF Consensus (H1 EMA50/200 flat)"
            return None

        # 3. ADX Trend Strength Gate (Relaxed from 25 to 18)
        adx_14 = market_data.m15_candles.get_indicator("adx_14")
        current_adx = adx_14[-1] if len(adx_14) > 0 else 0
        if current_adx < self.adx_min_threshold:
            self.last_rejection_reason = f"Trend Weak (ADX {current_adx:.1f} < {self.adx_min_threshold})"
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

        # 4. VWAP Value Alignment (supplementary, not primary filter)
        window = self.poc_window
        recent_c, recent_h, recent_l, recent_v = c[-window:], h[-window:], l[-window:], v[-window:]
        typical_price = (recent_h + recent_l + recent_c) / 3.0
        vwap = np.sum(typical_price * recent_v) / (np.sum(recent_v) + 1e-9)
        
        has_vwap_align = (mtf_trend == 1 and current_price > vwap) or \
                         (mtf_trend == -1 and current_price < vwap)

        # 5. Institutional Displacement Detection (Relaxed thresholds)
        ic_lookback = min(self.ic_lookback, len(c) - 2)
        if len(c) < ic_lookback + 2: return None
        
        valid_ic_idx = -1
        ic_type = 0
        best_volume_ratio = 0.0
        
        for i in range(len(c) - ic_lookback, len(c) - 1):
            body_size = abs(o[i] - c[i])
            candle_size = h[i] - l[i]
            avg_vol = np.mean(v[max(0, i-10):i]) if i >= 1 else v[0]
            
            is_large_body = body_size > current_atr * self.fvg_min_size_atr
            is_large_candle = candle_size > current_atr * self.displacement_threshold
            volume_ratio = v[i] / (avg_vol + 1e-9)
            has_volume_spike = volume_ratio >= self.volume_spike_factor
            
            if is_large_body and is_large_candle and has_volume_spike:
                valid_ic_idx = i
                best_volume_ratio = volume_ratio
                ic_type = 1 if c[i] > o[i] else -1
                
        if valid_ic_idx == -1:
            self.last_rejection_reason = "No Institutional Displacement (IC) found"
            return None

        # V6: Reject if IC direction conflicts with MTF trend
        if ic_type != mtf_trend:
            self.last_rejection_reason = f"IC direction ({ic_type}) conflicts with MTF trend ({mtf_trend})"
            return None

        # 6. Entry Zone (V6.1: Widened range for Gold's high-volatility M15 candles)
        # Entry zone = 0% to 100% retracement of the IC body (within full body range)
        # Plus ATR-based proximity as fallback
        ic_open = o[valid_ic_idx]
        ic_close = c[valid_ic_idx]
        ic_body_size = abs(ic_close - ic_open)
        
        if ic_type == 1:  # Bullish IC
            # Price should be anywhere within the IC body range or near it
            zone_high = ic_close + (current_atr * 0.5)  # Allow slightly above IC close
            zone_low = ic_open - (current_atr * 0.3)    # Allow slightly below IC open
            in_entry_zone = zone_low <= current_price <= zone_high
        else:  # Bearish IC
            zone_high = ic_open + (current_atr * 0.3)
            zone_low = ic_close - (current_atr * 0.5)
            in_entry_zone = zone_low <= current_price <= zone_high

        if not in_entry_zone:
            # Fallback: allow entry within 1.0 ATR of the IC midpoint
            ic_mid = (ic_open + ic_close) / 2
            if abs(current_price - ic_mid) <= current_atr * 1.0:
                in_entry_zone = True
            else:
                self.last_rejection_reason = "Price not in Fibonacci entry zone"
                return None

        # 7. Dynamic Confidence Scoring (V6: replaces hardcoded 0.85)
        confidence = self._calculate_dynamic_confidence(
            adx_val=current_adx,
            mtf_trend=mtf_trend,
            ic_type=ic_type,
            has_vwap_align=has_vwap_align,
            volume_ratio=best_volume_ratio
        )

        # 8. Direction assignment
        signal_dir = "BUY" if ic_type == 1 else "SELL"

        # 9. SL/TP using Swing Structure + ATR
        min_rr = 3.0
        swing_highs, swing_lows = self._get_swings(recent_h, recent_l, window=5)
        
        origin_low = l[valid_ic_idx]
        origin_high = h[valid_ic_idx]
        
        if signal_dir == "BUY":
            # SL: Structural (1.5 ATR)
            sl = origin_low - (current_atr * 1.5)
            risk = abs(current_price - sl)
            # TP: Reliable 1:2 RR for TrendFollowing
            tp = current_price + (risk * 2.0) 
        else:
            sl = origin_high + (current_atr * 1.5)
            risk = abs(current_price - sl)
            tp = current_price - (risk * 2.0)

        # R:R Safety Check
        risk = abs(current_price - sl)
        reward = abs(tp - current_price)
        rr = reward / risk if risk > 0 else 0
        
        if rr < min_rr:
            # Try extending TP to meet minimum R:R
            if signal_dir == "BUY":
                tp = current_price + (risk * (min_rr + 0.5))
            else:
                tp = current_price - (risk * (min_rr + 0.5))
            reward = abs(tp - current_price)
            rr = reward / risk if risk > 0 else 0
            
            if rr < min_rr:
                self.last_rejection_reason = f"Poor R:R ({rr:.2f} < {min_rr})"
                return None

        self._last_signal_bar = len(m15)
        
        return TradeSignal(
            direction=signal_dir, 
            confidence=confidence, 
            price=current_price,
            stop_loss=sl,
            take_profit=tp,
            reasons=[f"MTF:{mtf_trend}", f"ADX:{current_adx:.1f}", f"VOL:{best_volume_ratio:.1f}x",
                     f"RR:{rr:.2f}", f"VWAP:{'ALIGNED' if has_vwap_align else 'DIVERGENT'}"]
        )

    def get_metrics(self, market_data: MarketData) -> Dict[str, Any]:
        m15 = market_data.m15_candles
        if m15 is None or len(m15) < self.poc_window: return {}
        
        window = self.poc_window
        recent_c, recent_h, recent_l, recent_v = m15.c[-window:], m15.h[-window:], m15.l[-window:], m15.v[-window:]
        typical_price = (recent_h + recent_l + recent_c) / 3.0
        
        vwap = np.sum(typical_price * recent_v) / (np.sum(recent_v) + 1e-9)
        poc = self._calculate_poc(typical_price, recent_v, bins=50)
        mtf = self._get_mtf_trend(market_data)
        
        return {
            "SMC_VWAP": round(vwap, 3),
            "SMC_POC": round(poc, 3),
            "MTF_Trend": "Bullish" if mtf == 1 else ("Bearish" if mtf == -1 else "Neutral"),
            "Bias": "Bullish" if market_data.current_price > vwap and mtf == 1 else (
                "Bearish" if market_data.current_price < vwap and mtf == -1 else "Neutral")
        }