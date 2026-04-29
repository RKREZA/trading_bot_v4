"""
LIQUIDITY PRICE ACTION (LPA) STRATEGY
======================================
Institutional-grade Smart Money Concepts strategy using top-down analysis:
  H1 (Bias via BOS/CHoCH + EMA) → M15 (Zone Discovery) → M1 (Execution)

Zone Detection Tools (M15):
  1. Session VWAP + σ bands
  2. Fixed Range Volume Profile (POC/VAH/VAL)
  3. Turtle Soup (false breakout detection)
  4. Fair Value Gaps (FVG)
  5. Premium/Discount Fibonacci zones

Universal: Works on any symbol (XAU, FX, Oil, etc.)
"""

import numpy as np
import logging
from typing import Optional, Dict, Any, List, Tuple
from core.base_strategy import BaseStrategy, MarketData
from core.common.types import TradeSignal, CandleArray

logger = logging.getLogger("trading_bot.strategy.liquidity_price_action")

# --- Killzone windows (UTC hours) ---
KILLZONES = {
    "LONDON_OPEN": (7, 9),
    "NY_OPEN": (13, 15),
    "LONDON_NY_OVERLAP": (12, 16),
}


class LiquidityPriceActionStrategy(BaseStrategy):
    """
    Institutional LPA Strategy — M1 execution with H1→M15 top-down bias.
    Symbol-agnostic: uses MarketData.point for all price calculations.
    """

    def __init__(self, strategy_id: str, config: dict):
        super().__init__(strategy_id, config)
        cfg = self.get_strat_config()

        # VWAP
        self.vwap_session_bars = cfg.get("vwap_session_bars", 96)
        # Volume Profile
        self.vp_lookback = cfg.get("volume_profile_lookback", 20)
        self.vp_bins = cfg.get("volume_profile_bins", 50)
        # Turtle Soup
        self.ts_lookback = cfg.get("turtle_soup_lookback", 20)
        self.ts_atr_mult = cfg.get("turtle_soup_threshold_atr_mult", 0.3)
        # Zone proximity
        self.zone_prox_mult = cfg.get("zone_proximity_atr_mult", 0.5)
        # Volume spike
        self.vol_spike_factor = cfg.get("volume_spike_factor", 1.5)
        # ADX
        self.adx_threshold = cfg.get("adx_trend_threshold", 25)
        # SL buffer
        self.sl_atr_mult = cfg.get("sl_atr_buffer_mult", 0.3)
        # Min R:R
        self.min_rr = float(cfg.get("min_rr", 2.5))
        # FVG
        self.fvg_min_atr_mult = cfg.get("fvg_min_atr_mult", 0.3)

        self.last_rejection_reason = None

    # =========================================================================
    # MAIN SIGNAL PIPELINE
    # =========================================================================
    def generate_signal(self, market_data: MarketData) -> Optional[TradeSignal]:
        h1 = market_data.htf_candles
        m15 = market_data.m15_candles
        m1 = market_data.m1_candles

        if h1 is None or m15 is None or m1 is None:
            self.last_rejection_reason = "Missing MTF data (H1/M15/M1)"
            return None
        if len(h1) < 50 or len(m15) < 50 or len(m1) < 21:
            self.last_rejection_reason = "Insufficient history"
            return None

        price = market_data.bid if market_data.bid > 0 else market_data.current_price
        point = market_data.point or 0.01
        session = market_data.session

        # --- Session gate ---
        allowed_sessions = self.get_strat_config().get("allowed_sessions", ["TOKYO", "LONDON", "NEW_YORK", "LONDON/NY"])
        if session not in allowed_sessions:
            self.last_rejection_reason = f"Session gated: {session}"
            return None

        if not self.is_spread_safe(market_data):
            return None

        # --- Regime gate ---
        regime_allowed = market_data.regime in ["TRENDING", "STABLE", "LIQUIDITY_EVENT"]
        is_high_conf_regime = market_data.regime in ["RANGING", "TRANSITION", "EXPANSION"]
        
        # ATRs
        atr_h1 = h1.atr(14)
        atr_m15 = m15.atr(14)
        atr_m1 = m1.atr(14)
        if len(atr_h1) < 2 or len(atr_m15) < 2 or len(atr_m1) < 2:
            return None
            
        cur_atr_h1 = atr_h1[-1]
        cur_atr_m15 = atr_m15[-1]
        cur_atr_m1 = atr_m1[-1]
        
        if np.isnan(cur_atr_m15) or cur_atr_m15 == 0:
            return None

        # --- Dynamic Parameter Resolution ---
        cfg = self.get_strat_config(session)
        min_conf = float(cfg.get("min_confidence", self.min_confidence))
        min_rr = float(cfg.get("min_rr", self.min_rr))
        adx_thresh = float(cfg.get("adx_trend_threshold", 25))
        zone_prox = float(cfg.get("zone_proximity_atr_mult", 0.5))
        vol_spike = float(cfg.get("volume_spike_factor", 1.5))
        sl_mult = float(cfg.get("sl_atr_buffer_mult", 0.3))

        # =====================================================================
        # STEP 1: H1 BIAS (BOS/CHoCH + EMA 20/50 confirmation)
        # =====================================================================
        bias = self._get_htf_bias(h1)

        if bias == 0:
            self.last_rejection_reason = "No H1 directional bias"
            return None

        # =====================================================================
        # STEP 2: M15 ZONE DISCOVERY
        # =====================================================================
        zones = self._find_zones(m15, cur_atr_m15, bias, price, cfg)
        if not zones:
            self.last_rejection_reason = "No active M15 zones"
            return None

        # Find nearest zone within proximity
        proximity = cur_atr_m15 * (zone_prox * 1.5)  # Relaxed proximity for HFT
        active_zone = None
        for z in zones:
            if abs(price - z["price"]) <= proximity:
                active_zone = z
                break

        if active_zone is None:
            self.last_rejection_reason = "Price not at M15 zone"
            return None

        # =====================================================================
        # STEP 3: M1 HFT CONFIRMATION
        # =====================================================================
        confirmed, pattern = self._detect_m1_confirmation(m1, active_zone, bias)

        if not confirmed:
            self.last_rejection_reason = f"No M1 confirmation at {active_zone['type']}"
            return None

        # =====================================================================
        # STEP 4: KILLZONE CHECK
        # =====================================================================
        kz_bonus = self._is_killzone(market_data)

        # =====================================================================
        # STEP 5: CONFLUENCE SCORING
        # =====================================================================
        adx_series = h1.adx(14)
        adx_val = adx_series[-1] if len(adx_series) > 0 else 0
        adx_prev = adx_series[-2] if len(adx_series) > 1 else 0
        adx_increasing = adx_val > adx_prev
        
        vol_confirmed = self._is_volume_spike(m1, vol_spike)
        
        # RSI Filter (Institutional Reversal Confirmation)
        if m1 is not None and len(m1) >= 14:
            rsi_val = m1.rsi(14)[-1]
            rsi_ext = rsi_val < 25 if bias == 1 else rsi_val > 75
        else:
            rsi_m15 = m15.rsi(14)[-1]
            rsi_ext = rsi_m15 < 35 if bias == 1 else rsi_m15 > 65
        
        confluence = self._calculate_confluence(
            active_zone, vol_confirmed, adx_val, kz_bonus, pattern, adx_thresh
        )
        if adx_increasing: confluence += 0.05
        if rsi_ext: confluence += 0.15
        
        # Regime Scoring Adjustments
        if market_data.regime == "RANGING": confluence -= 0.05
        if market_data.regime == "LIQUIDITY_EVENT": confluence += 0.10
        if market_data.regime == "TRENDING": confluence += 0.05

        if confluence < min_conf:
            self.last_rejection_reason = f"Low confluence {confluence:.2f}"
            return None

        # =====================================================================
        # STEP 6: SL / TP / SIGNAL
        # =====================================================================
        direction = "BUY" if bias == 1 else "SELL"
        sl, tp, rr = self._compute_sl_tp(
            direction, price, active_zone, zones, cur_atr_m15, point, sl_mult, min_rr
        )
        if rr < min_rr:
            self.last_rejection_reason = f"RR {rr:.1f} < {min_rr}"
            return None

        return TradeSignal(
            direction=direction,
            price=price,
            stop_loss=sl,
            take_profit=tp,
            volume=0.0,
            confidence=confluence,
            rr_ratio=rr,
            reasons=[active_zone["type"], pattern],
        )

    # =========================================================================
    # H1 BIAS: BOS/CHoCH + EMA 20/50
    # =========================================================================
    def _get_htf_bias(self, h1: CandleArray) -> int:
        ema20 = h1.ema(20)
        ema50 = h1.ema(50)
        
        if len(ema20) < 2 or len(ema50) < 2:
            return 0
            
        e20, e50 = ema20[-1], ema50[-1]
        if np.isnan(e20) or np.isnan(e50):
            return 0

        # EMA alignment
        ema_bias = 0
        if e20 > e50:
            ema_bias = 1
        elif e20 < e50:
            ema_bias = -1

        # BOS/CHoCH: check H1 swings
        struct_bias = self._detect_h1_structure(h1, lookback=30)

        # High Probability Agreement
        if ema_bias == struct_bias and ema_bias != 0:
            return ema_bias
            
        # Structure overrides MA lag
        if struct_bias != 0:
            return struct_bias
            
        return ema_bias

    def _detect_h1_structure(self, candles: CandleArray, lookback: int = 30) -> int:
        if len(candles) < lookback + 5:
            return 0
        h = candles.h
        l = candles.l
        swing_highs = []
        swing_lows = []
        end = len(h)
        start = max(0, end - lookback)
        for i in range(start + 2, end - 2):
            if h[i] >= h[i - 1] and h[i] >= h[i - 2] and h[i] >= h[i + 1] and h[i] >= h[i + 2]:
                swing_highs.append(h[i])
            if l[i] <= l[i - 1] and l[i] <= l[i - 2] and l[i] <= l[i + 1] and l[i] <= l[i + 2]:
                swing_lows.append(l[i])

        if len(swing_highs) < 2 or len(swing_lows) < 2:
            return 0

        hh = swing_highs[-1] > swing_highs[-2]
        hl = swing_lows[-1] > swing_lows[-2]
        ll = swing_lows[-1] < swing_lows[-2]
        lh = swing_highs[-1] < swing_highs[-2]

        if hh and hl: return 1
        if ll and lh: return -1
        return 0

    # =========================================================================
    # ZONE DISCOVERY (M15)
    # =========================================================================
    def _find_zones(self, m15: CandleArray, atr: float,
                    bias: int, price: float, cfg: Dict) -> List[Dict]:
        zones = []
        vwap_bars = cfg.get("vwap_session_bars", 96)
        zones.extend(self._vwap_zones(m15, atr, bias, vwap_bars))
        
        vp_lookback = cfg.get("volume_profile_lookback", 24)
        vp_bins = cfg.get("volume_profile_bins", 50)
        zones.extend(self._volume_profile_zones(m15, atr, bias, vp_lookback, vp_bins))
        
        ts_lookback = cfg.get("turtle_soup_lookback", 20)
        ts_atr_mult = cfg.get("turtle_soup_threshold_atr_mult", 0.3)
        zones.extend(self._turtle_soup_zones(m15, atr, bias, ts_lookback, ts_atr_mult))
        
        fvg_atr_mult = cfg.get("fvg_min_atr_mult", 0.3)
        zones.extend(self._fvg_zones(m15, atr, bias, fvg_atr_mult))
        
        zones.extend(self._premium_discount_zones(m15, atr, bias))

        if not zones:
            return []

        zones.sort(key=lambda z: abs(price - z["price"]))

        for i, z in enumerate(zones):
            nearby_count = sum(
                1 for oz in zones
                if oz is not z and abs(oz["price"] - z["price"]) < atr * 0.3
            )
            z["confluence_count"] = nearby_count

        return zones

    def _vwap_zones(self, m15: CandleArray, atr: float, bias: int, vwap_session_bars: int) -> List[Dict]:
        zones = []
        if len(m15) < vwap_session_bars:
            return zones
        vwap, u1, l1, u2, l2 = m15.vwap(vwap_session_bars)
        if np.isnan(vwap[-1]):
            return zones

        if bias == 1:  
            zones.append({"price": float(l1[-1]), "type": "VWAP_-1SD", "tool": "vwap", "strength": 0.7})
            zones.append({"price": float(l2[-1]), "type": "VWAP_-2SD", "tool": "vwap", "strength": 0.9})
            zones.append({"price": float(vwap[-1]), "type": "VWAP", "tool": "vwap", "strength": 0.5})
        else:  
            zones.append({"price": float(u1[-1]), "type": "VWAP_+1SD", "tool": "vwap", "strength": 0.7})
            zones.append({"price": float(u2[-1]), "type": "VWAP_+2SD", "tool": "vwap", "strength": 0.9})
            zones.append({"price": float(vwap[-1]), "type": "VWAP", "tool": "vwap", "strength": 0.5})
        return zones

    def _volume_profile_zones(self, m15: CandleArray, atr: float, bias: int, vp_lookback: int, vp_bins: int) -> List[Dict]:
        zones = []
        if len(m15) < vp_lookback:
            return zones
        vp = m15.volume_profile(vp_lookback, vp_bins)
        if np.isnan(vp["poc"]):
            return zones

        zones.append({"price": vp["poc"], "type": "VP_POC", "tool": "volume_profile", "strength": 0.9})
        if bias == 1:
            zones.append({"price": vp["val"], "type": "VP_VAL", "tool": "volume_profile", "strength": 0.8})
        else:
            zones.append({"price": vp["vah"], "type": "VP_VAH", "tool": "volume_profile", "strength": 0.8})
        return zones

    def _turtle_soup_zones(self, m15: CandleArray, atr: float, bias: int, lookback: int, atr_mult: float) -> List[Dict]:
        zones = []
        n = len(m15)
        if n < lookback + 3:
            return zones
        threshold = atr * atr_mult
        recent_h = m15.h[-(lookback + 3):-3]
        recent_l = m15.l[-(lookback + 3):-3]
        swing_high = np.max(recent_h)
        swing_low = np.min(recent_l)

        last_h, last_l, last_c = m15.h[-2], m15.l[-2], m15.c[-2]

        if bias == -1 and last_h > swing_high and last_c < swing_high:
            sweep_dist = last_h - swing_high
            if sweep_dist <= threshold:
                zones.append({
                    "price": float(swing_high),
                    "type": "TURTLE_SOUP_HIGH",
                    "tool": "turtle_soup",
                    "strength": 0.85,
                    "sweep_wick": float(last_h),
                })

        if bias == 1 and last_l < swing_low and last_c > swing_low:
            sweep_dist = swing_low - last_l
            if sweep_dist <= threshold:
                zones.append({
                    "price": float(swing_low),
                    "type": "TURTLE_SOUP_LOW",
                    "tool": "turtle_soup",
                    "strength": 0.85,
                    "sweep_wick": float(last_l),
                })
        return zones

    def _fvg_zones(self, m15: CandleArray, atr: float, bias: int, fvg_min_atr_mult: float) -> List[Dict]:
        zones = []
        n = len(m15)
        if n < 20:
            return zones
        min_gap = atr * fvg_min_atr_mult
        for i in range(max(3, n - 15), n - 1):
            if m15.h[i - 2] < m15.l[i] - min_gap:
                fvg_mid = (m15.h[i - 2] + m15.l[i]) / 2.0
                if bias == 1:
                    zones.append({
                        "price": float(fvg_mid),
                        "type": "FVG_BULL",
                        "tool": "fvg",
                        "strength": 0.75,
                        "fvg_top": float(m15.l[i]),
                        "fvg_bot": float(m15.h[i - 2]),
                    })
            if m15.l[i - 2] > m15.h[i] + min_gap:
                fvg_mid = (m15.l[i - 2] + m15.h[i]) / 2.0
                if bias == -1:
                    zones.append({
                        "price": float(fvg_mid),
                        "type": "FVG_BEAR",
                        "tool": "fvg",
                        "strength": 0.75,
                        "fvg_top": float(m15.l[i - 2]),
                        "fvg_bot": float(m15.h[i]),
                    })
        return zones

    def _premium_discount_zones(self, m15: CandleArray, atr: float, bias: int) -> List[Dict]:
        zones = []
        n = len(m15)
        if n < 30:
            return zones

        seg = 30
        highs, lows = m15.h[-seg:], m15.l[-seg:]
        swing_hi_idx, swing_lo_idx = np.argmax(highs), np.argmin(lows)
        swing_hi, swing_lo = float(highs[swing_hi_idx]), float(lows[swing_lo_idx])
        leg = swing_hi - swing_lo
        if leg <= atr * 0.5:
            return zones

        fib_50 = swing_lo + leg * 0.5
        fib_618 = swing_lo + leg * 0.618

        if bias == 1 and swing_lo_idx < swing_hi_idx:
            zones.append({"price": float(fib_50), "type": "FIB_50_DISCOUNT", "tool": "fib", "strength": 0.65})
            zones.append({"price": float(fib_618), "type": "FIB_618_DISCOUNT", "tool": "fib", "strength": 0.7})
        elif bias == -1 and swing_hi_idx < swing_lo_idx:
            zones.append({"price": float(fib_50), "type": "FIB_50_PREMIUM", "tool": "fib", "strength": 0.65})
            zones.append({"price": float(fib_618), "type": "FIB_618_PREMIUM", "tool": "fib", "strength": 0.7})
        return zones

    # =========================================================================
    # ULTRASONIC CONFIRMATION
    # =========================================================================
    def _detect_m1_confirmation(self, m1: CandleArray, zone: Dict, bias: int) -> Tuple[bool, str]:
        if len(m1) < 4:
            return False, ""
        o, h, l, c = m1.o[-2], m1.h[-2], m1.l[-2], m1.c[-2]
        po, pc = m1.o[-3], m1.c[-3]
        body, total = abs(c - o), h - l
        if total == 0:
            return False, ""
        
        upper_wick = h - max(o, c)
        lower_wick = min(o, c) - l

        is_bull_reject = lower_wick > total * 0.45 and c > o
        is_bear_reject = upper_wick > total * 0.45 and c < o
        is_bull_engulf = c > o and c > m1.h[-3] and body > (abs(pc-po) * 1.1)
        is_bear_engulf = c < o and c < m1.l[-3] and body > (abs(pc-po) * 1.1)

        if bias == 1 and (is_bull_reject or is_bull_engulf):
            return True, "M1_PIN_BAR" if is_bull_reject else "M1_ENGULFING"
        if bias == -1 and (is_bear_reject or is_bear_engulf):
            return True, "M1_PIN_BAR" if is_bear_reject else "M1_ENGULFING"
        return False, ""


    def _is_killzone(self, md: MarketData) -> bool:
        try:
            hour = md.timestamp.hour
            for _, (start, end) in KILLZONES.items():
                if start <= hour < end:
                    return True
        except Exception:
            pass
        return False

    def _is_volume_spike(self, m1: CandleArray, vol_spike_factor: float) -> bool:
        if len(m1) < 21: return False
        avg_vol = np.mean(m1.v[-21:-1])
        if avg_vol == 0: return False
        return float(m1.v[-2]) > avg_vol * vol_spike_factor

    def _calculate_confluence(self, zone: Dict, vol_confirmed: bool, adx: float, 
                              killzone: bool, pattern: str, adx_threshold: float) -> float:
        score = 0.10 
        tool = zone.get("tool", "")
        if tool in ["vwap", "volume_profile", "fvg"]: score += 0.15
        elif tool == "turtle_soup": score += 0.20
        elif tool == "fib": score += 0.10

        score += zone.get("strength", 0.5) * 0.10
        confl = zone.get("confluence_count", 0)
        if confl >= 2: score += 0.15
        elif confl >= 1: score += 0.10
        
        if vol_confirmed: score += 0.10
        if adx > adx_threshold: score += 0.10
        if killzone: score += 0.10
        if pattern == "ENGULFING": score += 0.05
        return min(score, 0.95)

    def _compute_sl_tp(self, direction: str, price: float, zone: Dict,
                       all_zones: List[Dict], atr: float, point: float, 
                       sl_atr_mult: float, min_rr: float) -> Tuple[float, float, float]:
        sl_buffer = atr * sl_atr_mult
        wick = zone.get("sweep_wick", zone["price"])

        if direction == "BUY":
            sl = min(zone["price"], wick) - sl_buffer
            tp_candidates = [z["price"] for z in all_zones if z["price"] > price + atr * 2]
            tp = min(tp_candidates) if tp_candidates else price + (price - sl) * min_rr
        else:
            sl = max(zone["price"], wick) + sl_buffer
            tp_candidates = [z["price"] for z in all_zones if z["price"] < price - atr * 2]
            tp = max(tp_candidates) if tp_candidates else price - (sl - price) * min_rr

        risk = abs(price - sl)
        rr = abs(tp - price) / risk if risk > 0 else 0
        return sl, tp, rr

    def get_metrics(self, market_data: MarketData) -> Dict[str, Any]:
        return {"last_rejection": self.last_rejection_reason}

    def get_thresholds(self) -> Dict[str, Any]:
        return {
            "min_rr": self.min_rr,
            "adx_threshold": self.adx_threshold,
            "vwap_session_bars": self.vwap_session_bars,
            "vol_spike_factor": self.vol_spike_factor,
        }