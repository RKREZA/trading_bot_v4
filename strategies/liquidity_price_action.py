"""
LIQUIDITY PRICE ACTION (LPA) STRATEGY
======================================
Institutional-grade Smart Money Concepts strategy using top-down analysis:
  D1 (Bias via BOS/CHoCH + EMA) → H1 (Zone Discovery) → M15 (Execution)

Zone Detection Tools:
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

# --- Liquidity Price Action Strategy ---
# V6-INSIGNIA Institutional Gold Scalping Engine
# ZERO HARDCODING: All parameters driven by config/symbols/XAUUSDm.json


class LiquidityPriceActionStrategy(BaseStrategy):
    """
    Institutional LPA Strategy — M15 execution with D1→H1 top-down bias.
    Symbol-agnostic: uses MarketData.point for all price calculations.
    """

    def __init__(self, strategy_id: str, config: dict):
        super().__init__(strategy_id, config)
        
        # Load Strategy Configuration
        cfg = self.get_strat_config()
        lookbacks = cfg.get("lookbacks")
        weights = cfg.get("weights")
        
        # Lookbacks
        self.vwap_session_bars = lookbacks.get("vwap_session_bars")
        self.vp_lookback = lookbacks.get("volume_profile_lookback")
        self.vp_bins = lookbacks.get("volume_profile_bins")
        self.ts_lookback = lookbacks.get("turtle_soup_lookback")
        
        # Weights (Base)
        self.base_score = weights.get("base_score")
        
        # Killzones (from config)
        self.killzones = cfg.get("killzones")
        
        # Weights & Logic blocks
        self.weights = cfg.get("weights")
        self.confirmation = cfg.get("confirmation")
        self.strengths = cfg.get("strengths")
        self.thresholds = cfg.get("thresholds")
        
        # State tracking
        self.ts_atr_mult = cfg.get("turtle_soup_threshold_atr_mult")
        self.zone_prox_mult = cfg.get("zone_proximity_atr_mult")
        self.vol_spike_factor = cfg.get("volume_spike_factor")
        self.adx_threshold = cfg.get("adx_trend_threshold")
        self.sl_atr_mult = cfg.get("sl_atr_buffer_mult")
        
        self.fvg_lookback = lookbacks.get("fvg_lookback")
        self.fib_lookback = lookbacks.get("fib_lookback")
        self.struct_lookback = lookbacks.get("structure_lookback")
        self.fvg_min_atr_mult = cfg.get("fvg_min_atr_mult")
        self.min_rr = cfg.get("min_rr")
        
        self.last_rejection_reason = None

    # =========================================================================
    # MAIN SIGNAL PIPELINE
    # =========================================================================
    def generate_signal(self, market_data: MarketData) -> Optional[TradeSignal]:
        d1 = market_data.d1_candles
        h1 = market_data.htf_candles
        m15 = market_data.m15_candles

        if h1 is None or m15 is None:
            self.last_rejection_reason = "Missing MTF data (H1/M15)"
            return None
        if len(h1) < 30 or len(m15) < 30:
            self.last_rejection_reason = "Insufficient history"
            return None
        # D1 is optional — if missing, fall back to H1-only bias
        has_d1 = d1 is not None and len(d1) >= 60

        price = market_data.bid if market_data.bid > 0 else market_data.current_price
        point = market_data.point or 0.01
        session = market_data.session

        # --- Session gate ---
        strat_cfg = self.get_strat_config()
        allowed_sessions = strat_cfg.get("allowed_sessions")
            
        if session not in allowed_sessions:
            self.last_rejection_reason = f"Session gated: {session}"
            return None

        # =====================================================================
        # STEP 1: D1 BIAS (BOS/CHoCH + EMA confirmation) or H1 fallback
        # =====================================================================
        bias = 0
        if has_d1:
            bias = self._get_daily_bias(d1)
            
        # THE FIX: If D1 is chopping/ranging (bias == 0) OR missing, drop down to H1 trend
        if bias == 0:
            bias = self._get_ema_trend(h1, fast=20, slow=50)

        if bias == 0:
            self.last_rejection_reason = "No directional bias"
            return None

        # ATRs
        atr_h1 = h1.atr(14)
        atr_m15 = m15.atr(14)
        if len(atr_h1) < 2 or len(atr_m15) < 2:
            return None
        cur_atr_h1 = atr_h1[-1]
        cur_atr_m15 = atr_m15[-1]
        if np.isnan(cur_atr_h1) or cur_atr_h1 == 0:
            return None

        # --- Dynamic Parameter Resolution (STRICT CONFIG) ---
        cfg = self.get_strat_config(session)
        min_conf = float(cfg.get("min_confidence"))
        min_rr = float(cfg.get("min_rr"))
        adx_thresh = float(cfg.get("adx_trend_threshold"))
        zone_prox = float(cfg.get("zone_proximity_atr_mult"))
        vol_spike = float(cfg.get("volume_spike_factor"))
        sl_mult = float(cfg.get("sl_atr_buffer_mult"))
        fvg_min_atr = float(cfg.get("fvg_min_atr_mult"))



        # =====================================================================
        # STEP 2: H1 ZONE DISCOVERY
        # =====================================================================
        zones = self._find_zones(h1, m15, cur_atr_h1, bias, price, cfg)
        if not zones:
            self.last_rejection_reason = "No active zones"
            return None

        # Find nearest zone within proximity
        proximity = cur_atr_h1 * zone_prox
        active_zone = None
        for z in zones:
            if abs(price - z["price"]) <= proximity:
                active_zone = z
                break

        if active_zone is None:
            self.last_rejection_reason = "Price not at zone"
            return None

        # =====================================================================
        # STEP 3: M1 ULTRASONIC CONFIRMATION (Extreme Frequency)
        # =====================================================================
        m1 = market_data.m1_candles
        if m1 is None or len(m1) < 10:
            # Fallback to M5 if M1 missing
            confirmed, pattern = self._detect_m5_confirmation(market_data.m5_candles, active_zone, bias)
        else:
            confirmed, pattern = self._detect_m1_confirmation(m1, active_zone, bias)

        if not confirmed:
            self.last_rejection_reason = f"No M1/M5 confirmation at {active_zone['type']}"
            return None

        # =====================================================================
        # STEP 4: KILLZONE CHECK
        # =====================================================================
        kz_bonus = self._is_killzone(market_data.timestamp.hour)

        # =====================================================================
        # STEP 5: CONFLUENCE SCORING
        # =====================================================================
        adx_series = h1.adx(14)
        adx_val = adx_series[-1] if len(adx_series) > 0 else 0
        adx_prev = adx_series[-2] if len(adx_series) > 1 else 0
        adx_increasing = adx_val > adx_prev
        
        vol_confirmed = self._is_volume_spike(m15, vol_spike)
        
        # RSI Filter (Institutional Reversal Confirmation)
        # Use M1 RSI for extreme precision
        m1 = market_data.m1_candles
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
    # EMA TREND HELPER (H1 fallback when D1 unavailable)
    # =========================================================================
    def _get_ema_trend(self, candles: CandleArray, fast: int = 20, slow: int = 50) -> int:
        ema_f = candles.ema(fast)
        ema_s = candles.ema(slow)
        if len(ema_f) < 2 or len(ema_s) < 2:
            return 0
        if np.isnan(ema_f[-1]) or np.isnan(ema_s[-1]):
            return 0
        if ema_f[-1] > ema_s[-1]:
            return 1
        elif ema_f[-1] < ema_s[-1]:
            return -1
        return 0

    # =========================================================================
    # D1 BIAS: BOS/CHoCH + EMA 20/50 (Relaxed Intraday Variant)
    # =========================================================================
    def _get_daily_bias(self, d1: CandleArray) -> int:
        ema20 = d1.ema(20)
        ema50 = d1.ema(50)
        
        if len(ema20) < 2 or len(ema50) < 2:
            return 0
            
        e20, e50 = ema20[-1], ema50[-1]
        if np.isnan(e20) or np.isnan(e50):
            return 0

        # Faster EMA alignment
        ema_bias = 0
        if e20 > e50:
            ema_bias = 1
        elif e20 < e50:
            ema_bias = -1

        # BOS/CHoCH: check last 20 daily swings
        struct_bias = self._detect_structure(d1, lookback=20)

        # SCENARIO A: Absolute Agreement (High Probability)
        if ema_bias == struct_bias and ema_bias != 0:
            return ema_bias
            
        # SCENARIO B: Structure overrides MA lag (CHoCH occurred, EMAs haven't crossed yet)
        if struct_bias != 0:
            return struct_bias
            
        # SCENARIO C: Price is structurally ranging, but has short-term EMA momentum
        if struct_bias == 0 and ema_bias != 0:
            return ema_bias

        return 0

    def _detect_structure(self, candles: CandleArray, lookback: int = 20) -> int:
        """Detects Break of Structure: higher-highs/higher-lows = bullish, vice versa."""
        if len(candles) < lookback + 5:
            return 0
        h = candles.h
        l = candles.l
        # Find swing highs/lows (3-bar pivots)
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

        # HH + HL = bullish BOS
        hh = swing_highs[-1] > swing_highs[-2]
        hl = swing_lows[-1] > swing_lows[-2]
        # LL + LH = bearish BOS
        ll = swing_lows[-1] < swing_lows[-2]
        lh = swing_highs[-1] < swing_highs[-2]

        if hh and hl:
            return 1
        if ll and lh:
            return -1
        # CHoCH: structure shift (e.g. was making HH but now LL)
        if ll and not lh:
            return -1  # bearish CHoCH
        if hh and not hl:
            return 1  # bullish CHoCH
        return 0

    # =========================================================================
    # ZONE DISCOVERY (H1 + M15)
    # =========================================================================
    def _find_zones(self, h1: CandleArray, m15: CandleArray, atr: float,
                    bias: int, price: float, cfg: Dict) -> List[Dict]:
        """Aggregates zones from all tools, sorted by relevance to current price."""
        zones = []

        # 1. VWAP bands
        vwap_bars = self.vwap_session_bars
        zones.extend(self._vwap_zones(m15, atr, bias, vwap_bars))
        # 2. Volume Profile
        vp_lookback = self.vp_lookback
        vp_bins = self.vp_bins
        zones.extend(self._volume_profile_zones(h1, atr, bias, vp_lookback, vp_bins))
        # 3. Turtle Soup
        ts_lookback = self.ts_lookback
        ts_atr_mult = self.ts_atr_mult
        zones.extend(self._turtle_soup_zones(h1, atr, bias, ts_lookback, ts_atr_mult))
        # 4. Fair Value Gaps
        fvg_atr_mult = self.fvg_min_atr_mult
        zones.extend(self._fvg_zones(h1, atr, bias, fvg_atr_mult))
        # 5. Premium/Discount
        zones.extend(self._premium_discount_zones(h1, atr, bias))

        if not zones:
            return []

        # Sort by proximity to current price
        zones.sort(key=lambda z: abs(price - z["price"]))

        # Check for confluence (multiple tools marking the same area)
        for i, z in enumerate(zones):
            nearby_count = sum(
                1 for oz in zones
                if oz is not z and abs(oz["price"] - z["price"]) < atr * self.zone_prox_mult
            )
            z["confluence_count"] = nearby_count

        return zones

    def _vwap_zones(self, m15: CandleArray, atr: float, bias: int, vwap_session_bars: int) -> List[Dict]:
        """VWAP ±1σ/±2σ as dynamic zones."""
        zones = []
        if len(m15) < vwap_session_bars:
            return zones
        vwap, u1, l1, u2, l2 = m15.vwap(vwap_session_bars)
        if np.isnan(vwap[-1]):
            return zones

        s = self.strengths
        if bias == 1:  # Bullish: look for support zones
            zones.append({"price": float(l1[-1]), "type": "VWAP_-1SD", "tool": "vwap", "strength": s.get("vwap_sd1")})
            zones.append({"price": float(l2[-1]), "type": "VWAP_-2SD", "tool": "vwap", "strength": s.get("vwap_sd2")})
            zones.append({"price": float(vwap[-1]), "type": "VWAP", "tool": "vwap", "strength": s.get("vwap_base")})
        else:  # Bearish: look for resistance zones
            zones.append({"price": float(u1[-1]), "type": "VWAP_+1SD", "tool": "vwap", "strength": s.get("vwap_sd1")})
            zones.append({"price": float(u2[-1]), "type": "VWAP_+2SD", "tool": "vwap", "strength": s.get("vwap_sd2")})
            zones.append({"price": float(vwap[-1]), "type": "VWAP", "tool": "vwap", "strength": s.get("vwap_base")})
        return zones

    def _volume_profile_zones(self, h1: CandleArray, atr: float, bias: int, vp_lookback: int, vp_bins: int) -> List[Dict]:
        """POC/VAH/VAL from Fixed Range Volume Profile."""
        zones = []
        if len(h1) < vp_lookback:
            return zones
        vp = h1.volume_profile(vp_lookback, vp_bins)
        if np.isnan(vp["poc"]):
            return zones

        s = self.strengths
        zones.append({"price": vp["poc"], "type": "VP_POC", "tool": "volume_profile", "strength": s.get("vp_poc")})
        if bias == 1:
            zones.append({"price": vp["val"], "type": "VP_VAL", "tool": "volume_profile", "strength": s.get("vp_val_vah")})
        else:
            zones.append({"price": vp["vah"], "type": "VP_VAH", "tool": "volume_profile", "strength": s.get("vp_val_vah")})
        return zones

    def _turtle_soup_zones(self, h1: CandleArray, atr: float, bias: int, lookback: int, atr_mult: float) -> List[Dict]:
        """Turtle Soup: detect false breakouts of prior swing extremes."""
        zones = []
        n = len(h1)
        if n < lookback + 3:
            return zones

        threshold = atr * atr_mult

        # Find the 20-bar extreme
        recent_h = h1.h[-(lookback + 3):-3]
        recent_l = h1.l[-(lookback + 3):-3]
        swing_high = np.max(recent_h)
        swing_low = np.min(recent_l)

        last_h = h1.h[-2]
        last_l = h1.l[-2]
        last_c = h1.c[-2]

        # Bearish Turtle Soup: price sweeps above swing high then closes back below
        if bias == -1 and last_h > swing_high and last_c < swing_high:
            sweep_dist = last_h - swing_high
            if sweep_dist <= threshold:
                zones.append({
                    "price": float(swing_high),
                    "type": "TURTLE_SOUP_HIGH",
                    "tool": "turtle_soup",
                    "strength": self.strengths.get("turtle_soup"),
                    "sweep_wick": float(last_h),
                })

        # Bullish Turtle Soup: price sweeps below swing low then closes back above
        if bias == 1 and last_l < swing_low and last_c > swing_low:
            sweep_dist = swing_low - last_l
            if sweep_dist <= threshold:
                zones.append({
                    "price": float(swing_low),
                    "type": "TURTLE_SOUP_LOW",
                    "tool": "turtle_soup",
                    "strength": self.strengths.get("turtle_soup"),
                    "sweep_wick": float(last_l),
                })
        return zones

    def _fvg_zones(self, h1: CandleArray, atr: float, bias: int, fvg_min_atr_mult: float) -> List[Dict]:
        """Fair Value Gaps (3-candle imbalances) on H1."""
        zones = []
        n = len(h1)
        if n < 20:
            return zones

        min_gap = atr * fvg_min_atr_mult
        # Scan configured lookback for FVGs
        fvg_lookback = self.fvg_lookback
        for i in range(max(3, n - fvg_lookback), n - 1):
            # Bullish FVG: candle[i-2].high < candle[i].low (gap up)
            if h1.h[i - 2] < h1.l[i] - min_gap:
                fvg_mid = (h1.h[i - 2] + h1.l[i]) / 2.0
                if bias == 1:
                    zones.append({
                        "price": float(fvg_mid),
                        "type": "FVG_BULL",
                        "tool": "fvg",
                        "strength": self.strengths.get("fvg"),
                        "fvg_top": float(h1.l[i]),
                        "fvg_bot": float(h1.h[i - 2]),
                    })
            # Bearish FVG: candle[i-2].low > candle[i].high (gap down)
            if h1.l[i - 2] > h1.h[i] + min_gap:
                fvg_mid = (h1.l[i - 2] + h1.h[i]) / 2.0
                if bias == -1:
                    zones.append({
                        "price": float(fvg_mid),
                        "type": "FVG_BEAR",
                        "tool": "fvg",
                        "strength": self.strengths.get("fvg"),
                        "fvg_top": float(h1.l[i - 2]),
                        "fvg_bot": float(h1.h[i]),
                    })
        return zones

    def _premium_discount_zones(self, h1: CandleArray, atr: float, bias: int) -> List[Dict]:
        """Premium/Discount via Fibonacci of last H1 impulse leg."""
        zones = []
        n = len(h1)
        if n < 30:
            return zones

        # Find last significant impulse (swing high to swing low or vice versa)
        seg = self.fib_lookback
        highs = h1.h[-seg:]
        lows = h1.l[-seg:]
        swing_hi_idx = np.argmax(highs)
        swing_lo_idx = np.argmin(lows)
        swing_hi = float(highs[swing_hi_idx])
        swing_lo = float(lows[swing_lo_idx])
        leg = swing_hi - swing_lo
        if leg <= atr * self.thresholds.get("fib_min_leg_atr_mult"):
            return zones

        # Fib levels
        fib_50 = swing_lo + leg * 0.5
        fib_618 = swing_lo + leg * 0.618
        
        s = self.strengths
        if bias == 1 and swing_lo_idx < swing_hi_idx:
            # Bullish: discount zone is 50-79% retracement (lower prices)
            zones.append({"price": float(fib_50), "type": "FIB_50_DISCOUNT", "tool": "fib", "strength": s.get("fib_50")})
            zones.append({"price": float(fib_618), "type": "FIB_618_DISCOUNT", "tool": "fib", "strength": s.get("fib_618")})
        elif bias == -1 and swing_hi_idx < swing_lo_idx:
            # Bearish: premium zone is 50-79% retracement (higher prices)
            zones.append({"price": float(fib_50), "type": "FIB_50_PREMIUM", "tool": "fib", "strength": s.get("fib_50")})
            zones.append({"price": float(fib_618), "type": "FIB_618_PREMIUM", "tool": "fib", "strength": s.get("fib_618")})
        return zones

    # =========================================================================
    # M1 ULTRASONIC CONFIRMATION
    # =========================================================================
    def _detect_m1_confirmation(self, m1: CandleArray, zone: Dict,
                                  bias: int) -> Tuple[bool, str]:
        """Checks last completed M1 candle for surgical rejection/engulfing."""
        if len(m1) < 4:
            return False, ""

        o, h, l, c = m1.o[-2], m1.h[-2], m1.l[-2], m1.c[-2]
        po, pc = m1.o[-3], m1.c[-3]
        body = abs(c - o)
        total = h - l
        if total == 0:
            return False, ""

        upper_wick = h - max(o, c)
        lower_wick = min(o, c) - l

        # M1 requirement: Wick/Engulfing (Config Driven)
        wick_ratio = self.confirmation.get("m1_wick_ratio")
        engulf_factor = self.confirmation.get("m1_engulf_factor")
        
        is_bull_reject = lower_wick > total * wick_ratio and c > o
        is_bear_reject = upper_wick > total * wick_ratio and c < o

        # Full body engulfing on M1
        is_bull_engulf = c > o and c > m1.h[-3] and pc < po and body > (abs(pc-po) * engulf_factor)
        is_bear_engulf = c < o and c < m1.l[-3] and pc > po and body > (abs(pc-po) * engulf_factor)

        if bias == 1 and (is_bull_reject or is_bull_engulf):
            pattern = "M1_PIN_BAR" if is_bull_reject else "M1_ENGULFING"
            return True, pattern
        if bias == -1 and (is_bear_reject or is_bear_engulf):
            pattern = "M1_PIN_BAR" if is_bear_reject else "M1_ENGULFING"
            return True, pattern

        return False, ""
    # =========================================================================
    def _detect_m5_confirmation(self, m5: CandleArray, zone: Dict,
                                  bias: int) -> Tuple[bool, str]:
        """Checks last completed M5 candle for rejection/engulfing at zone."""
        if len(m5) < 4:
            return False, ""

        o, h, l, c = m5.o[-2], m5.h[-2], m5.l[-2], m5.c[-2]
        po, pc = m5.o[-3], m5.c[-3]
        body = abs(c - o)
        total = h - l
        if total == 0:
            return False, ""

        upper_wick = h - max(o, c)
        lower_wick = min(o, c) - l

        # Liquidity Sweep Requirement: M5 wick must exceed previous N bars (from config)
        lookback = self.confirmation.get("m5_sweep_lookback")
        swept_low = l < min(m5.l[-(lookback+2):-2]) if bias == 1 else False
        swept_high = h > max(m5.h[-(lookback+2):-2]) if bias == -1 else False

        if bias == 1 and not swept_low and zone.get("tool") != "fvg":
            return False, ""
        if bias == -1 and not swept_high and zone.get("tool") != "fvg":
            return False, ""

        # Pin bar / rejection
        wick_ratio = self.confirmation.get("m5_wick_ratio")
        is_bull_reject = lower_wick > total * wick_ratio and c > o
        is_bear_reject = upper_wick > total * wick_ratio and c < o

        # Engulfing
        is_bull_engulf = c > o and c > m5.h[-3] and pc < po
        is_bear_engulf = c < o and c < m5.l[-3] and pc > po

        if bias == 1 and (is_bull_reject or is_bull_engulf):
            pattern = "PIN_BAR" if is_bull_reject else "ENGULFING"
            return True, pattern
        if bias == -1 and (is_bear_reject or is_bear_engulf):
            pattern = "PIN_BAR" if is_bear_reject else "ENGULFING"
            return True, pattern

        return False, ""

    # =========================================================================
    # KILLZONE
    # =========================================================================
    def _is_killzone(self, hour: int) -> bool:
        """Checks if current hour is within any configured killzone."""
        for name, (start, end) in self.killzones.items():
            if start <= hour < end:
                return True
        return False

    # =========================================================================
    # VOLUME SPIKE
    # =========================================================================
    def _is_volume_spike(self, m15: CandleArray, vol_spike_factor: float) -> bool:
        if len(m15) < 21:
            return False
        avg_vol = np.mean(m15.v[-21:-1])
        if avg_vol == 0:
            return False
        return float(m15.v[-2]) > avg_vol * vol_spike_factor

    # =========================================================================
    # CONFLUENCE SCORING
    # =========================================================================
    # CONFLUENCE SCORING
    # =========================================================================
    def _calculate_confluence(self, zone: Dict, vol_confirmed: bool,
                                adx: float, killzone: bool, pattern: str, adx_threshold: float) -> float:
        w = self.weights
        score = w.get("base_score")

        # Zone tool weight
        tool = zone.get("tool", "")
        if tool == "vwap":
            score += w.get("vwap_weight")
        elif tool == "volume_profile":
            score += w.get("vp_weight")
        elif tool == "turtle_soup":
            score += w.get("ts_weight")
        elif tool == "fvg":
            score += w.get("fvg_weight")
        elif tool == "fib":
            score += w.get("fib_weight")

        # Zone strength
        score += zone.get("strength", 0.5) * w.get("strength_mult")

        # Multi-zone confluence
        confl = zone.get("confluence_count", 0)
        if confl >= 2:
            score += w.get("confluence_2_plus")
        elif confl >= 1:
            score += w.get("confluence_1_plus")

        # Volume confirmation
        if vol_confirmed:
            score += w.get("vol_spike_weight")

        # ADX trending
        if adx > adx_threshold:
            score += w.get("adx_weight")

        # Killzone bonus
        if killzone:
            score += w.get("killzone_weight")

        # Pattern bonus
        if pattern == "ENGULFING":
            score += w.get("engulfing_weight")

        return min(score, self.thresholds.get("max_confluence_score"))

    def _compute_sl_tp(self, direction: str, price: float, zone: Dict,
                        all_zones: List[Dict], atr: float,
                        point: float, sl_atr_mult: float, min_rr: float) -> Tuple[float, float, float]:
        sl_buffer = atr * sl_atr_mult

        if direction == "BUY":
            # SL below zone / sweep wick
            wick = zone.get("sweep_wick", zone["price"])
            sl = min(zone["price"], wick) - sl_buffer
            # TP at nearest resistance zone or opposing VWAP band
            tp_candidates = [
                z["price"] for z in all_zones
                if z["price"] > price and z.get("tool") != zone.get("tool")
            ]
            if not tp_candidates:
                tp = price + (price - sl) * min_rr
            else:
                tp = min(tp_candidates)
        else:
            wick = zone.get("sweep_wick", zone["price"])
            sl = max(zone["price"], wick) + sl_buffer
            tp_candidates = [
                z["price"] for z in all_zones
                if z["price"] < price and z.get("tool") != zone.get("tool")
            ]
            if not tp_candidates:
                tp = price - (sl - price) * min_rr
            else:
                tp = max(tp_candidates)

        risk = abs(price - sl)
        reward = abs(tp - price)
        rr = reward / risk if risk > 0 else 0
        return sl, tp, rr

    # =========================================================================
    # TELEMETRY
    # =========================================================================
    def get_metrics(self, market_data: MarketData) -> Dict[str, Any]:
        return {"last_rejection": self.last_rejection_reason}

    def get_thresholds(self) -> Dict[str, Any]:
        return {
            "min_rr": self.min_rr,
            "adx_threshold": self.adx_threshold,
            "vwap_session_bars": self.vwap_session_bars,
            "vol_spike_factor": self.vol_spike_factor,
        }
