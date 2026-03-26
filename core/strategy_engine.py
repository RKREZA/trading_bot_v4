"""
TRADING BOT V3 - HYBRID BREAKOUT STRATEGY
Breakout + Pullback with Strong Trend

Strategy:
1. H4 EMA Trend Direction (Primary)
2. M30 Breakout Confirmation
3. M15 Pullback Entry
4. Session Filter (London/NY only)
5. Fixed 1:2 R:R with proper SL placement
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger("trading_bot.strategy")

# Default sessions — overridden by config
_DEFAULT_SESSIONS = {"LONDON", "NEW_YORK", "LONDON/NY", "TOKYO"}

# Config key → internal session name mapping
_SESSION_KEY_MAP = {
    "TOKYO": "TOKYO",
    "LONDON": "LONDON",
    "LONDON_NY": "LONDON/NY",
    "NEW_YORK": "NEW_YORK",
}


@dataclass
class TradeSignal:
    """Trade signal data class."""
    direction: str
    entry_price: float
    stop_loss: float
    take_profit: float
    confidence: float
    confluence_score: int
    reasons: List[str] = field(default_factory=list)
    rejection_type: str = ""
    order_block: Optional[dict] = None
    fvg: Optional[dict] = None
    liquidity: Optional[dict] = None
    timestamp: datetime = None
    rr_ratio: float = 2.0


class StrategyEngine:
    """
    HYBRID BREAKOUT STRATEGY V3

    Breakout + pullback entries with multi-timeframe trend alignment
    and session filtering (London/NY only).
    """

    def __init__(self, config: dict, analysis_logger=None):
        self.config = config
        self.strategy_config = config.get("strategy", {})
        self.analysis_logger = analysis_logger
        self.silent = False

        # Settings (now configurable)
        self.min_confluence_score = self.strategy_config.get("min_confluence_score", 4)
        self.min_confidence = self.strategy_config.get("min_confidence", 50)
        self.pullback_distance_pct = self.strategy_config.get("pullback_distance_pct", 0.5)
        self.atr_period = self.strategy_config.get("atr_period", 14)
        self.swing_lookback = self.strategy_config.get("swing_lookback", 20)
        self.ema_fast = self.strategy_config.get("ema_fast", 20)
        self.ema_slow = self.strategy_config.get("ema_slow", 50)
        self.min_candle_body_pct = self.strategy_config.get("min_candle_body_pct", 40)
        self.sl_atr_buffer = self.strategy_config.get("sl_atr_buffer", 0.4)

        # Volatility filter
        self.vol_cfg = self.strategy_config.get("volatility_filter", {})
        self.vol_enabled = self.vol_cfg.get("enabled", False)
        self.vol_mult_high = self.vol_cfg.get("atr_multiplier_high", 2.5)
        self.vol_mult_low = self.vol_cfg.get("atr_multiplier_low", 0.5)
        self.vol_lookback = self.vol_cfg.get("lookback", 100)

        # Build tradeable sessions from config (true/false toggles)
        sessions_cfg = config.get("sessions", {})
        if sessions_cfg:
            self.tradeable_sessions = {
                _SESSION_KEY_MAP[k]
                for k, v in sessions_cfg.items()
                if v and k in _SESSION_KEY_MAP
            }
        else:
            self.tradeable_sessions = _DEFAULT_SESSIONS

        # Track breakouts for pullback entries
        self.recent_breakouts = {}

        self._log("=" * 50)
        self._log("HYBRID BREAKOUT STRATEGY V3")
        self._log(f"Min Confluence: {self.min_confluence_score}")
        self._log(f"Min Confidence: {self.min_confidence}%")
        self._log(f"Pullback Dist: {self.pullback_distance_pct}%")
        self._log(f"ATR Period: {self.atr_period}")
        self._log(f"Swing Lookback: {self.swing_lookback}")
        self._log(f"Volatility Filter: {self.vol_enabled} (High: {self.vol_mult_high}x, Low: {self.vol_mult_low}x)")
        self._log(f"Sessions: {', '.join(sorted(self.tradeable_sessions))}")
        self._log("=" * 50)

    def _log(self, message: str, level: str = "INFO"):
        """Log to both the analysis logger (dashboard) and Python logger."""
        if self.silent:
            return
        if self.analysis_logger:
            self.analysis_logger.log(message, level)
        logger.info(message)

    def analyze(self, symbol: str, h4_candles: List[dict], m30_candles: List[dict],
                m5_candles: List[dict], current_price: float,
                d1_candles: Optional[List[dict]] = None, session: Optional[str] = None):
        """
        Main Analysis — Breakout + Pullback Strategy.

        Args:
            symbol: Trading symbol
            h4_candles: H4 timeframe candle data
            m30_candles: M30 timeframe candle data
            m5_candles: M5 timeframe candle data
            current_price: Current market price
            session: Current trading session (e.g., "LONDON", "NEW_YORK").
                     If provided, signals are only generated during tradeable sessions.

        Returns:
            Tuple of (TradeSignal or None, h4_trend str) so callers don't
            need to recompute the trend with a separate EMA pass.
        """
        if len(h4_candles) < 50 or len(m30_candles) < 100 or len(m5_candles) < 100:
            return None, "RANGING"

        # ==========================================
        # STEP 0: SESSION FILTER
        # ==========================================
        if session and session not in self.tradeable_sessions:
            self._log(f"Session: {session} — not tradeable, skipping")
            return None, "RANGING"

        # ==========================================
        # STEP 1: H4 TREND (Simple & Clear)
        # ==========================================
        h4_trend, h4_strength = self._get_h4_trend(h4_candles)

        # D1 Super-Filter (Optional)
        d1_trend = "RANGING"
        if d1_candles:
            d1_trend, d1_strength = self._get_d1_trend(d1_candles)
            if self.config.get("strategy", {}).get("use_d1_filter", True):
                if d1_trend != "RANGING" and d1_trend != h4_trend:
                    self._log(f"Signal rejected: D1 trend ({d1_trend}) conflicts with H4 trend ({h4_trend})")
                    return None, h4_trend

        if h4_trend == "RANGING" or h4_strength < 60:
            self._log(f"H4: {h4_trend} ({h4_strength}%) — SKIP")
            return None, h4_trend

        self._log(f"H4: {h4_trend} ({h4_strength}%)")

        # ==========================================
        # STEP 1.5: VOLATILITY FILTER (Tiered)
        # ==========================================
        atr = self._calculate_atr(m30_candles)
        if self.vol_enabled:
            # Use last N candles for average ATR
            if len(m30_candles) >= self.vol_lookback:
                atr_avg = self._calculate_atr(m30_candles[-self.vol_lookback:])
            else:
                atr_avg = atr
            
            # 1. TOO HIGH: ATR > ATR_avg * self.vol_mult_high
            if atr > atr_avg * self.vol_mult_high:
                self._log(f"Volatility TOO HIGH: ATR {atr:.2f} > {atr_avg:.2f} × {self.vol_mult_high}, skipping")
                return None, h4_trend
            
            # 2. TOO LOW: ATR < ATR_avg * self.vol_mult_low
            if atr < atr_avg * self.vol_mult_low:
                self._log(f"Volatility TOO LOW: ATR {atr:.2f} < {atr_avg:.2f} × {self.vol_mult_low}, skipping")
                return None, h4_trend
            
            # 3. NORMAL HIGH / NORMAL LOW continue to trade
            if atr > atr_avg * 1.5:
                self._log(f"Volatility Normal High: ATR {atr:.2f}")
            elif atr < atr_avg * 0.8:
                self._log(f"Volatility Normal Low: ATR {atr:.2f}")

        # ==========================================
        # STEP 2: M30 STRUCTURE CONFIRMATION
        # ==========================================
        m30_trend, m30_structure = self._get_m30_structure(m30_candles, h4_trend)

        if not m30_structure:
            self._log("M30 structure NOT aligned")
            return None, h4_trend

        self._log(f"M30: {m30_trend} — Structure OK")

        # ==========================================
        # STEP 3: FIND ENTRY SIGNAL
        # ==========================================
        breakout_signal = self._check_breakout_entry(m30_candles, m5_candles, h4_trend, current_price)
        pullback_signal = self._check_pullback_entry(m30_candles, m5_candles, h4_trend, current_price)

        signal = breakout_signal or pullback_signal

        if not signal:
            self._log("No valid entry signal")
            return None, h4_trend

        # ==========================================
        # STEP 3.5: CANDLE BODY SIZE FILTER
        # ==========================================
        last_candle = m30_candles[-1]
        candle_range = last_candle["high"] - last_candle["low"]
        candle_body = abs(last_candle["close"] - last_candle["open"])
        if candle_range > 0:
            body_pct = (candle_body / candle_range) * 100
            if body_pct < self.min_candle_body_pct:
                self._log(f"Candle body too small: {body_pct:.0f}% < {self.min_candle_body_pct}%")
                return None, h4_trend

        # ==========================================
        # STEP 4: CALCULATE SL/TP (1:2–1:3 R:R)  — reuses atr from STEP 1.5
        # ==========================================
        signal = self._set_sl_tp(signal, atr, m30_candles, h4_strength)

        # ==========================================
        # STEP 5: FINAL CONFIDENCE CHECK
        # ==========================================
        confluence, reasons = self._calculate_confluence(
            h4_trend, h4_strength, m30_trend, signal, m30_candles, m5_candles
        )

        signal.confluence_score = confluence
        signal.reasons = reasons

        if confluence < self.min_confluence_score:
            self._log(f"Confluence too low: {confluence}")
            return None, h4_trend

        confidence = self._calculate_confidence(confluence, h4_strength, signal)
        signal.confidence = confidence

        if confidence < self.min_confidence:
            self._log(f"Confidence too low: {confidence}%")
            return None, h4_trend

        self._log(f"✓ SIGNAL: {signal.direction} @ {signal.entry_price:.5f}")
        self._log(f"  SL: {signal.stop_loss:.5f} | TP: {signal.take_profit:.5f}")
        self._log(f"  R:R: {signal.rr_ratio:.2f} | Conf: {confidence:.0f}%")

        return signal, h4_trend

    # ------------------------------------------------------------------
    # Trend Detection
    # ------------------------------------------------------------------

    def _get_d1_trend(self, d1_candles: list) -> tuple:
        """Analyze Daily (D1) trend for long-term bias."""
        if not d1_candles or len(d1_candles) < 50:
            return "RANGING", 0
        
        closes = np.array([c["close"] for c in d1_candles])
        
        # EMA alignment for D1
        ema_fast = self._ema(closes, self.ema_fast)
        ema_slow = self._ema(closes, self.ema_slow)
        
        ema_aligned_bull = ema_fast[-1] > ema_slow[-1]
        ema_aligned_bear = ema_fast[-1] < ema_slow[-1]
        
        highs = [c["high"] for c in d1_candles[-10:]]
        lows = [c["low"] for c in d1_candles[-10:]]
        hh_hl = highs[-1] > highs[-3] and lows[-1] > lows[-3]
        lh_ll = highs[-1] < highs[-3] and lows[-1] < lows[-3]
        
        if ema_aligned_bull and hh_hl:
            return "BULLISH", 80
        if ema_aligned_bear and lh_ll:
            return "BEARISH", 80
            
        return "RANGING", 40

    def _get_h4_trend(self, candles: List[dict]) -> Tuple[str, int]:
        """
        Determine H4 trend using EMA slope and price position.
        Returns: (trend_direction, strength_percentage)
        """
        if len(candles) < 50:
            return "RANGING", 0

        closes = np.array([c["close"] for c in candles])

        ema20 = self._ema(closes, self.ema_fast)
        ema50 = self._ema(closes, self.ema_slow)

        price = closes[-1]

        above_20 = price > ema20[-1]
        above_50 = price > ema50[-1]
        below_20 = price < ema20[-1]
        below_50 = price < ema50[-1]

        # EMA Slope (comparing last 5 candles)
        ema20_slope = (ema20[-1] - ema20[-5]) / ema20[-5] * 100 if ema20[-5] != 0 else 0
        ema50_slope = (ema50[-1] - ema50[-5]) / ema50[-5] * 100 if ema50[-5] != 0 else 0

        # Bullish scoring
        bull_score = 0
        if above_20 and above_50:
            bull_score += 30
        if ema20_slope > 0.1:
            bull_score += 25
        if ema50_slope > 0.05:
            bull_score += 15
        if ema20[-1] > ema50[-1]:
            bull_score += 20
        highs = [c["high"] for c in candles[-10:]]
        lows = [c["low"] for c in candles[-10:]]
        if highs[-1] > highs[-3] and lows[-1] > lows[-3]:
            bull_score += 10

        # Bearish scoring
        bear_score = 0
        if below_20 and below_50:
            bear_score += 30
        if ema20_slope < -0.1:
            bear_score += 25
        if ema50_slope < -0.05:
            bear_score += 15
        if ema20[-1] < ema50[-1]:
            bear_score += 20
        if highs[-1] < highs[-3] and lows[-1] < lows[-3]:
            bear_score += 10

        # Decision (relaxed to 60 for higher trade frequency)
        if bull_score >= 60 and bull_score > bear_score + 15:
            return "BULLISH", bull_score
        if bear_score >= 60 and bear_score > bull_score + 15:
            return "BEARISH", bear_score

        return "RANGING", max(bull_score, bear_score)

    def _get_m30_structure(self, candles: List[dict], h4_trend: str) -> Tuple[str, bool]:
        """Check if M30 price structure aligns with the H4 trend."""
        if len(candles) < 50:
            return "RANGING", False

        highs = [c["high"] for c in candles[-20:]]
        lows = [c["low"] for c in candles[-20:]]
        swing_high = max(highs[-10:])
        swing_low = min(lows[-10:])
        current_price = candles[-1]["close"]

        if h4_trend == "BULLISH":
            recent_lows = lows[-15:]
            hl_count = sum(1 for i in range(1, len(recent_lows)) if recent_lows[i] >= recent_lows[i - 1] * 0.999)
            if hl_count >= 8:
                return "BULLISH", True
            midpoint = (swing_high + swing_low) / 2
            if current_price > midpoint:
                return "BULLISH", True

        if h4_trend == "BEARISH":
            recent_highs = highs[-15:]
            lh_count = sum(1 for i in range(1, len(recent_highs)) if recent_highs[i] <= recent_highs[i - 1] * 1.001)
            if lh_count >= 8:
                return "BEARISH", True
            midpoint = (swing_high + swing_low) / 2
            if current_price < midpoint:
                return "BEARISH", True

        return "RANGING", False

    # ------------------------------------------------------------------
    # M5 EMA Alignment Helper
    # ------------------------------------------------------------------

    def _check_m5_ema_alignment(self, m5_candles: List[dict], trend: str) -> bool:
        """
        Check if M5 EMA20 slope aligns with the intended trade direction.
        Returns True if aligned, False otherwise.
        """
        if len(m5_candles) < 30:
            return True  # not enough data, skip filter

        m5_closes = np.array([c["close"] for c in m5_candles])
        m5_ema = self._ema(m5_closes, self.ema_fast)

        # Slope over last 5 M5 candles (25 mins)
        slope = (m5_ema[-1] - m5_ema[-5]) / m5_ema[-5] * 100 if m5_ema[-5] != 0 else 0

        if trend == "BULLISH" and slope <= 0:
            return False
        if trend == "BEARISH" and slope >= 0:
            return False

        return True

    # ------------------------------------------------------------------
    # Entry Detection
    # ------------------------------------------------------------------

    def _check_breakout_entry(self, m30_candles: List[dict], m5_candles: List[dict],
                               trend: str, current_price: float) -> Optional[TradeSignal]:
        """Detect breakout entry: price closes above/below recent swing level.
        Requires M5 EMA20 slope alignment with trade direction."""
        lookback = self.swing_lookback
        recent = m30_candles[-lookback:]

        # M5 EMA trend alignment check
        m5_ema_aligned = self._check_m5_ema_alignment(m5_candles, trend)
        if not m5_ema_aligned:
            return None

        if trend == "BULLISH":
            resistance = max(c["high"] for c in recent[:-1])
            # Current candle color must be green
            curr_candle_green = m30_candles[-1]["close"] > m30_candles[-1]["open"]
            if current_price > resistance and curr_candle_green:
                # Current: at least 1 M5 candle strong close
                m5_recent = m5_candles[-5:]
                strong_close = m5_recent[-1]["close"] > resistance
                if strong_close:
                    return TradeSignal(
                        direction="BUY", entry_price=current_price,
                        stop_loss=0, take_profit=0, confidence=0, confluence_score=0,
                        rejection_type="BREAKOUT", timestamp=datetime.now(),
                    )

        elif trend == "BEARISH":
            support = min(c["low"] for c in recent[:-1])
            # Current candle color must be red
            curr_candle_red = m30_candles[-1]["close"] < m30_candles[-1]["open"]
            if current_price < support and curr_candle_red:
                # Current: at least 1 M5 candle strong close
                m5_recent = m5_candles[-5:]
                strong_close = m5_recent[-1]["close"] < support
                if strong_close:
                    return TradeSignal(
                        direction="SELL", entry_price=current_price,
                        stop_loss=0, take_profit=0, confidence=0, confluence_score=0,
                        rejection_type="BREAKOUT", timestamp=datetime.now(),
                    )

        return None

    def _check_pullback_entry(self, m30_candles: List[dict], m5_candles: List[dict],
                               trend: str, current_price: float) -> Optional[TradeSignal]:
        """
        Detect pullback entry: price has pulled back *to* EMA20 (touched it from
        the trend side) and is now bouncing in the trend direction.

        Requires:
          1. Price came close enough to EMA (within pullback_distance_pct).
          2. The *prior* M30 candle actually crossed or touched the EMA band.
          3. M5 shows at least 2 candles confirming the bounce direction.
          4. M5 EMA20 slope aligns with trade direction.
        """
        # M5 EMA trend alignment check
        m5_ema_aligned = self._check_m5_ema_alignment(m5_candles, trend)
        if not m5_ema_aligned:
            return None

        closes = np.array([c["close"] for c in m30_candles])
        ema20 = self._ema(closes, self.ema_fast)
        ema_value = ema20[-1]
        ema_prev = ema20[-2]  # previous bar EMA for touch confirmation

        if trend == "BULLISH" and current_price > ema_value:
            # Current candle must be green
            curr_candle_green = m30_candles[-1]["close"] > m30_candles[-1]["open"]
            distance_pct = (current_price - ema_value) / current_price * 100
            # Previous candle must have touched or dipped below EMA (the actual pullback touch)
            prev_low = m30_candles[-2]["low"]
            touched_ema = prev_low <= ema_prev * 1.002  # within 0.2% below EMA
            if distance_pct < self.pullback_distance_pct and touched_ema and curr_candle_green:
                m5_recent = m5_candles[-3:]
                bullish_candles = sum(1 for c in m5_recent if c["close"] > c["open"])
                if bullish_candles >= 2:
                    return TradeSignal(
                        direction="BUY", entry_price=current_price,
                        stop_loss=0, take_profit=0, confidence=0, confluence_score=0,
                        rejection_type="PULLBACK_EMA", timestamp=datetime.now(),
                    )

        elif trend == "BEARISH" and current_price < ema_value:
            # Current candle must be red
            curr_candle_red = m30_candles[-1]["close"] < m30_candles[-1]["open"]
            distance_pct = (ema_value - current_price) / current_price * 100
            # Previous candle must have touched or risen above EMA
            prev_high = m30_candles[-2]["high"]
            touched_ema = prev_high >= ema_prev * 0.998  # within 0.2% above EMA
            if distance_pct < self.pullback_distance_pct and touched_ema and curr_candle_red:
                m5_recent = m5_candles[-3:]
                bearish_candles = sum(1 for c in m5_recent if c["close"] < c["open"])
                if bearish_candles >= 2:
                    return TradeSignal(
                        direction="SELL", entry_price=current_price,
                        stop_loss=0, take_profit=0, confidence=0, confluence_score=0,
                        rejection_type="PULLBACK_EMA", timestamp=datetime.now(),
                    )

        return None

    # ------------------------------------------------------------------
    # SL/TP and Scoring
    # ------------------------------------------------------------------

    def _set_sl_tp(self, signal: TradeSignal, atr: float, m30_candles: List[dict],
                   h4_strength: int = 60) -> TradeSignal:
        """
        Tight Stop Loss + Dynamic Take Profit (1:2 to 1:4 R:R).

        SL placement:
          - Uses the last 5 candles for the swing pivot (tight, precise)
          - Adds a 0.2× ATR buffer beyond the swing to absorb wick noise
          - Enforces a minimum SL size of 0.5× ATR so spread cannot trigger it

        TP placement (trend-strength based):
          - h4_strength >= 80  →  1:4 R:R (strong trend, ride it)
          - h4_strength >= 70  →  1:3 R:R (moderate trend)
          - default            →  1:2 R:R (minimum acceptable)
        """
        # 5-candle tight swing window
        tight_window = 5
        recent = m30_candles[-tight_window:]

        # Determine R:R multiplier from trend strength (lowered for realism)
        if h4_strength >= 80:
            rr_multiplier = 3.0
        elif h4_strength >= 70:
            rr_multiplier = 2.5
        else:
            rr_multiplier = 2.0

        sl_buffer = self.sl_atr_buffer  # configurable, default 0.4
        min_sl_distance = atr * 0.5  # floor to avoid spread noise

        if signal.direction == "BUY":
            swing_low = min(c["low"] for c in recent)
            raw_sl = swing_low - atr * sl_buffer
            # Enforce minimum distance from entry
            if signal.entry_price - raw_sl < min_sl_distance:
                raw_sl = signal.entry_price - min_sl_distance
            signal.stop_loss = raw_sl
            risk = signal.entry_price - signal.stop_loss
            signal.take_profit = signal.entry_price + (risk * rr_multiplier)
        else:
            swing_high = max(c["high"] for c in recent)
            raw_sl = swing_high + atr * sl_buffer
            if raw_sl - signal.entry_price < min_sl_distance:
                raw_sl = signal.entry_price + min_sl_distance
            signal.stop_loss = raw_sl
            risk = signal.stop_loss - signal.entry_price
            signal.take_profit = signal.entry_price - (risk * rr_multiplier)

        actual_risk   = abs(signal.entry_price - signal.stop_loss)
        actual_reward = abs(signal.take_profit  - signal.entry_price)
        signal.rr_ratio = actual_reward / actual_risk if actual_risk > 0 else 0

        return signal

    def _calculate_confluence(self, h4_trend: str, h4_strength: int,
                               m30_trend: str, signal: TradeSignal,
                               m30_candles: List[dict],
                               m5_candles: List[dict] = None) -> Tuple[int, List[str]]:
        """Calculate confluence score based on multiple factors."""
        score = 0
        reasons = []

        # H4 Trend (1-2 points)
        if h4_strength >= 70:
            score += 2
            reasons.append(f"H4_Strong({h4_strength}%)")
        else:
            score += 1
            reasons.append(f"H4_Trend({h4_strength}%)")

        # M30 Alignment (1-2 points)
        if h4_trend == m30_trend:
            score += 2
            reasons.append("M30_Aligned")
        else:
            score += 1
            reasons.append("M30_OK")

        # Entry Type (1-2 points)
        if signal.rejection_type == "BREAKOUT":
            score += 2
            reasons.append("Breakout")
        else:
            score += 1
            reasons.append("Pullback")

        # Momentum (0-1 points)
        recent = m30_candles[-5:]
        if signal.direction == "BUY":
            if all(c["close"] > c["open"] for c in recent[-2:]):
                score += 1
                reasons.append("Momentum")
        else:
            if all(c["close"] < c["open"] for c in recent[-2:]):
                score += 1
                reasons.append("Momentum")

        # M5 EMA alignment (0-1 points)
        if m5_candles and len(m5_candles) >= 30:
            m5_closes = np.array([c["close"] for c in m5_candles])
            m5_ema = self._ema(m5_closes, self.ema_fast)
            m5_slope = (m5_ema[-1] - m5_ema[-5]) / m5_ema[-5] * 100 if m5_ema[-5] != 0 else 0
            if (signal.direction == "BUY" and m5_slope > 0.05) or \
               (signal.direction == "SELL" and m5_slope < -0.05):
                score += 1
                reasons.append("M5_EMA")

        # Clean candle structure (0-1 points) — NEW
        last_m30 = m30_candles[-1]
        m30_range = last_m30["high"] - last_m30["low"]
        m30_body = abs(last_m30["close"] - last_m30["open"])
        if m30_range > 0 and (m30_body / m30_range) >= 0.6:
            score += 1
            reasons.append("CleanCandle")

        return score, reasons

    def _calculate_confidence(self, confluence: int, h4_strength: int, signal: TradeSignal) -> float:
        """Calculate confidence percentage from confluence and R:R."""
        base = confluence * 10
        base += (h4_strength - 50) * 0.3

        if signal.rr_ratio >= 2.0:
            base += 10
        elif signal.rr_ratio >= 1.8:
            base += 5

        return min(max(base, 30), 95)

    def _determine_trend(self, candles: List[dict]) -> str:
        """Alias for dashboard compatibility."""
        trend, _ = self._get_h4_trend(candles)
        return trend

    # ------------------------------------------------------------------
    # Technical Indicators
    # ------------------------------------------------------------------

    @staticmethod
    def _ema(data: np.ndarray, period: int) -> np.ndarray:
        """
        Calculate Exponential Moving Average using NumPy (vectorized).
        """
        if len(data) == 0:
            return np.array([])
        
        alpha = 2 / (period + 1)
        # Use a vectorized core for EMA calculation
        # This is significantly faster than a Python loop
        ema = np.empty_like(data)
        ema[0] = data[0]
        for i in range(1, len(data)):
            ema[i] = data[i] * alpha + ema[i-1] * (1 - alpha)
        return ema

    def _calculate_atr(self, candles: List[dict], period: int = None) -> float:
        """
        Calculate Average True Range using vectorized NumPy operations.
        """
        if period is None:
            period = self.atr_period
        
        if len(candles) < 2:
            return 100.0
            
        # Convert to numpy arrays for vectorization
        highs = np.array([c['high'] for c in candles])
        lows = np.array([c['low'] for c in candles])
        prev_closes = np.array([candles[i-1]['close'] for i in range(1, len(candles))])
        
        # True Range components
        h_l = highs[1:] - lows[1:]
        h_pc = np.abs(highs[1:] - prev_closes)
        l_pc = np.abs(lows[1:] - prev_closes)
        
        tr = np.maximum(h_l, np.maximum(h_pc, l_pc))
        
        # Simple Moving Average of True Range over the period
        if len(tr) < period:
            return float(np.mean(tr)) if len(tr) > 0 else 100.0
            
        return float(np.mean(tr[-period:]))

    @staticmethod
    def get_session_from_hour(hour: int) -> str:
        """
        Determine trading session based on UTC+0 hour.
        
        TOKYO: 00:00 - 08:00
        LONDON: 08:00 - 13:00 (exclusive of overlap)
        LONDON/NY: 13:00 - 17:00
        NEW_YORK: 17:00 - 22:00
        """
        if 0 <= hour < 8:
            return "TOKYO"
        if 8 <= hour < 13:
            return "LONDON"
        if 13 <= hour < 17:
            return "LONDON/NY"
        if 17 <= hour < 22:
            return "NEW_YORK"
        return "CLOSED"