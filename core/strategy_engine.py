"""
TRADING BOT V3 - HYBRID BREAKOUT STRATEGY (PHASE 7 HIGH-CONF)
Refined for XAUUSDm M5 with MTF Alignment, Weighted Confluence (70%+), and High Frequency (3+/day).
"""

import logging
import numpy as np
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

# Relative imports from core package
from .regime import MarketRegime

logger = logging.getLogger("trading_bot.strategy")

_DEFAULT_SESSIONS = {"LONDON", "NEW_YORK", "LONDON/NY", "TOKYO"}
_SESSION_KEY_MAP = {
    "TOKYO": "TOKYO",
    "LONDON": "LONDON",
    "LONDON_NY": "LONDON/NY",
    "LONDON/NY": "LONDON/NY",
    "NEW_YORK": "NEW_YORK",
}

@dataclass
class TradeSignal:
    direction: str
    entry_price: float
    stop_loss: float
    take_profit: float
    tp1_price: float = 0.0
    tp2_price: float = 0.0
    tp3_price: float = 0.0
    confidence: float = 0.0
    confluence_score: int = 0
    reasons: List[str] = field(default_factory=list)
    rejection_type: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    rr_ratio: float = 2.0

class StrategyEngine:
    def __init__(self, config: dict, analysis_logger=None, silent: bool = False):
        self.config = config
        self.strategy_config = config.get("strategy_defaults", {})
        self.analysis_logger = analysis_logger
        self.silent = silent

        self.min_confluence_score = self.strategy_config.get("min_confluence_score", 2)
        self.min_confidence = self.strategy_config.get("min_confidence", 60)
        self.pullback_distance_pct = self.strategy_config.get("pullback_distance_pct", 0.5)
        self.atr_period = self.strategy_config.get("atr_period", 14)
        self.swing_lookback = self.strategy_config.get("swing_lookback", 3)
        self.ema_fast = self.strategy_config.get("ema_fast", 20)
        self.ema_slow = self.strategy_config.get("ema_slow", 50)
        self.min_candle_body_pct = self.strategy_config.get("min_candle_body_pct", 20)
        self.sl_atr_buffer = self.strategy_config.get("sl_atr_buffer", 0.8)

        self.vol_cfg = self.strategy_config.get("volatility_filter", {})
        self.vol_enabled = self.vol_cfg.get("enabled", False)
        self.vol_mult_high = self.vol_cfg.get("atr_multiplier_high", 4.0)
        self.vol_mult_low = self.vol_cfg.get("atr_multiplier_low", 0.2)
        self.vol_lookback = self.vol_cfg.get("lookback", 200)

        # Choppy Mitigation
        self.max_daily_losses = self.strategy_config.get("max_daily_losses", 3)
        self.cooldown_candles = self.strategy_config.get("cooldown_candles", 20)
        self.last_stop_time: Optional[datetime] = None
        self.daily_losses = 0
        self.last_loss_date = None
        self.m5_trade_counter = 0 # To track cooldown in backtest candles
        self.last_m5_stop_index = -999

        # Sessions
        self.session_cfg = config.get("session_config", {})
        self.tradeable_sessions = {
            _SESSION_KEY_MAP[k] for k, v in self.session_cfg.items()
            if v.get("enabled", False) and k in _SESSION_KEY_MAP
        } if self.session_cfg else _DEFAULT_SESSIONS

    @staticmethod
    def get_session_from_hour(hour: int, utc_offset: int = 0) -> str:
        """
        Determines session from candle hour.
        Adjusts for UTC offset (MT5 time -> UTC time) before classification.
        """
        # Convert local/server hour to UTC
        # If server is UTC+2 (10 AM), UTC is 8 AM. So 10 - 2 = 8.
        utc_hour = (hour - utc_offset) % 24
        
        if 8 <= utc_hour < 14: return "LONDON"
        if 14 <= utc_hour < 17: return "LONDON/NY"
        if 17 <= utc_hour < 22: return "NEW_YORK"
        return "TOKYO"

    def _log(self, message: str, level: str = "INFO"):
        if self.silent: return
        if self.analysis_logger: self.analysis_logger.log(message, level)
        logger.info(message)

    def _calculate_rsi(self, prices: np.ndarray, period: int = 14) -> float:
        if len(prices) < period + 1: return 50.0
        deltas = np.diff(prices)
        if len(deltas) == 0: return 50.0
        up = np.where(deltas > 0, deltas, 0)
        down = np.where(deltas < 0, -deltas, 0)
        
        avg_up = np.mean(up[:period])
        avg_down = np.mean(down[:period])
        
        for i in range(period, len(deltas)):
            avg_up = (avg_up * (period - 1) + up[i]) / period
            avg_down = (avg_down * (period - 1) + down[i]) / period
            
        if avg_down == 0: return 100.0
        rs = avg_up / avg_down
        return 100.0 - (100.0 / (1.0 + rs))

    def _calculate_ema_series(self, prices: np.ndarray, period: int) -> np.ndarray:
        if len(prices) == 0: return np.array([])
        if period <= 0: return prices
        alpha = 2 / (period + 1)
        ema_series = np.zeros_like(prices)
        ema_series[0] = prices[0]
        for i in range(1, len(prices)):
            ema_series[i] = (prices[i] - ema_series[i-1]) * alpha + ema_series[i-1]
        return ema_series

    def _calculate_ema(self, prices: np.ndarray, period: int) -> float:
        series = self._calculate_ema_series(prices, period)
        return series[-1] if len(series) > 0 else 0.0

    def _calculate_atr(self, candles: List[dict], period: Optional[int] = None) -> float:
        if period is None:
            period = self.atr_period
            
        if not candles or len(candles) < 2: return 0.1
        highs = np.array([c['high'] for c in candles])
        lows = np.array([c['low'] for c in candles])
        closes = np.array([c['close'] for c in candles])
        
        if len(highs) < 2: return 0.1

        tr = np.maximum(highs[1:] - lows[1:], 
                        np.maximum(np.abs(highs[1:] - closes[:-1]), 
                                   np.abs(lows[1:] - closes[:-1])))
        if len(tr) == 0: return 0.1
        return np.mean(tr[-period:]) if len(tr) >= period else np.mean(tr)

    def analyze(self, symbol: str, h4_candles: List[dict], h1_candles: List[dict], m30_candles: List[dict],
                m5_candles: List[dict], current_price: float,
                d1_candles: Optional[List[dict]] = None, session: Optional[str] = None):
        
        if not h4_candles or not h1_candles or not m30_candles or not m5_candles:
            return None, "RANGING"
        
        # Performance: Truncate slices to reasonable lookback for indicator stability
        # O(N^2) Mitigation: 200 candles is enough for EMA50/RSI14 to stabilize.
        h4_candles = h4_candles[-200:]
        h1_candles = h1_candles[-200:]
        m30_candles = m30_candles[-200:]
        m5_candles = m5_candles[-400:] # M5 needs a bit more for breakout lookbacks
        
        if len(h4_candles) < 20 or len(h1_candles) < 20 or len(m30_candles) < 20 or len(m5_candles) < 20:
            return None, "RANGING"

        if session and session not in self.tradeable_sessions:
            return None, "RANGING"

        # Reset to base parameters before applying session-specific settings
        for param, value in self.strategy_config.items():
            if not isinstance(value, dict) and hasattr(self, param):
                setattr(self, param, value)
        
        # Apply Session-Specific Strategy individually
        session_data = self.session_cfg.get(session, {})
        overrides = session_data.get("strategy", {})
        for param, value in overrides.items():
            if hasattr(self, param):
                setattr(self, param, value)

        # 0. Choppy Mitigation Guards
        raw_ts = m5_candles[-1]['time']
        if isinstance(raw_ts, (int, float)):
            timestamp = datetime.fromtimestamp(raw_ts, tz=timezone.utc)
        elif isinstance(raw_ts, str):
            try:
                timestamp = datetime.fromisoformat(raw_ts)
            except ValueError:
                timestamp = datetime.fromtimestamp(int(raw_ts), tz=timezone.utc)
        else:
            timestamp = raw_ts # Assume datetime object
            
        current_date = timestamp.date()
        
        if self.last_loss_date != current_date:
            self.daily_losses = 0
            self.last_loss_date = current_date
            
        if self.daily_losses >= self.max_daily_losses:
            return None, "DAILY_LOSS_LIMIT"
            
        self.m5_trade_counter += 1
        if self.m5_trade_counter - self.last_m5_stop_index < self.cooldown_candles:
            return None, "COOLDOWN"

        # 1. MTF Trend & Momentum
        h4_trend, h4_strength = self._get_h4_trend(h4_candles)
        h1_closes = np.array([c["close"] for c in h1_candles])
        h1_ema20 = self._calculate_ema(h1_closes, 14)
        h1_trend = "BULLISH" if h1_closes[-1] > h1_ema20 else "BEARISH"
        h1_rsi = self._calculate_rsi(h1_closes, 14)
        
        # High Frequency Push: Allow H1 trend if H4 is Ranging
        effective_trend = h4_trend
        if h4_trend == "RANGING":
            effective_trend = h1_trend
            h4_strength = 20 # Minimum strength floor

        # 2. Volatility Check
        atr = self._calculate_atr(m30_candles)
        if self.vol_enabled:
            atr_lookback = m30_candles[-self.vol_lookback:] if len(m30_candles) > self.vol_lookback else m30_candles
            atr_avg = self._calculate_atr(atr_lookback)
            if atr > atr_avg * (self.vol_mult_high + 2.0) or atr < atr_avg * (self.vol_mult_low - 0.5):
                return None, h4_trend

        # 3. Regime Branching
        regime = MarketRegime.classify(m30_candles)
        m30_closes = np.array([c['close'] for c in m30_candles])
        m30_ema_val = self._calculate_ema(m30_closes, 10) 
        
        if abs(current_price - m30_ema_val) > atr * 5.0: # Extreme gap
            return None, h4_trend

        signal = None
        if regime == MarketRegime.TRENDING:
            if effective_trend == "BULLISH":
                if current_price > m30_ema_val and 30 < h1_rsi < 95:
                    signal = self._check_breakout_entry(m30_candles, m5_candles, "BULLISH", current_price)
            elif effective_trend == "BEARISH":
                if current_price < m30_ema_val and 5 < h1_rsi < 70:
                    signal = self._check_breakout_entry(m30_candles, m5_candles, "BEARISH", current_price)
        else: # Pullback
            if effective_trend == "BULLISH":
                signal = self._check_pullback_entry(m30_candles, m5_candles, "BULLISH", current_price)
            elif effective_trend == "BEARISH":
                signal = self._check_pullback_entry(m30_candles, m5_candles, "BEARISH", current_price)

        if signal:
            confluence, reasons = self._calculate_confluence(h4_trend, h4_strength, regime, signal, m30_candles, m5_candles)
            signal.confluence_score = confluence
            signal.reasons = reasons
            
            # AI Bias from context (passed through or read from config if available)
            ai_bias = self.config.get("ai_advisor", {}).get("bias", 0.0)
            
            signal = self._set_sl_tp(signal, atr, m30_candles, h4_strength, session, confluence, ai_bias)
            
            sl_dist = abs(signal.entry_price - signal.stop_loss)
            max_sl = 8.0 if "XAUUSD" in symbol else 0
            if max_sl > 0 and sl_dist > max_sl:
                return None, h4_trend
                
            if signal.confluence_score < self.min_confluence_score:
                return None, h4_trend

            signal.confidence = self._calculate_confidence(confluence, h4_strength, signal)
            
            if signal.confidence < self.min_confidence:
                return None, h4_trend
            
            self._log(f"PHASE 7 SIGNAL: {signal.direction} | Conf: {signal.confidence:.0f}% | PF Goal: 2.0+")
            
        return signal, h4_trend

    def _get_h4_trend(self, h4_candles: List[dict]) -> Tuple[str, int]:
        if not h4_candles or len(h4_candles) < 5: return "RANGING", 50
        closes = np.array([c["close"] for c in h4_candles])
        ema20 = self._calculate_ema_series(closes, self.ema_fast)
        ema50 = self._calculate_ema_series(closes, self.ema_slow)
        
        if len(ema20) < 2 or len(ema50) < 1: return "RANGING", 50
        
        score = 50 
        if closes[-1] > ema20[-1]: score += 15
        if ema20[-1] > ema50[-1]: score += 15
        if ema20[-1] > ema20[-2]: score += 10
        
        if closes[-1] < ema20[-1]: score -= 15
        if ema20[-1] < ema50[-1]: score -= 15
        if ema20[-1] < ema20[-2]: score -= 10
        
        if score > 55: return "BULLISH", score
        if score < 45: return "BEARISH", (100 - score)
        return "RANGING", 50

    def _check_breakout_entry(self, m30_candles: List[dict], m5_candles: List[dict], trend: str, current_price: float) -> Optional[TradeSignal]:
        # 1. Higher Timeframe (M30) Breakout
        lookback = self.swing_lookback
        recent_m30 = m30_candles[-lookback:]
        if trend == "BULLISH":
            res_m30 = max(c["high"] for c in recent_m30[:-1])
            if current_price > res_m30:
                return TradeSignal("BUY", current_price, 0, 0, reasons=["M30 Breakout"])
        else:
            sup_m30 = min(c["low"] for c in recent_m30[:-1])
            if current_price < sup_m30:
                return TradeSignal("SELL", current_price, 0, 0, reasons=["M30 Breakout"])
                
        # 2. Lower Timeframe (M5) Breakout (Hyper Frequency)
        m5_lookback = max(3, self.swing_lookback // 2)
        recent_m5 = m5_candles[-m5_lookback:]
        if trend == "BULLISH":
            res_m5 = max(c["high"] for c in recent_m5[:-1])
            if current_price > res_m5:
                m5_closes = np.array([c["close"] for c in m5_candles])
                m5_ema = self._calculate_ema(m5_closes, self.ema_fast)
                if current_price > m5_ema:
                    return TradeSignal("BUY", current_price, 0, 0, reasons=["M5 Breakout"])
        else:
            sup_m5 = min(c["low"] for c in recent_m5[:-1])
            if current_price < sup_m5:
                m5_closes = np.array([c["close"] for c in m5_candles])
                m5_ema = self._calculate_ema(m5_closes, self.ema_fast)
                if current_price < m5_ema:
                    return TradeSignal("SELL", current_price, 0, 0, reasons=["M5 Breakout"])
                
        return None

    def _check_pullback_entry(self, m30_candles: List[dict], m5_candles: List[dict], trend: str, current_price: float) -> Optional[TradeSignal]:
        closes = np.array([c["close"] for c in m30_candles])
        ema20 = self._calculate_ema(closes, 20)
        buffer = self.pullback_distance_pct / 100
        
        if trend == "BULLISH" and current_price > ema20 * 0.99:
            if m30_candles[-2]["low"] <= ema20 * (1 + buffer):
                return TradeSignal("BUY", current_price, 0, 0, rejection_type="PULLBACK")
        elif trend == "BEARISH" and current_price < ema20 * 1.01:
            if m30_candles[-2]["high"] >= ema20 * (1 - buffer):
                return TradeSignal("SELL", current_price, 0, 0, rejection_type="PULLBACK")
        return None

    def _set_sl_tp(self, signal: TradeSignal, atr: float, m30_candles: List[dict], 
                   h4_strength: int, session: str, confluence_score: int, ai_bias: float = 0.0) -> TradeSignal:
        recent = m30_candles[-3:]
        
        # 1. Base RR from Trend Strength
        if h4_strength > 70:
            rr = 3.0
        elif h4_strength > 55:
            rr = 2.5
        elif h4_strength == 50: # RANGING
            rr = 1.8
        else:
            rr = 2.0
            
        # 2. Volatility Adjustment (ATR Ratio)
        atr_avg = np.mean([abs(c['high'] - c['low']) for c in m30_candles[-20:]])
        vol_factor = atr / atr_avg if atr_avg > 0 else 1.0
        if vol_factor > 1.2:
            rr *= 1.2 # Widen in high vol
        elif vol_factor < 0.8:
            rr *= 0.8 # Narrow in low vol

        # 3. Session Adjustment
        if session == "LONDON/NY":
            rr += 0.5 # Higher potential during overlap
        elif session == "TOKYO":
            rr -= 0.3 # Lower potential in Tokyo
            
        # 4. AI Bias Adjustment
        rr += ai_bias # Direct adjustment from AI sentiment
        
        # 5. Confluence Adjustment
        if confluence_score >= 6:
            rr += 0.3
            
        rr = max(1.5, min(rr, 5.0)) # Clamp to reasonable bounds
            
        buffer = self.sl_atr_buffer * atr
        
        if signal.direction == "BUY":
            signal.stop_loss = min(c["low"] for c in recent) - buffer
            risk = signal.entry_price - signal.stop_loss
            signal.take_profit = signal.entry_price + (risk * rr)
            signal.tp1_price = signal.entry_price + (risk * 1.0)
            signal.tp2_price = signal.entry_price + (risk * 2.0)
            signal.tp3_price = signal.take_profit
        else:
            signal.stop_loss = max(c["high"] for c in recent) + buffer
            risk = signal.stop_loss - signal.entry_price
            signal.take_profit = signal.entry_price - (risk * rr)
            signal.tp1_price = signal.entry_price - (risk * 1.0)
            signal.tp2_price = signal.entry_price - (risk * 2.0)
            signal.tp3_price = signal.take_profit
            
        signal.rr_ratio = rr
        return signal

    def _calculate_confluence(self, h4_trend: str, h4_strength: int, regime: str, signal: TradeSignal,
                              m30_candles: List[dict], m5_candles: List[dict]) -> Tuple[int, List[str]]:
        reasons = []
        score = 0
        if (signal.direction == "BUY" and h4_trend == "BULLISH") or (signal.direction == "SELL" and h4_trend == "BEARISH"):
            score += 2; reasons.append("H4 Alignment")
        
        closes = np.array([c["close"] for c in m30_candles])
        h1_rsi = self._calculate_rsi(closes, 14)
        if (signal.direction == "BUY" and h1_rsi > 50) or (signal.direction == "SELL" and h1_rsi < 50):
            score += 1; reasons.append("H1 Momentum")
            
        current_hour = signal.timestamp.hour
        session = self.get_session_from_hour(current_hour)
        if session in ["LONDON", "LONDON/NY", "NEW_YORK"]:
            score += 1; reasons.append(f"{session} Session")
            
        atr = self._calculate_atr(m5_candles)
        if atr > self._calculate_atr(m30_candles) * 0.1:
            score += 1; reasons.append("Volatility Exp")
            
        return score, reasons

    def _calculate_confidence(self, confluence: int, h4_strength: int, signal: TradeSignal) -> float:
        base = 50.0  # Lowered base for more sensitivity in min_confidence (65-85)
        conf_bonus = confluence * 5.0
        strength_bonus = (h4_strength / 100) * 10
        return min(95.0, base + conf_bonus + strength_bonus)

    def report_trade_result(self, result: str, timestamp: datetime):
        """Called by bot/backtester to report trade exit results."""
        if result == "SL":
            self.daily_losses += 1
            self.last_m5_stop_index = self.m5_trade_counter
            self.last_stop_time = timestamp