"""
TRADING BOT V3 - HYBRID BREAKOUT STRATEGY (PHASE 7 HIGH-CONF)
Refined for XAUUSDm M5 with MTF Alignment, Weighted Confluence (70%+), and High Frequency (3+/day).
"""

import logging
import numpy as np
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple, Any, TYPE_CHECKING
from dataclasses import dataclass, field

if TYPE_CHECKING:
    from .types import CandleArray

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
        self.daily_trades = 0
        self.last_loss_date = None
        self.m5_trade_counter = 0 # To track cooldown in backtest candles
        self.last_m5_stop_index = -999

        # Sessions & Cooldown
        self.session_cfg = config.get("session_config", {})
        self.tradeable_sessions = {
            _SESSION_KEY_MAP[k] for k, v in self.session_cfg.items()
            if isinstance(v, dict) and v.get("enabled", False) and k in _SESSION_KEY_MAP
        } if self.session_cfg else _DEFAULT_SESSIONS
        
        self.consecutive_losses = {s: 0 for s in _DEFAULT_SESSIONS}
        self.session_cooldown_active = {s: False for s in _DEFAULT_SESSIONS}
        
        import threading
        self.lock = threading.Lock()

    @staticmethod
    def get_session_from_hour(hour: int, utc_offset: int = 0) -> str:
        """
        Determines session from candle hour.
        Adjusts for UTC offset (MT5 time -> UTC time) before classification.
        """
        utc_hour = (hour - utc_offset) % 24
        
        if 8 <= utc_hour < 14: return "LONDON"
        if 14 <= utc_hour < 17: return "LONDON/NY"
        if 17 <= utc_hour < 22: return "NEW_YORK"
        return "TOKYO"

    def _log(self, message: str, level: str = "INFO"):
        if self.silent: return
        if self.analysis_logger: self.analysis_logger.log(message, level)
        logger.info(message)

    def _calculate_rsi_series(self, prices: np.ndarray, period: int = 14) -> np.ndarray:
        """Vectorized RSI calculation for a series."""
        if len(prices) < period + 1:
            return np.full(len(prices), 50.0)

        deltas = np.diff(prices)
        ups = np.where(deltas > 0, deltas, 0)
        downs = np.where(deltas < 0, -deltas, 0)

        n = len(prices)
        avg_up = np.zeros(n)
        avg_down = np.zeros(n)

        avg_up[period] = np.mean(ups[:period])
        avg_down[period] = np.mean(downs[:period])

        for i in range(period + 1, n):
            avg_up[i] = (avg_up[i-1] * (period - 1) + ups[i-1]) / period
            avg_down[i] = (avg_down[i-1] * (period - 1) + downs[i-1]) / period

        rs = np.divide(avg_up, avg_down, out=np.zeros_like(avg_up), where=avg_down != 0)
        rsi = 100 - (100 / (1 + rs))
        rsi[:period] = 50.0 
        return rsi

    def _calculate_rsi(self, prices: np.ndarray, period: int = 14) -> float:
        series = self._calculate_rsi_series(prices, period)
        return float(series[-1])

    def _calculate_ema_series(self, prices: np.ndarray, period: int) -> np.ndarray:
        """High-performance pure NumPy EMA series."""
        if len(prices) == 0:
            return np.array([])
        if period <= 1:
            return prices

        alpha = 2.0 / (period + 1.0)
        n = len(prices)
        ema = np.empty(n)
        ema[0] = prices[0]
        
        for i in range(1, n):
            ema[i] = prices[i] * alpha + ema[i-1] * (1 - alpha)
        return ema

    def _calculate_ema(self, prices: np.ndarray, period: int) -> float:
        series = self._calculate_ema_series(prices, period)
        return series[-1] if len(series) > 0 else 0.0

    def _calculate_atr(self, candles: 'CandleArray', period: Optional[int] = None) -> float:
        if period is None:
            period = self.atr_period
            
        if not candles or len(candles) < 2: return 0.1
        highs = candles.high
        lows = candles.low
        closes = candles.close
        
        if len(highs) < 2: return 0.1

        tr = np.maximum(highs[1:] - lows[1:], 
                        np.maximum(np.abs(highs[1:] - closes[:-1]), 
                                   np.abs(lows[1:] - closes[:-1])))
        if len(tr) == 0: return 0.1
        return np.mean(tr[-period:]) if len(tr) >= period else np.mean(tr)

    def analyze(self, symbol: str, h4_candles: 'CandleArray', h1_candles: 'CandleArray', m30_candles: 'CandleArray',
                m5_candles: 'CandleArray', current_price: float,
                d1_candles: Optional['CandleArray'] = None, session: Optional[str] = None,
                preprocessed: Optional[dict] = None, circuit_breaker_safe: bool = True) -> Tuple[Optional[TradeSignal], str, str]:
        
        # 0. Daily Reset Logic
        raw_ts = m5_candles.time[-1]
        timestamp = datetime.fromtimestamp(float(raw_ts), tz=timezone.utc)
        current_date = timestamp.date()
        if self.last_loss_date != current_date:
            self.reset_daily_stats()
            self.last_loss_date = current_date

        # --- PHASE 11: BOOLEAN GATEKEEPERS (LOW-COST CHECKS) ---
        gate_status, gate_reason = self._check_gatekeepers(session, current_price, preprocessed, circuit_breaker_safe)
        if not gate_status:
            return None, gate_reason, "NEUTRAL"

        # Track M5 cooldown
        self.m5_trade_counter += 1
        if self.m5_trade_counter - self.last_m5_stop_index < self.cooldown_candles:
            return None, "COOLDOWN", "NEUTRAL"
            
        if preprocessed:
            # High-Performance Backtest Path
            effective_trend = preprocessed["h4_trend"]
            h1_rsi = preprocessed["h1_rsi"]
            m30_ema_val = preprocessed["m30_ema"]
            m30_atr = preprocessed["m30_atr"]
            m5_atr = preprocessed["m5_atr"]
            regime = preprocessed["regime"]
            h4_strength = preprocessed["h4_strength"]
            vol_scaling_flag = preprocessed["vol_scaling"]

            signal = None
            # --- PHASE 11: MODULE DISPATCHER (REGIME-ADAPTIVE) ---
            if session == "TOKYO":
                signal = self._tokyo_mean_reversion(m30_candles, m5_candles, current_price, m30_ema_val, h1_rsi)
            else:
                # London/NY Momentum Logic
                if effective_trend == "BULLISH" and current_price > m30_ema_val and 25 < h1_rsi < 95:
                    signal = self._check_breakout_entry(m30_candles, m5_candles, "BULLISH", current_price)
                elif effective_trend == "BEARISH" and current_price < m30_ema_val and 5 < h1_rsi < 75:
                    signal = self._check_breakout_entry(m30_candles, m5_candles, "BEARISH", current_price)
            
            if signal:
                signal.confluence_score, signal.reasons = self._calculate_confluence(
                    effective_trend, h4_strength, regime, signal, m30_candles, m5_candles, m30_atr, m5_atr, session
                )
                
                # PHASE 11: Session-Aware Confluence Thresholds
                session_threshold = self.session_cfg.get(session, {}).get("min_confluence_score", self.min_confluence_score)
                if signal.confluence_score < session_threshold: 
                    return None, effective_trend, str(regime)
                
                signal = self._set_sl_tp(signal, m30_atr, m30_candles, h4_strength, session, signal.confluence_score)
                signal.confidence = self._calculate_confidence(signal.confluence_score, h4_strength, signal)
            
            return signal, effective_trend, str(regime)
        
        if not h4_candles or not h1_candles or not m30_candles or not m5_candles:
            return None, "RANGING", "NEUTRAL"
        
        # Slicing via `__getitem__` logic 
        h4_candles = h4_candles[-200:]
        h1_candles = h1_candles[-200:]
        m30_candles = m30_candles[-200:]
        m5_candles = m5_candles[-400:] 
        
        if len(h4_candles) < 20 or len(h1_candles) < 20 or len(m30_candles) < 20 or len(m5_candles) < 20:
            return None, "RANGING", "NEUTRAL"

        if session and session not in self.tradeable_sessions:
            return None, "RANGING", "NEUTRAL"

        for param, value in self.strategy_config.items():
            if not isinstance(value, dict) and hasattr(self, param):
                setattr(self, param, value)
        
        session_data = self.session_cfg.get(session, {})
        overrides = session_data.get("strategy", {})
        for param, value in overrides.items():
            if hasattr(self, param):
                setattr(self, param, value)

        if self.m5_trade_counter - self.last_m5_stop_index < self.cooldown_candles:
            return None, "COOLDOWN", "NEUTRAL"

        # 1. Volatility Filter (ATR Ratio)
        m30_atr = self._calculate_atr(m30_candles)
        m5_atr = self._calculate_atr(m5_candles)
        
        highs = m30_candles.high
        lows = m30_candles.low
        closes = m30_candles.close
        tr = np.maximum(highs[1:] - lows[1:], 
                        np.maximum(np.abs(highs[1:] - closes[:-1]), 
                                   np.abs(lows[1:] - closes[:-1])))
        
        if len(tr) >= 20:
            sma_atr_20 = np.mean(tr[-20:])
        else:
            sma_atr_20 = m30_atr
            
        atr_ratio = m30_atr / sma_atr_20 if sma_atr_20 > 0 else 1.0
        
        vol_scaling_flag = False
        if atr_ratio < 0.6 or atr_ratio > 2.5:
            vol_scaling_flag = True 
            
        h4_trend, h4_strength = self._get_h4_trend(h4_candles)
        h1_closes = h1_candles.close
        h1_ema20 = self._calculate_ema(h1_closes, 14)
        h1_trend = "BULLISH" if h1_closes[-1] > h1_ema20 else "BEARISH"
        h1_rsi = self._calculate_rsi(h1_closes, 14)
        
        effective_trend = h4_trend
        if h4_trend == "RANGING":
            effective_trend = h1_trend
            h4_strength = 20

        regime = MarketRegime.classify(m30_candles)

        # 2. Institutional Volatility Envelope (Phase 4 Harden)
        # Using 100-period ATR SMA to define the "Normal" volatility regime
        atr_sma_100 = self._calculate_ema(m30_candles.high - m30_candles.low, 100)
        if m30_atr > atr_sma_100 * 2.5: # Volatility Spike (News/Chaotic)
            return None, "VOLATILITY_SPIKE", str(regime)
        if m30_atr < atr_sma_100 * 0.5: # Stagnation (Low Probability)
            return None, "STAGNATION", str(regime)

        # 3. LOW_LIQUIDITY Check (Exempt TOKYO as its baseline is low-vol/low-liq)
        if regime == MarketRegime.LOW_LIQUIDITY and session != "TOKYO":
            return None, "LOW_LIQUIDITY", str(regime)
            
        if not circuit_breaker_safe:
            return None, h4_trend, str(regime)

        # LIVE TRADING BRANCH
        if session == "TOKYO":
            # Tokyo session logic
            signal = self._tokyo_mean_reversion(m30_candles, m5_candles, current_price, m30_ema_val, h1_rsi)
        else:
            # London/NY Momentum Logic
            # [NEW] Phase 11 RSI Momentum filter specifically for New York to avoid churn
            if session == "NEW_YORK":
                if effective_trend == "BULLISH" and h1_rsi > 40:
                    signal = self._check_breakout_entry(m30_candles, m5_candles, "BULLISH", current_price)
                elif effective_trend == "BEARISH" and h1_rsi < 60:
                    signal = self._check_breakout_entry(m30_candles, m5_candles, "BEARISH", current_price)
            else:
                if effective_trend == "BULLISH" and current_price > m30_ema_val and 25 < h1_rsi < 95:
                    signal = self._check_breakout_entry(m30_candles, m5_candles, "BULLISH", current_price)
                elif effective_trend == "BEARISH" and current_price < m30_ema_val and 5 < h1_rsi < 75:
                    signal = self._check_breakout_entry(m30_candles, m5_candles, "BEARISH", current_price)

        if signal:
            confluence, reasons = self._calculate_confluence(h4_trend, h4_strength, regime, signal, 
                                                           m30_candles, m5_candles, m30_atr, m5_atr, session)
            signal.confluence_score = confluence
            signal.reasons = reasons
            
            ai_bias = self.config.get("ai_advisor", {}).get("bias", 0.0)
            signal = self._set_sl_tp(signal, m30_atr, m30_candles, h4_strength, session, confluence, ai_bias)
            
            if signal.confluence_score < self.min_confluence_score:
                return None, h4_trend, str(regime)

            signal.confidence = self._calculate_confidence(confluence, h4_strength, signal)
            if signal.confidence < self.min_confidence:
                return None, h4_trend, str(regime)
            
            self._log(f"PHASE 11 SIGNAL: {signal.direction} | Conf: {signal.confidence:.0f}%")
            
        return signal, h4_trend, str(regime)

    def _check_gatekeepers(self, session: str, current_price: float, preprocessed: Optional[dict] = None, circuit_breaker_safe: bool = True) -> Tuple[bool, str]:
        """
        Boolean Gatekeeper System (Ordered by computational cost).
        Signal must pass 100% of these for further processing.
        """
        # 1. MT5 Circuit Breaker Check (Cheapest)
        if not circuit_breaker_safe:
            return False, "CIRCUIT_BREAKER_TRIPPED"

        # 2. Session Enablement Check
        if session and session not in self.tradeable_sessions:
            return False, "SESSION_DISABLED"

        # 3. Session Cooldown Check
        if session and self.session_cooldown_active.get(session, False):
            return False, "SESSION_COOLDOWN"

        # 4. Spread Gatekeeper (Implemented in ExecutionPipeline/Backtester for real-time accuracy)
        # Note: In StrategyEngine, we assume the pipeline/backtester has already gated high spreads.

        return True, "OK"

    def _tokyo_mean_reversion(self, m30_candles: 'CandleArray', m5_candles: 'CandleArray', current_price: float, m30_ema: float, h1_rsi: float) -> Optional[TradeSignal]:
        """
        Specialized Module for Tokyo Liquidity Cycles.
        Focuses on Mean-Reversion during low-volatility hours.
        """
        # Use Bollinger Band style excursion for mean-reversion
        m30_closes = m30_candles.close
        if len(m30_closes) < 20: return None
        
        m30_std = np.std(m30_closes[-20:])
        upper_band = m30_ema + (1.5 * m30_std) # [ALPHA-MAX] Relaxed to 1.5 for Tokyo liquidity
        lower_band = m30_ema - (1.5 * m30_std)
        
        # Mean Reversion: Buy when oversold and price below lower band returning to EMA
        # [RELAXED] RSI from 30/70 to 40/60 for Tokyo quiet hours
        if current_price < lower_band and h1_rsi < 40:
            if m5_candles.close[-1] > m5_candles.open[-1]: # Basic Bullish Confirmation
                return TradeSignal("BUY", current_price, 0, 0, reasons=["Tokyo Mean-Rev"])
        
        # Sell when overbought and price above upper band returning to EMA
        if current_price > upper_band and h1_rsi > 60:
            if m5_candles.close[-1] < m5_candles.open[-1]: # Basic Bearish Confirmation
                return TradeSignal("SELL", current_price, 0, 0, reasons=["Tokyo Mean-Rev"])
                
        return None

    def _get_h4_trend(self, h4_candles: 'CandleArray') -> Tuple[str, int]:
        if not h4_candles or len(h4_candles) < 5: return "RANGING", 50
        closes = h4_candles.close
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

    def _check_breakout_entry(self, m30_candles: 'CandleArray', m5_candles: 'CandleArray', trend: str, current_price: float) -> Optional[TradeSignal]:
        # 1. Institutional Body Filter (Phase 5)
        # We only trade candles that show "intent" (long bodies, short wicks)
        m30_last_body = abs(m30_candles.close[-1] - m30_candles.open[-1])
        m30_last_range = m30_candles.high[-1] - m30_candles.low[-1]
        body_pct = (m30_last_body / m30_last_range) if m30_last_range > 0 else 0
        
        if body_pct < 0.40: # Reject indecision/doji candles
            return None

        # 2. Volatility Expansion Filter (Phase 5)
        # Only enter if current ATR is expanding beyond the 100-period average
        m30_atr = self._calculate_atr(m30_candles)
        atr_sma_100 = self._calculate_ema(m30_candles.high - m30_candles.low, 100)
        if m30_atr < atr_sma_100:
            return None

        # 3. Higher Timeframe (M30) Breakout
        lookback = self.swing_lookback
        if trend == "BULLISH":
            res_m30 = np.max(m30_candles.high[-lookback:-1])
            if current_price > res_m30:
                return TradeSignal("BUY", current_price, 0, 0, reasons=["M30 Breakout"])
        else:
            sup_m30 = np.min(m30_candles.low[-lookback:-1])
            if current_price < sup_m30:
                return TradeSignal("SELL", current_price, 0, 0, reasons=["M30 Breakout"])
                
        # 4. Lower Timeframe (M5) Breakout (Hyper Frequency)
        m5_lookback = 5 # [FIX]: Increased from 3 to 5 to filter out micro-fakeouts
        if trend == "BULLISH":
            res_m5 = np.max(m5_candles.high[-m5_lookback:-1])
            if current_price > res_m5:
                m5_closes = m5_candles.close
                m5_ema = self._calculate_ema(m5_closes, self.ema_fast)
                if current_price > m5_ema:
                    return TradeSignal("BUY", current_price, 0, 0, reasons=["M5 Breakout"])
        else:
            sup_m5 = np.min(m5_candles.low[-m5_lookback:-1])
            if current_price < sup_m5:
                m5_closes = m5_candles.close
                m5_ema = self._calculate_ema(m5_closes, self.ema_fast)
                if current_price < m5_ema:
                    return TradeSignal("SELL", current_price, 0, 0, reasons=["M5 Breakout"])

                
        return None

    def _check_pullback_entry(self, m30_candles: 'CandleArray', m5_candles: 'CandleArray', trend: str, current_price: float) -> Optional[TradeSignal]:
        closes = m30_candles.close
        ema20 = self._calculate_ema(closes, 20)
        buffer = self.pullback_distance_pct / 100
        
        if trend == "BULLISH" and current_price > ema20 * 0.99:
            if m30_candles.low[-2] <= ema20 * (1 + buffer):
                return TradeSignal("BUY", current_price, 0, 0, rejection_type="PULLBACK")
        elif trend == "BEARISH" and current_price < ema20 * 1.01:
            if m30_candles.high[-2] >= ema20 * (1 - buffer):
                return TradeSignal("SELL", current_price, 0, 0, rejection_type="PULLBACK")
        return None

    def _set_sl_tp(self, signal: TradeSignal, atr: float, m30_candles: 'CandleArray', 
                   h4_strength: int, session: str, confluence_score: int, ai_bias: float = 0.0) -> TradeSignal:
        
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
        atr_avg = np.mean(m30_candles.high[-20:] - m30_candles.low[-20:])
        vol_factor = atr / atr_avg if atr_avg > 0 else 1.0
        if vol_factor > 1.2:
            rr *= 1.2 
        elif vol_factor < 0.8:
            rr *= 0.8 

        # 3. Session Adjustment
        if session == "LONDON/NY":
            rr += 0.5 
        elif session == "TOKYO":
            rr -= 0.3 
            
        # 4. AI Bias Adjustment
        rr += ai_bias
        
        # 5. Confluence Adjustment
        if confluence_score >= 6:
            rr += 0.3
            
        rr = max(1.5, min(rr, 5.0))
        buffer = self.sl_atr_buffer * atr
        
        if signal.direction == "BUY":
            signal.stop_loss = np.min(m30_candles.low[-3:]) - buffer
            risk = signal.entry_price - signal.stop_loss
            signal.take_profit = signal.entry_price + (risk * rr)
            signal.tp1_price = signal.entry_price + (risk * 1.0)
            signal.tp2_price = signal.entry_price + (risk * 2.0)
            signal.tp3_price = signal.take_profit
        else:
            signal.stop_loss = np.max(m30_candles.high[-3:]) + buffer
            risk = signal.stop_loss - signal.entry_price
            signal.take_profit = signal.entry_price - (risk * rr)
            signal.tp1_price = signal.entry_price - (risk * 1.0)
            signal.tp2_price = signal.entry_price - (risk * 2.0)
            signal.tp3_price = signal.take_profit
            
        signal.rr_ratio = rr
        return signal

    def _calculate_confluence(self, h4_trend: str, h4_strength: int, regime: str, signal: TradeSignal,
                              m30_candles: 'CandleArray', m5_candles: 'CandleArray',
                              m30_atr: float, m5_atr: float, session: str) -> Tuple[int, List[str]]:
        reasons = []
        score = 0
        if (signal.direction == "BUY" and h4_trend == "BULLISH") or (signal.direction == "SELL" and h4_trend == "BEARISH"):
            score += 2; reasons.append("H4 Alignment")
        
        closes = m30_candles.close
        h1_rsi = self._calculate_rsi(closes, 14)
        if (signal.direction == "BUY" and h1_rsi > 50) or (signal.direction == "SELL" and h1_rsi < 50):
            score += 1; reasons.append("H1 Momentum")
            
        if session and session in ["LONDON", "LONDON/NY", "NEW_YORK", "TOKYO"]:
            score += 1; reasons.append(f"{session} Session")
            
        if m5_atr > m30_atr * 0.1:
            score += 1; reasons.append("Volatility Exp")
            
        m5_closes = m5_candles.close
        m5_ema20 = self._calculate_ema_series(m5_closes, 20)
        if len(m5_ema20) > 2:
            if signal.direction == "BUY" and m5_ema20[-1] > m5_ema20[-2]:
                score += 1; reasons.append("M5 EMA Slope")
            elif signal.direction == "SELL" and m5_ema20[-1] < m5_ema20[-2]:
                score += 1; reasons.append("M5 EMA Slope")
            
        return score, reasons

    def _calculate_confidence(self, confluence: int, h4_strength: int, signal: TradeSignal) -> float:
        base = 50.0 
        conf_bonus = confluence * 5.0
        strength_bonus = (h4_strength / 100) * 10
        return min(95.0, base + conf_bonus + strength_bonus)

    def report_trade_result(self, result: str, timestamp: datetime, session: Optional[str] = None):
        """Called by bot/backtester to report trade exit results. Thread-safe."""
        with self.lock:
            self.daily_trades += 1
            if result == "SL":
                self.daily_losses += 1
                self.last_m5_stop_index = self.m5_trade_counter
                self.last_stop_time = timestamp
                
                if session:
                    self.consecutive_losses[session] = self.consecutive_losses.get(session, 0) + 1
                    if self.consecutive_losses[session] >= 2:
                        self.session_cooldown_active[session] = True
            elif result == "TP":
                if session:
                    self.consecutive_losses[session] = 0
                    self.session_cooldown_active[session] = False

    def reset_daily_stats(self):
        """Reset tracking for daily trade limits and session losses."""
        with self.lock:
            self.daily_losses = 0
            self.daily_trades = 0
            for s in self.consecutive_losses:
                self.consecutive_losses[s] = 0
                self.session_cooldown_active[s] = False

    def preprocess_history(self, h4: 'CandleArray', h1: 'CandleArray', m30: 'CandleArray', m5: 'CandleArray') -> dict:
        """
        Pre-calculates all technical indicators and trends for full history speed.
        Returns a dictionary of mapped states for each M5 candle.
        """
        import time as timer
        import logging
        logger = logging.getLogger("trading_bot.strategy")
        
        start = timer.time()
        logger.info("[Strategy] Vectorized Preprocessing started...")

        # 1. Indicator Pre-calculations
        h4_closes = h4.close
        h1_closes = h1.close
        m30_closes = m30.close
        m5_closes = m5.close
        
        m30_ema = self._calculate_ema_series(m30_closes, 10)
        h1_ema20 = self._calculate_ema_series(h1_closes, 14)
        h1_rsi = self._calculate_rsi_series(h1_closes, 14)
        
        # 2. Time-based Mappings
        m5_times = m5.time
        h4_times = h4.time
        h1_times = h1.time
        m30_times = m30.time
        
        h4_trend_data = []
        for i in range(len(h4)):
            slice_h4 = h4[:i+1]
            if len(slice_h4) < 20:
                h4_trend_data.append(("RANGING", 0))
            else:
                h4_trend_data.append(self._get_h4_trend(slice_h4))
        
        m30_regime_data = []
        m30_atr_data = []
        sma_atr_20_data = []
        
        h30 = m30.high
        l30 = m30.low
        c30 = m30.close
        tr30 = np.zeros_like(c30)
        tr30[1:] = np.maximum(h30[1:] - l30[1:], 
                              np.maximum(np.abs(h30[1:] - c30[:-1]), 
                                         np.abs(l30[1:] - c30[:-1])))
        
        h5 = m5.high
        l5 = m5.low
        c5 = m5.close
        tr5 = np.zeros_like(c5)
        tr5[1:] = np.maximum(h5[1:] - l5[1:], 
                             np.maximum(np.abs(h5[1:] - c5[:-1]), 
                                        np.abs(l5[1:] - c5[:-1])))
        m5_atrs = np.zeros_like(c5)
        for i in range(len(m5)):
            m5_atrs[i] = np.mean(tr5[max(0, i-13):i+1]) if i > 0 else 0.1
        
        for i in range(len(m30)):
            slice_m30 = m30[:i+1]
            reg = MarketRegime.classify(slice_m30)
            m30_regime_data.append(reg)
            
            atr = np.mean(tr30[max(0, i-13):i+1]) if i > 0 else 0.1
            m30_atr_data.append(atr)
            sma_atr_20_data.append(np.mean(tr30[max(0, i-19):i+1]) if i > 0 else atr)

        # 3. Final M5 Mapping
        m5_precomputed = []
        for i in range(len(m5)):
            t = m5_times[i]
            
            # [FIX]: Subtract timeframe duration to prevent Look-ahead Bias
            h4_idx = np.searchsorted(h4_times, t - 14400, side='right') - 1
            h1_idx = np.searchsorted(h1_times, t - 3600, side='right') - 1
            m30_idx = np.searchsorted(m30_times, t - 1800, side='right') - 1
            
            h4_idx = max(0, h4_idx)
            h1_idx = max(0, h1_idx)
            m30_idx = max(0, m30_idx)

            h1_t = "BULLISH" if h1_closes[h1_idx] > h1_ema20[h1_idx] else "BEARISH"
            h4_t, h4_s = h4_trend_data[h4_idx]
            
            effective_trend = h4_t
            if h4_t == "RANGING":
                effective_trend = h1_t
                h4_s = 20
                
            atr_ratio = m30_atr_data[m30_idx] / sma_atr_20_data[m30_idx] if sma_atr_20_data[m30_idx] > 0 else 1.0
            
            m5_precomputed.append({
                "h4_trend": effective_trend,
                "h4_strength": h4_s,
                "regime": m30_regime_data[m30_idx],
                "h1_rsi": h1_rsi[h1_idx],
                "m30_ema": m30_ema[m30_idx],
                "m30_atr": m30_atr_data[m30_idx],
                "sma_atr_20": sma_atr_20_data[m30_idx],
                "vol_scaling": (atr_ratio < 0.6 or atr_ratio > 2.5),
                "m5_atr": m5_atrs[i]
            })
            
        logger.info("[Strategy] Preprocessing complete in %.2fs", timer.time() - start)
        return {"m5": m5_precomputed}