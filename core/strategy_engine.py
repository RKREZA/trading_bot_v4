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
    session: str = "GLOBAL" # [PHASE 7] Tracking for session-specific exits
    tp1_price: float = 0.0
    tp2_price: float = 0.0
    tp3_price: float = 0.0
    confidence: float = 0.0
    confluence_score: int = 0
    reasons: List[str] = field(default_factory=list)
    rejection_type: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    rr_ratio: float = 2.0
    volatility_spike: bool = False

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
        self.sl_atr_buffer = self.strategy_config.get("sl_atr_buffer", 1.5) 
        self.rr_ratio = self.strategy_config.get("rr_ratio", 2.5) 

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

    def _calculate_sma_series(self, prices: np.ndarray, period: int) -> np.ndarray:
        """Vectorized SMA series."""
        n = len(prices)
        if n < period:
            return np.full(n, np.mean(prices) if n > 0 else 0.0)
        ret = np.cumsum(prices, dtype=float)
        ret[period:] = ret[period:] - ret[:-period]
        res = np.zeros(n)
        res[period-1:] = ret[period-1:] / period
        res[:period-1] = res[period-1] # Pad start
        return res

    def _calculate_macd_series(self, prices: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Vectorized MACD calculation (MACD, Signal, Histogram)."""
        n = len(prices)
        if n < slow + signal:
            return np.zeros(n), np.zeros(n), np.zeros(n)
        
        ema_fast = self._calculate_ema_series(prices, fast)
        ema_slow = self._calculate_ema_series(prices, slow)
        macd = ema_fast - ema_slow
        macd_signal = self._calculate_ema_series(macd, signal)
        histogram = macd - macd_signal
        return macd, macd_signal, histogram

    def _calculate_vw_rsi_series(self, prices: np.ndarray, volumes: np.ndarray, period: int = 14) -> np.ndarray:
        """Volume-Weighted RSI (using PriceChange * Volume)."""
        n = len(prices)
        if n < period + 1:
            return np.full(n, 50.0)
            
        deltas = np.diff(prices)
        # Use common length for volumes and deltas
        v_deltas = deltas * volumes[1:] 
        
        ups = np.where(v_deltas > 0, v_deltas, 0)
        downs = np.where(v_deltas < 0, -v_deltas, 0)
        
        avg_up = np.zeros(n)
        avg_down = np.zeros(n)
        
        avg_up[period] = np.mean(ups[:period])
        avg_down[period] = np.mean(downs[:period])
        
        for i in range(period + 1, n):
            avg_up[i] = (avg_up[i-1] * (period - 1) + ups[i-1]) / period
            avg_down[i] = (avg_down[i-1] * (period - 1) + downs[i-1]) / period
            
        rs = np.divide(avg_up, avg_down, out=np.zeros_like(avg_up), where=avg_down != 0)
        vwrsi = 100 - (100 / (1 + rs))
        vwrsi[:period] = 50.0
        return vwrsi

    def _calculate_efficiency_ratio_series(self, prices: np.ndarray, period: int = 14) -> np.ndarray:
        """Kaufman's Efficiency Ratio series."""
        n = len(prices)
        er = np.zeros(n)
        if n <= period: return er
        
        # Net change over 'period' bars
        net_change = np.abs(prices[period:] - prices[:-period])
        # Sum of absolute 1-bar changes over 'period' bars
        abs_diff = np.abs(np.diff(prices))
        sum_abs_diff = np.zeros(n)
        # Rolling sum of abs_diff
        window_sum = np.convolve(abs_diff, np.ones(period), mode='valid')
        sum_abs_diff[period:] = window_sum
        
        er[period:] = np.divide(net_change, sum_abs_diff[period:], out=np.zeros_like(net_change), where=sum_abs_diff[period:] > 0)
        return er

    def _calculate_adx_series(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
        """Vectorized ADX series using Wilder's smoothing."""
        n = len(high)
        if n < period * 2: return np.zeros(n)
        
        tr = np.maximum(high[1:] - low[1:], 
                        np.maximum(np.abs(high[1:] - close[:-1]), 
                                   np.abs(low[1:] - close[:-1])))
        
        up_move = high[1:] - high[:-1]
        down_move = low[:-1] - low[1:]
        
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        
        def wilders_smoothing(vals, p):
            res = np.zeros(len(vals) + 1)
            res[p] = np.mean(vals[:p])
            for i in range(p + 1, len(res)):
                res[i] = (res[i-1] * (p - 1) + vals[i-1]) / p
            return res
            
        atr = wilders_smoothing(tr, period)
        p_dm_s = wilders_smoothing(plus_dm, period)
        m_dm_s = wilders_smoothing(minus_dm, period)
        
        plus_di = 100 * p_dm_s / np.maximum(atr, 0.00001)
        minus_di = 100 * m_dm_s / np.maximum(atr, 0.00001)
        
        dx = 100 * np.abs(plus_di - minus_di) / np.maximum(plus_di + minus_di, 0.00001)
        adx_s = wilders_smoothing(dx[period:], period)
        
        final_adx = np.zeros(n)
        # Offset alignment
        final_adx[period*2 - 1:] = adx_s[period:]
        return final_adx

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

    def analyze(self, symbol: str, h4_candles: 'CandleArray', m30_candles: 'CandleArray', 
                m15_candles: 'CandleArray', m5_candles: 'CandleArray', current_price: float,
                d1_candles: Optional['CandleArray'] = None, session: Optional[str] = None,
                preprocessed: Optional[dict] = None, circuit_breaker_safe: bool = True) -> Tuple[Optional[TradeSignal], str, str]:
        """
        [PHASE 11] Institutional Multi-Timeframe Analysis.
        H4 (Anchor) -> M30 (Momentum) -> M15 (Execution) -> M5 (Trigger).
        """
        # 0. Daily Reset Logic
        raw_ts = m5_candles.time[-1]
        timestamp = datetime.fromtimestamp(float(raw_ts), tz=timezone.utc)
        current_date = timestamp.date()
        if self.last_loss_date != current_date:
            self.reset_daily_stats()
            self.last_loss_date = current_date
            
        # --- GATEKEEPERS ---
        gate_status, gate_reason = self._check_gatekeepers(session, current_price, preprocessed, circuit_breaker_safe)
        if not gate_status:
            return None, gate_reason, "NEUTRAL"

        self.m5_trade_counter += 1
        if self.m5_trade_counter - self.last_m5_stop_index < self.cooldown_candles:
            return None, "COOLDOWN", "NEUTRAL"

        if preprocessed:
            # Vectorized Backtest Optimization Path
            return self._analyze_preprocessed(preprocessed, m30_candles, m15_candles, m5_candles, current_price, session)

        # LIVE ANALYSIS PATH
        if not all([h4_candles, m30_candles, m15_candles, m5_candles]):
            return None, "MISSING_DATA", "NEUTRAL"

        # 1. H4 Anchor Conviction
        regime = MarketRegime.classify(m30_candles)
        h4_trend, h4_conviction = self._get_h4_trend(h4_candles)
        h4_atr = self._calculate_atr(h4_candles)
        
        # 2. Mode Selection (Adaptive)
        # If H4 is Ranging or Conviction is weak, default to Mean-Reversion
        is_ranging = (h4_trend == "RANGING" or h4_conviction < 40)
        
        if is_ranging:
            # [PHASE 13.4] Disable TOKYO Mean-Reversion to focus on high-conviction trends
            return None, "RANGING_FILTER", "NEUTRAL"


        # 3. M30 Momentum Quality Layer
        m30_closes = m30_candles.close
        m30_vols = m30_candles.tick_volume
        
        # MACD Velocity (Delta)
        _, _, hist = self._calculate_macd_series(m30_closes)
        macd_vel_up = hist[-1] > hist[-2]
        macd_vel_down = hist[-1] < hist[-2]
        
        # Efficiency Ratio (Kaufman's ER)
        er = self._calculate_efficiency_ratio_series(m30_closes)
        m30_er = er[-1]
        
        # Volume-Weighted RSI
        vw_rsi = self._calculate_vw_rsi_series(m30_closes, m30_vols)
        current_vw_rsi = vw_rsi[-1]
        
        # Momentum Quality Gate
        # Only proceed if ER > 0.35 and MACD Delta aligns with trend
        if m30_er < 0.35:
            return None, "LOW_EFFICIENCY", str(regime)
            
        if h4_trend == "BULLISH" and not macd_vel_up:
            return None, "MOMENTUM_DECAY", str(regime)
        if h4_trend == "BEARISH" and not macd_vel_down:
            return None, "MOMENTUM_DECAY", str(regime)
            
        # VW-RSI Filter
        if h4_trend == "BULLISH" and current_vw_rsi < 45: # Requires some conviction
            return None, "LOW_VOL_CONVICTION", str(regime)
        if h4_trend == "BEARISH" and current_vw_rsi > 55:
            return None, "LOW_VOL_CONVICTION", str(regime)

        # 4. Execution & Trigger Layer
        signal = self._check_breakout_entry(m15_candles, m5_candles, h4_trend, current_price, session, h4_atr)
        
        # [PHASE 12] Dual-Trigger System: If no breakout, check for pullback
        if not signal:
            signal = self._check_pullback_entry(m15_candles, m5_candles, h4_trend, current_price, session)

        if signal:
            # 5. Risk & Final Confluence
            # Weighted Score based on multiple metrics
            confluence, reasons = self._calculate_confluence(h4_trend, h4_conviction, regime, signal, 
                                                           m30_candles, m5_candles, 0, 0, session)
            signal.confluence_score = confluence
            signal.reasons = reasons
            
            # High Conviction RR Boost
            ai_bias = self.config.get("ai_advisor", {}).get("bias", 0.0)
            if h4_conviction > 75:
                ai_bias += 0.5 # Institutional conviction boost
                
            signal = self._set_sl_tp(signal, self._calculate_atr(m15_candles), m30_candles, h4_conviction, session, confluence, ai_bias, m5_candles)
            
            # Final confidence mapping
            signal.confidence = self._calculate_confidence(confluence, h4_conviction, signal)
            
            # Gating
            session_conf = self.config.get("session_config", {}).get(session, {})
            min_conf = session_conf.get("min_confluence_score", self.min_confluence_score)
            
            if signal.confluence_score < min_conf:
                return None, h4_trend, str(regime)

            self._log(f"PHASE 11 SIGNAL: {signal.direction} | Conv: {h4_conviction} | Conf: {signal.confidence:.0f}%")
            
        return signal, h4_trend, str(regime)

    def _analyze_preprocessed(self, preprocessed: dict, m30_candles: 'CandleArray', m15_candles: 'CandleArray', 
                              m5_candles: 'CandleArray', current_price: float, session: str) -> Tuple[Optional[TradeSignal], str, str]:
        """[PHASE 13.3] Optimized Backtest Path. Synchronized with live logic."""
        h4_trend = preprocessed["h4_trend"]
        h4_conviction = preprocessed["h4_conviction"]
        h4_atr = preprocessed["h4_atr"]
        regime = preprocessed["regime"]
        
        # 1. Gatekeepers (Simplified for speed in backtest)
        if session and session not in self.tradeable_sessions:
            return None, "SESSION_DISABLED", "NEUTRAL"
            
        is_ranging = (h4_trend == "RANGING" or h4_conviction < 40)
        if is_ranging:
            return None, "RANGING_FILTER", "NEUTRAL"


        # 2. Momentum Quality
        m30_er = preprocessed["m30_er"]
        if m30_er < 0.35: return None, "LOW_EFFICIENCY", str(regime)
        
        # 3. Execution Layer
        signal = self._check_breakout_entry(m15_candles, m5_candles, h4_trend, current_price, session, h4_atr)
        if not signal:
            signal = self._check_pullback_entry(m15_candles, m5_candles, h4_trend, current_price, session)

        if signal:
            # 4. Confluence & Scaling
            confluence, reasons = self._calculate_confluence(h4_trend, h4_conviction, regime, signal, 
                                                           m30_candles, m5_candles, 0, 0, session)
            signal.confluence_score = confluence
            signal.reasons = reasons
            
            ai_bias = self.config.get("ai_advisor", {}).get("bias", 0.0)
            if h4_conviction > 75: ai_bias += 0.5
            
            # [MICRO-STOP] Pass m5_candles to enable tighter Stops
            signal = self._set_sl_tp(signal, preprocessed["m15_atr"], m30_candles, h4_conviction, session, confluence, ai_bias, m5_candles)
            signal.confidence = self._calculate_confidence(confluence, h4_conviction, signal)
            
            # Gating
            session_conf = self.config.get("session_config", {}).get(session, {})
            min_conf = session_conf.get("min_confluence_score", self.min_confluence_score)
            if signal.confluence_score < min_conf:
                return None, h4_trend, str(regime)
            
        return signal, h4_trend, str(regime)


        if signal:
            confluence, reasons = self._calculate_confluence(h4_trend, h4_strength, regime, signal, 
                                                           m30_candles, m5_candles, m30_atr, m5_atr, session)
            signal.confluence_score = confluence
            signal.reasons = reasons
            
            ai_bias = self.config.get("ai_advisor", {}).get("bias", 0.0)
            signal = self._set_sl_tp(signal, m30_atr, m30_candles, h4_strength, session, confluence, ai_bias)
            
            # [PHASE 7] Session-aware Confluence Check
            session_conf = self.config.get("session_config", {}).get(session, {})
            min_conf = session_conf.get("min_confluence_score", self.min_confluence_score)
            
            if signal.confluence_score < min_conf:
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

    def _tokyo_mean_reversion(self, m30_candles: 'CandleArray', m5_candles: 'CandleArray', current_price: float, m30_ema: float, h1_rsi: float, session: str) -> Optional[TradeSignal]:
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
                return TradeSignal("BUY", current_price, 0, 0, reasons=["Tokyo Mean-Rev"], session=session)
        
        # Sell when overbought and price above upper band returning to EMA
        if current_price > upper_band and h1_rsi > 60:
            if m5_candles.close[-1] < m5_candles.open[-1]: # Basic Bearish Confirmation
                return TradeSignal("SELL", current_price, 0, 0, reasons=["Tokyo Mean-Rev"], session=session)
                
        return None

    def _get_h4_trend(self, h4_candles: 'CandleArray') -> Tuple[str, int]:
        """
        [PHASE 11] Institutional Anchor Logic.
        Returns (Trend, Conviction Score 0-100).
        """
        if not h4_candles or len(h4_candles) < 30: return "RANGING", 0
        
        closes = h4_candles.close
        highs = h4_candles.high
        lows = h4_candles.low
        
        ema20 = self._calculate_ema_series(closes, self.ema_fast)
        ema50 = self._calculate_ema_series(closes, self.ema_slow)
        adx = self._calculate_adx_series(highs, lows, closes, 14)
        
        current_adx = adx[-1]
        
        # 1. Ranging Check (ADX < 20)
        if current_adx < 20:
            return "RANGING", int(current_adx * 2) # Low conviction for trending
            
        score = 0
        trend = "RANGING"
        
        # 2. EMA Three-Factor (30 pts)
        ema_score = 0
        if ema20[-1] > ema50[-1]: # Alignment
            ema_score += 10
        if closes[-1] > ema20[-1] and closes[-1] > ema50[-1]: # Location
            ema_score += 10
        if ema50[-1] > ema50[-4]: # Slope (4-candle lookback)
            ema_score += 10
            
        # Bearish version
        ema_score_bear = 0
        if ema20[-1] < ema50[-1]:
            ema_score_bear += 10
        if closes[-1] < ema20[-1] and closes[-1] < ema50[-1]:
            ema_score_bear += 10
        if ema50[-1] < ema50[-4]:
            ema_score_bear += 10
            
        # 3. ADX Intensity (30 pts)
        adx_score = 0
        if current_adx > 25:
            adx_score = 30
        elif current_adx > 20:
            adx_score = 15
            
        # 4. Structural Market Geometry (40 pts)
        struct_score = 0
        # Bullish Structure: Higher Highs
        recent_highs = highs[-20:]
        curr_sh = np.max(recent_highs[-5:])
        prev_sh = np.max(recent_highs[-15:-5])
        if curr_sh > prev_sh:
            struct_score = 40
            
        # Bearish Structure: Lower Lows
        recent_lows = lows[-20:]
        curr_sl = np.min(recent_lows[-5:])
        prev_sl = np.min(recent_lows[-15:-5])
        struct_score_bear = 0
        if curr_sl < prev_sl:
            struct_score_bear = 40
            
        bull_total = ema_score + adx_score + struct_score
        bear_total = ema_score_bear + adx_score + struct_score_bear
        
        if bull_total > bear_total and bull_total > 30: # [ALPHA-EXPANSION] 40 -> 30
            return "BULLISH", bull_total
        elif bear_total > bull_total and bear_total > 30:
            return "BEARISH", bear_total
            
        return "RANGING", 50

    def _check_breakout_entry(self, m15_candles: 'CandleArray', m5_candles: 'CandleArray', 
                              trend: str, current_price: float, session: str, h4_atr: float) -> Optional[TradeSignal]:
        """
        [PHASE 12] Adaptive Execution Logic.
        Replaces 'Kill-Filters' with 'Confluence Scoring' to improve trade frequency.
        """
        exec_score = 0
        reasons = []

        # 1. M15 Execution Layer (Weighted)
        m15_atr = self._calculate_atr(m15_candles)
        m15_vols = m15_candles.tick_volume
        m15_adx = self._calculate_adx_series(m15_candles.high, m15_candles.low, m15_candles.close, 14)
        
        # M15 RVOL (2 pts)
        m15_vol_sma20 = np.mean(m15_vols[-20:])
        if m15_vols[-1] >= m15_vol_sma20:
            exec_score += 2
        elif m15_vols[-1] >= m15_vol_sma20 * 0.8:
            exec_score += 1 # Borderline volume OK
            
        # M15 ADX Slope (2 pts)
        adx_diff = m15_adx[-1] - m15_adx[-2]
        if adx_diff > 0:
            exec_score += 2
        elif abs(adx_diff) < (m15_adx[-1] * 0.05): # Flat ADX is acceptable
            exec_score += 1
            
        vol_spike = m15_atr > (h4_atr * 2.0)
        
        # 2. M5 Trigger Layer (Weighted)
        m5_body = abs(m5_candles.close[-1] - m5_candles.open[-1])
        m5_range = m5_candles.high[-1] - m5_candles.low[-1]
        body_ratio = (m5_body / m5_range) if m5_range > 0 else 0
        
        if body_ratio >= 0.60:
            exec_score += 3
        elif body_ratio >= 0.45:
            exec_score += 2 # Relaxed ratio for Gold volatility
        elif body_ratio >= 0.30:
            exec_score += 1
            
        # Spring Effect (1 pt)
        tr5 = np.maximum(m5_candles.high[1:] - m5_candles.low[1:], 
                         np.maximum(np.abs(m5_candles.high[1:] - m5_candles.close[:-1]), 
                                    np.abs(m5_candles.low[1:] - m5_candles.close[:-1])))
        m5_atr_tight = np.mean(tr5[-3:])
        m5_atr_baseline = np.mean(tr5[-20:])
        if m5_atr_tight <= m5_atr_baseline:
            exec_score += 1
            
        # Fuel Check (2 pts)
        m5_vols = m5_candles.tick_volume
        m5_vol_sma10 = np.mean(m5_vols[-11:-1])
        if m5_vols[-1] >= (m5_vol_sma10 * 1.5):
            exec_score += 2
        elif m5_vols[-1] >= (m5_vol_sma10 * 1.1):
            exec_score += 1

        # GATING: Require at least 5/10 Performance Score for Breakout
        if exec_score < 5:
            return None 

        # 3. Final Breakout Structure Logic
        lookback = self.swing_lookback 
        if trend == "BULLISH":
            res_m15 = np.max(m15_candles.high[-lookback:-1])
            if current_price > res_m15:
                m5_ema = self._calculate_ema(m5_candles.close, self.ema_fast)
                if current_price > m5_ema:
                    signal = TradeSignal("BUY", current_price, 0, 0, reasons=["Breakout", f"Score:{exec_score}"], session=session)
                    signal.volatility_spike = vol_spike
                    return signal
        else:
            sup_m15 = np.min(m15_candles.low[-lookback:-1])
            if current_price < sup_m15:
                m5_ema = self._calculate_ema(m5_candles.close, self.ema_fast)
                if current_price < m5_ema:
                    signal = TradeSignal("SELL", current_price, 0, 0, reasons=["Breakout", f"Score:{exec_score}"], session=session)
                    signal.volatility_spike = vol_spike
                    return signal
                
        return None

    def _check_pullback_entry(self, m15_candles: 'CandleArray', m5_candles: 'CandleArray', 
                              trend: str, current_price: float, session: str) -> Optional[TradeSignal]:
        """
        [PHASE 12] Pullback Logic (Mean-Reversion in trend).
        Catches entries on EMA retests after a recent M15 breakout.
        """
        if trend == "RANGING": return None
        
        m15_ema20 = self._calculate_ema(m15_candles.close, 20)
        m15_ema50 = self._calculate_ema(m15_candles.close, 50)
        
        # Pullback: Price retraces towards EMA20 while trend is established
        if trend == "BULLISH":
            # 1. Structural requirements: 20 > 50
            if m15_ema20 > m15_ema50:
                # 2. Touch/Approach Check: Current price near or slightly below EMA20 but above EMA50
                if m15_ema50 < current_price < (m15_ema20 * 1.002):
                    # 3. Confirmation: M5 candle must be closing higher (Rejection of the low)
                    if m5_candles.close[-1] > m5_candles.open[-1] and m5_candles.close[-1] > m15_ema20:
                        return TradeSignal("BUY", current_price, 0, 0, reasons=["EMA Pullback"], session=session)
        else: # BEARISH
            if m15_ema20 < m15_ema50:
                if m15_ema50 > current_price > (m15_ema20 * 0.998):
                    if m5_candles.close[-1] < m5_candles.open[-1] and m5_candles.close[-1] < m15_ema20:
                        return TradeSignal("SELL", current_price, 0, 0, reasons=["EMA Pullback"], session=session)
        return None

    def _set_sl_tp(self, signal: TradeSignal, atr: float, m30_candles: 'CandleArray', 
                   h4_strength: int, session: str, confluence_score: int, ai_bias: float = 0.0, 
                   m5_candles: Optional['CandleArray'] = None) -> TradeSignal:
        
        """
        [PHASE 13.6] High-Leverage Institutional Executioner.
        Uses M5 Extremums to minimize SL distance, maximizing lot size and R-growth.
        """
        session_conf = self.config.get("session_config", {}).get(session, {})
        base_rr = session_conf.get("rr_ratio", self.rr_ratio)
        rr = base_rr + (0.5 if h4_strength > 75 else -0.5 if h4_strength < 40 else 0)
        rr = max(1.5, min(rr + ai_bias, 6.0))
        
        # 4. Global Minimum SL (Safety Floor)
        min_sl_pts = self.config.get("risk", {}).get("min_sl_points", 50)
        point = 0.01 # Gold standard
        
        # 5. Micro-Stop Placement
        if signal.direction == "BUY":
            # [Micro-Stop] Low of the trigger candle + small cushion
            if m5_candles is not None:
                m5_low = np.min(m5_candles.low[-2:])
                target_sl = m5_low - (5 * point)
                # Ensure distance is at least min_sl_pts
                if (signal.entry_price - target_sl) < (min_sl_pts * point):
                    target_sl = signal.entry_price - (min_sl_pts * point)
                signal.stop_loss = target_sl
            else:
                # Fallback to ATR if M5 not available
                signal.stop_loss = signal.entry_price - (2.0 * atr)
                
            risk = signal.entry_price - signal.stop_loss
            signal.take_profit = signal.entry_price + (risk * rr)
        else:
            if m5_candles is not None:
                m5_high = np.max(m5_candles.high[-2:])
                target_sl = m5_high + (5 * point)
                if (target_sl - signal.entry_price) < (min_sl_pts * point):
                    target_sl = signal.entry_price + (min_sl_pts * point)
                signal.stop_loss = target_sl
            else:
                signal.stop_loss = signal.entry_price + (2.0 * atr)
                
            risk = signal.stop_loss - signal.entry_price
            signal.take_profit = signal.entry_price - (risk * rr)
            
        signal.rr_ratio = rr
        return signal

            
        signal.rr_ratio = rr
        return signal

    def _tokyo_mean_reversion(self, m30_candles: 'CandleArray', m5_candles: 'CandleArray', 
                              current_price: float, m30_ema: float, m30_rsi: float, session: str) -> Optional[TradeSignal]:
        """
        [PHASE 11] Adaptive Mean-Reversion Trigger.
        Used for Tokyo session or Ranging/Low-ADX markets.
        """
        # Overbought/Oversold thresholds for MR
        if m30_rsi > 70 and current_price < m30_ema:
            # Signal: High rejection from overbought top, returning to mean
            return TradeSignal("SELL", current_price, 0, 0, reasons=["Tokyo MR (Resistance)"], session=session)
        elif m30_rsi < 30 and current_price > m30_ema:
            # Signal: Strong bounce from oversold bottom, returning to mean
            return TradeSignal("BUY", current_price, 0, 0, reasons=["Tokyo MR (Support)"], session=session)
            
        return None

    def _calculate_confluence(self, h4_trend: str, h4_conviction: int, regime: str, signal: TradeSignal,
                               m30_candles: 'CandleArray', m5_candles: 'CandleArray',
                               m30_atr: float, m5_atr: float, session: str) -> Tuple[int, List[str]]:
        """Institutional Confluence Scoring without H1 dependency."""
        reasons = []
        score = 0
        # 1. Trend Alignment (4 pts)
        if (signal.direction == "BUY" and h4_trend == "BULLISH") or (signal.direction == "SELL" and h4_trend == "BEARISH"):
            score += 2; reasons.append("H4 Alignment")
            if h4_conviction > 60:
                score += 2; reasons.append("High Conviction Anchor")
            elif h4_conviction > 40:
                score += 1; reasons.append("Moderate Trend Strength")
            
        # 2. Session Momentum (1 pt)
        if session and session in ["LONDON", "LONDON/NY", "NEW_YORK"]:
            score += 1; reasons.append(f"{session} Momentum Session")
            
        # 3. M5 EMA Alignment (2 pts)
        m5_closes = m5_candles.close
        m5_ema20 = self._calculate_ema_series(m5_closes, 20)
        m5_ema50 = self._calculate_ema_series(m5_closes, 50)
        
        if len(m5_ema20) > 2:
            if signal.direction == "BUY" and m5_ema20[-1] > m5_ema50[-1]:
                score += 2; reasons.append("M5 EMA Alignment")
            elif signal.direction == "SELL" and m5_ema20[-1] < m5_ema50[-1]:
                score += 2; reasons.append("M5 EMA Alignment")
            
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

    # Removed duplicate _analyze_preprocessed to ensure Phase 13.3 logic is used.


    def preprocess_history(self, h4: 'CandleArray', m30: 'CandleArray', m15: 'CandleArray', m5: 'CandleArray') -> dict:
        """
        [PHASE 11] Institutional Vectorized Preprocessing.
        Maps H4-M30-M15 state to each M5 candle using strict look-behind offsets.
        """
        import time as timer
        start = timer.time()
        logger.info("[Strategy] Phase 11 Vectorized Preprocessing started...")

        # 1. H4 Pre-calculations (Anchor)
        h4_trend_series = []
        h4_conv_series = []
        h4_atr_series = []
        for i in range(len(h4)):
            slice_h4 = h4[:i+1]
            t, c = self._get_h4_trend(slice_h4)
            h4_trend_series.append(t)
            h4_conv_series.append(c)
            h4_atr_series.append(self._calculate_atr(slice_h4))

        # 2. M30 Pre-calculations (Momentum)
        m30_closes = m30.close
        m30_vols = m30.tick_volume
        m30_ema = self._calculate_ema_series(m30_closes, 10)
        m30_rsi = self._calculate_rsi_series(m30_closes, 14)
        m30_vw_rsi = self._calculate_vw_rsi_series(m30_closes, m30_vols, 14)
        m30_er = self._calculate_efficiency_ratio_series(m30_closes, 14)
        _, _, m30_hist = self._calculate_macd_series(m30_closes)
        
        m30_regimes = []
        for i in range(len(m30)):
            m30_regimes.append(MarketRegime.classify(m30[:i+1]))

        # 2b. M15 ATR Pre-calculation
        m15_atr_series = []
        for i in range(len(m15)):
            m15_atr_series.append(self._calculate_atr(m15[:i+1]))

        # 3. Time-based Mapping to M5
        m5_times = m5.time
        h4_times = h4.time
        m30_times = m30.time
        m15_times = m15.time # though not used for mapping yet, good to have
        
        m5_precomputed = []
        for i in range(len(m5)):
            t = m5_times[i]
            
            # Offsets to prevent look-ahead bias
            h4_idx = np.searchsorted(h4_times, t - 14400, side='right') - 1
            m30_idx = np.searchsorted(m30_times, t - 1800, side='right') - 1
            m15_idx = np.searchsorted(m15_times, t - 900, side='right') - 1
            
            h4_idx = max(0, h4_idx)
            m30_idx = max(0, m30_idx)
            m15_idx = max(0, m15_idx)
            
            # MACD Velocity
            m30_macd_vel_up = m30_hist[m30_idx] > m30_hist[max(0, m30_idx-1)]
            m30_macd_vel_down = m30_hist[m30_idx] < m30_hist[max(0, m30_idx-1)]
            
            m5_precomputed.append({
                "h4_trend": h4_trend_series[h4_idx],
                "h4_conviction": h4_conv_series[h4_idx],
                "h4_atr": h4_atr_series[h4_idx],
                "regime": m30_regimes[m30_idx],
                "m30_ema_10": m30_ema[m30_idx], # Sync key with _analyze_preprocessed
                "m30_rsi_14": m30_rsi[m30_idx],
                "m30_vw_rsi": m30_vw_rsi[m30_idx],
                "m30_er": m30_er[m30_idx],
                "m15_atr": m15_atr_series[m15_idx], # Added for Micro-Stops
                "m30_macd_vel_up": m30_macd_vel_up,
                "m30_macd_vel_down": m30_macd_vel_down
            })
            
        logger.info("[Strategy] Preprocessing complete in %.2fs", timer.time() - start)
        return {"m5": m5_precomputed}