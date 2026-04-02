"""
TRADING BOT V3 - PURE PRICE ACTION SNIPER (5M)
Scales sniping logic to M5 for reduced noise and H1 Institutional Zones.
Hierarchy: H1 (Zones) -> M15 (Bias) -> M5 (Entries/Trailing)
"""

import logging
import numpy as np
import pandas as pd
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
    "TOKYO": "TOKYO", "LONDON": "LONDON", "LONDON_NY": "LONDON/NY", "LONDON/NY": "LONDON/NY", "NEW_YORK": "NEW_YORK"
}

@dataclass
class TradeSignal:
    direction: str
    entry_price: float
    stop_loss: float
    take_profit: float
    session: str = "GLOBAL"
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
        self.pa_config = self.strategy_config.get("price_action", {})
        self.analysis_logger = analysis_logger
        self.silent = silent

        # Configuration Primitives
        self.swing_lookback = int(self.pa_config.get("swing_lookback", 12))
        self.min_wick_pct = float(self.pa_config.get("min_wick_pct", 40.0))
        self.min_body_pct = float(self.pa_config.get("min_body_pct", 15.0))
        self.fixed_rr = float(self.pa_config.get("fixed_rr", 3.0))

        # Legacy/Misc Config
        self.min_confidence = self.strategy_config.get("min_confidence", 65)
        self.cooldown_candles = int(self.strategy_config.get("cooldown_candles", 12)) # ~1 hour at M5
        
        # Choppy Mitigation
        self.last_stop_time: Optional[datetime] = None
        self.daily_losses = 0; self.daily_trades = 0
        self.last_loss_date = None; self.trade_counter = 0; self.last_stop_index = -999

        # Sessions
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
        utc_hour = (hour - utc_offset) % 24
        if 8 <= utc_hour < 14: return "LONDON"
        if 14 <= utc_hour < 17: return "LONDON/NY"
        if 17 <= utc_hour < 22: return "NEW_YORK"
        return "TOKYO"

    def _log(self, message: str, level: str = "INFO"):
        if self.silent: return
        if self.analysis_logger: self.analysis_logger.log(message, level)
        logger.info(message)

    def _calculate_atr(self, candles: 'CandleArray', period: int = 14) -> float:
        if not candles or len(candles) < 2: return 0.1
        h, l, c = candles.high, candles.low, candles.close
        tr = np.maximum(h[1:] - l[1:], np.maximum(np.abs(h[1:] - c[:-1]), np.abs(l[1:] - c[:-1])))
        return np.mean(tr[-period:]) if len(tr) >= period else np.mean(tr)

    def _is_rejection_candle(self, open_p, high_p, low_p, close_p, direction: str) -> bool:
        cr = high_p - low_p
        if cr <= 0: return False
        body = abs(close_p - open_p)
        if direction == "BUY":
            return ((min(open_p, close_p) - low_p) / cr * 100) >= self.min_wick_pct and (body / cr * 100) >= self.min_body_pct
        else:
            return ((high_p - max(open_p, close_p)) / cr * 100) >= self.min_wick_pct and (body / cr * 100) >= self.min_body_pct

    def _is_engulfing(self, po, pc, co, cc, direction: str) -> bool:
        if direction == "BUY": return cc > co and pc < po and cc > po and co < pc
        return cc < co and pc > po and cc < po and co > pc

    def analyze(self, symbol: str, h1_candles: 'CandleArray', m15_candles: 'CandleArray', 
                m5_candles_original: 'CandleArray', current_price: float,
                d1_candles: Optional['CandleArray'] = None, session: Optional[str] = None,
                preprocessed: Optional[dict] = None, circuit_breaker_safe: bool = True) -> Tuple[Optional[TradeSignal], str, str]:
        
        # Primary trigger timeframe is now M5
        raw_ts = m5_candles_original.time[-1]
        timestamp = datetime.fromtimestamp(float(raw_ts), tz=timezone.utc)
        if self.last_loss_date != timestamp.date():
            self.reset_daily_stats(); self.last_loss_date = timestamp.date()

        gate_status, gate_reason = self._check_gatekeepers(session, current_price, circuit_breaker_safe)
        if not gate_status: return None, gate_reason, "NEUTRAL"

        self.trade_counter += 1
        if self.trade_counter - self.last_stop_index < self.cooldown_candles: return None, "COOLDOWN", "NEUTRAL"

        m5 = m5_candles_original[-100:]
        if len(m5) < 20: return None, "INSUFFICIENT_DATA", "NEUTRAL"

        # sniper logic: Get precomputed Bias (M15) and Zones (H1)
        bias = preprocessed.get("m_bias", "NEUTRAL") if preprocessed else "NEUTRAL"
        m_high = preprocessed.get("m_high", 999999.0) if preprocessed else 999999.0
        m_low = preprocessed.get("m_low", 0.0) if preprocessed else 0.0
        
        in_demand = preprocessed.get("in_demand", False) if preprocessed else False
        in_supply = preprocessed.get("in_supply", False) if preprocessed else False
        
        last_m5 = m5[-1]; prev_m5 = m5[-2]; atr = self._calculate_atr(m5, 14)
        signal = None; reason = ""

        # --- M5 SNIPER ENTRY (Session-Aware Precision) ---
        vol_sma = preprocessed.get("vol_sma", 0.0)
        
        # Session Tuning: Tokyo needs more volume and deeper zone entries due to low liquidity
        vol_mult = 1.25 if session == "TOKYO" else 1.1
        depth_thresh = 20 if session == "TOKYO" else 30
        
        vol_expansion = last_m5['tick_volume'] > vol_sma * vol_mult if vol_sma > 0 else True
        
        if bias in ["BULLISH", "NEUTRAL"] and in_demand and preprocessed.get("d_depth", 50) < depth_thresh:
            if last_m5['low'] < m_low and current_price > m_low and vol_expansion:
                if self._is_rejection_candle(last_m5['open'], last_m5['high'], last_m5['low'], last_m5['close'], "BUY") or \
                   self._is_engulfing(prev_m5['open'], prev_m5['close'], last_m5['open'], last_m5['close'], "BUY"):
                    signal = TradeSignal("BUY", current_price, 0, 0, session=session)
                    reason = "M5 Demand Snipe (Outer Zone)"

        if not signal and bias in ["BEARISH", "NEUTRAL"] and in_supply and preprocessed.get("s_depth", 50) > (100 - depth_thresh):
            if last_m5['high'] > m_high and current_price < m_high and vol_expansion:
                if self._is_rejection_candle(last_m5['open'], last_m5['high'], last_m5['low'], last_m5['close'], "SELL") or \
                   self._is_engulfing(prev_m5['open'], prev_m5['close'], last_m5['open'], last_m5['close'], "SELL"):
                    signal = TradeSignal("SELL", current_price, 0, 0, session=session)
                    reason = "M5 Supply Sniper"

        if signal:
            signal.reasons = [reason, f"Bias: {bias}"]; signal.confluence_score = 10
            signal.confidence = 90.0
            signal = self._setup_trade_params(signal, last_m5, atr, m_high, m_low)
            self._log(f"SNIPER SIGNAL: {signal.direction} | {reason} | SL: {signal.stop_loss:.2f} | TP: {signal.take_profit:.2f}")

        return signal, bias, "PA_SNIPER"

    def _setup_trade_params(self, signal: TradeSignal, last_candle: dict, atr: float, 
                            sh: float, sl: float) -> TradeSignal:
        # TIGHT SL: Extreeme of rejection candle + 0.1 ATR (M5 needs more room than M1)
        buffer = atr * 0.1
        if signal.direction == "BUY":
            signal.stop_loss = last_candle['low'] - buffer
            risk = signal.entry_price - signal.stop_loss
            signal.take_profit = signal.entry_price + (risk * 20.0) # Expanded range for M5
            signal.tp1_price = signal.entry_price + risk # 1:1 BE Move
        else:
            signal.stop_loss = last_candle['high'] + buffer
            risk = signal.stop_loss - signal.entry_price
            signal.take_profit = signal.entry_price - (risk * 20.0)
            signal.tp1_price = signal.entry_price - risk
        signal.rr_ratio = 20.0
        signal.tp2_price = signal.take_profit
        return signal

    def _check_gatekeepers(self, session: str, cp: float, cb_safe: bool = True) -> Tuple[bool, str]:
        if not cb_safe: return False, "CB_TRIPPED"
        if session and session not in self.tradeable_sessions: return False, "SESSION_OFF"
        if session and self.session_cooldown_active.get(session, False): return False, "COOLDOWN"
        return True, "OK"

    def report_trade_result(self, result: str, timestamp: datetime, session: Optional[str] = None):
        with self.lock:
            self.daily_trades += 1
            if result == "SL":
                self.daily_losses += 1; self.last_stop_index = self.trade_counter
                if session:
                    self.consecutive_losses[session] += 1
                    if self.consecutive_losses[session] >= 2: self.session_cooldown_active[session] = True
            elif result == "TP":
                if session: self.consecutive_losses[session] = 0; self.session_cooldown_active[session] = False

    def reset_daily_stats(self):
        with self.lock:
            self.daily_losses = 0; self.daily_trades = 0
            for s in self.consecutive_losses: self.consecutive_losses[s] = 0; self.session_cooldown_active[s] = False

    def preprocess_history(self, h1: 'CandleArray', m15: 'CandleArray', m5_alias: 'CandleArray', m5_original: 'CandleArray') -> dict:
        """Vectorized Preprocessing for H1/M15/M5 snipe hierarchy."""
        logger.info("[Strategy] M5 Sniper Preprocessing...")
        h1_c = h1.close; h1_h = h1.high; h1_l = h1.low; h1_t = h1.time
        m15_c = m15.close; m15_t = m15.time
        m5_c = m5_original.close; m5_h = m5_original.high; m5_l = m5_original.low; m5_t = m5_original.time; m5_v = m5_original.tick_volume
        
        # 1. Zone Detection (H1)
        h1_body = np.abs(h1_c - h1.open)
        h1_atr = pd.Series(h1_body).rolling(20).mean().values
        h1_is_bear = h1_c < h1.open; h1_is_bull = h1_c > h1.open
        
        demand_mask = (h1_is_bear[:-1] & h1_is_bull[1:] & (h1_body[1:] > (h1_atr[:-1] * 1.5)))
        supply_mask = (h1_is_bull[:-1] & h1_is_bear[1:] & (h1_body[1:] > (h1_atr[:-1] * 1.5)))
        
        demand_zones = []; supply_zones = []
        for i in range(len(demand_mask)):
            if demand_mask[i]: demand_zones.append({"high": h1_h[i], "low": h1_l[i], "time": h1_t[i]})
            if supply_mask[i]: supply_zones.append({"high": h1_h[i], "low": h1_l[i], "time": h1_t[i]})

        # 2. Bias (M15)
        m15_is_high = (m15_c == pd.Series(m15_c).rolling(window=7, center=True).max().values)
        m15_is_low = (m15_c == pd.Series(m15_c).rolling(window=7, center=True).min().values)
        m15_biases = []
        for i in range(len(m15)):
            hi = np.where(m15_is_high[:i])[0]; li = np.where(m15_is_low[:i])[0]
            if len(hi) < 2 or len(li) < 2: m15_biases.append("NEUTRAL")
            elif m15_c[hi[-1]] > m15_c[hi[-2]] and m15_c[li[-1]] > m15_c[li[-2]]: m15_biases.append("BULLISH")
            elif m15_c[hi[-1]] < m15_c[hi[-2]] and m15_c[li[-1]] < m15_c[li[-2]]: m15_biases.append("BEARISH")
            else: m15_biases.append("NEUTRAL")

        # 3. M5 Sweeps & Volume
        m5_swing_high = pd.Series(m5_h).shift(1).rolling(window=self.swing_lookback).max().values
        m5_swing_low = pd.Series(m5_l).shift(1).rolling(window=self.swing_lookback).min().values
        m5_vol_sma = pd.Series(m5_v).rolling(window=20).mean().values

        # 4. Final Mapping (M5 steps)
        m5_precomputed = []
        for i in range(len(m5_original)):
            t = m5_t[i]; cp = m5_c[i]
            m15_idx = np.searchsorted(m15_t, t - 300, side='right') - 1; m15_idx = max(0, m15_idx)
            
            relevant_demand = [z for z in demand_zones if z["time"] < t and z["time"] > t - 86400 * 5] # 5 day lookback for zones
            relevant_supply = [z for z in supply_zones if z["time"] < t and z["time"] > t - 86400 * 5]
            
            in_d = False; d_depth = 50.0
            for z in relevant_demand:
                if z["low"] <= cp <= z["high"]:
                    in_d = True
                    d_depth = (cp - z["low"]) / (z["high"] - z["low"]) * 100 if z["high"] > z["low"] else 50.0
                    break
            
            in_s = False; s_depth = 50.0
            for z in relevant_supply:
                if z["low"] <= cp <= z["high"]:
                    in_s = True
                    s_depth = (cp - z["low"]) / (z["high"] - z["low"]) * 100 if z["high"] > z["low"] else 50.0
                    break
            
            m5_precomputed.append({
                "m_bias": m15_biases[m15_idx], "m_high": m5_swing_high[i], "m_low": m5_swing_low[i],
                "in_demand": in_d, "in_supply": in_s, "d_depth": d_depth, "s_depth": s_depth,
                "vol_sma": m5_vol_sma[i]
            })
        return {"m5": m5_precomputed}