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
    """
    Represents a generated trading signal with all necessary parameters for execution.
    
    Attributes:
        direction (str): 'BUY' or 'SELL'.
        entry_price (float): The price at which the trade should be entered.
        stop_loss (float): The price level for stopping losses.
        take_profit (float): The primary take profit target.
        session (str): The trading session name (e.g., 'LONDON', 'NEW_YORK').
        tp1_price (float): First partial take profit target.
        tp2_price (float): Second partial take profit target.
        tp3_price (float): Third partial take profit target.
        confidence (float): A percentage score (0-100) representing the signal's strength.
        confluence_score (int): Number of confluence factors aligned for this trade.
        reasons (List[str]): Human-readable reasons why this signal was generated.
        rejection_type (str): If signal is None, why it was rejected (internal use).
        timestamp (datetime): When the signal was created.
        rr_ratio (float): Risk-to-Reward ratio.
    """
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
    """
    Core engine responsible for analyzing market data and generating signals.
    
    Uses a multi-timeframe hierarchy:
    H1: Institutional Zone detection (Demand/Supply).
    M15: Bias determination (Bullish/Bearish/Neutral).
    M5: Entry precision, volume expansion, and rejection candle detection.
    """
    def __init__(self, config: dict, analysis_logger=None, silent: bool = False):
        """
        Initializes the StrategyEngine with config parameters.
        
        Args:
            config (dict): Global configuration dictionary.
            analysis_logger (Optional[Dashboard.AnalysisLogger]): Logger for UI dashboard updates.
            silent (bool): If True, suppresses standard logging output.
        """
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
        self.research_mode = config.get("research_mode", False)
        
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
        """
        Maps a UTC hour to a trading session.
        
        Args:
            hour (int): Current hour (0-23).
            utc_offset (int): Offset from UTC.
            
        Returns:
            str: Session name ('LONDON', 'NEW_YORK', 'LONDON/NY', 'TOKYO').
        """
        utc_hour = (hour - utc_offset) % 24
        if 8 <= utc_hour < 14: return "LONDON"
        if 14 <= utc_hour < 17: return "LONDON/NY"
        if 17 <= utc_hour < 22: return "NEW_YORK"
        return "TOKYO"

    def _log(self, message: str, level: str = "INFO"):
        """
        Internal logging wrapper that bridges to the analysis logger.
        
        Args:
            message (str): Log message.
            level (str): Log level ('INFO', 'WARNING', 'ERROR').
        """
        if self.silent: return
        if self.analysis_logger: self.analysis_logger.log(message, level)
        logger.info(message)

    def _calculate_atr(self, candles: 'CandleArray', period: int = 14) -> float:
        """
        Calculates the Average True Range (ATR) using NumPy for performance.
        
        Args:
            candles (CandleArray): Input candle data.
            period (int): Period for ATR calculation.
            
        Returns:
            float: The calculated ATR value.
        """
        if not candles or len(candles) < 2: return 0.1
        h, l, c = candles.high, candles.low, candles.close
        tr = np.maximum(h[1:] - l[1:], np.maximum(np.abs(h[1:] - c[:-1]), np.abs(l[1:] - c[:-1])))
        return np.mean(tr[-period:]) if len(tr) >= period else np.mean(tr)

    def _is_rejection_candle(self, open_p: float, high_p: float, low_p: float, close_p: float, direction: str) -> bool:
        """
        Detects price rejection (long wicks relative to body).
        
        Args:
            open_p, high_p, low_p, close_p (float): OHLC values for the current candle.
            direction (str): 'BUY' (look for bottom wick) or 'SELL' (look for top wick).
            
        Returns:
            bool: True if criteria for rejection are met.
        """
        cr = high_p - low_p
        if cr <= 0: return False
        body = abs(close_p - open_p)
        if direction == "BUY":
            return ((min(open_p, close_p) - low_p) / cr * 100) >= self.min_wick_pct and (body / cr * 100) >= self.min_body_pct
        else:
            return ((high_p - max(open_p, close_p)) / cr * 100) >= self.min_wick_pct and (body / cr * 100) >= self.min_body_pct

    def _is_engulfing(self, po: float, pc: float, co: float, cc: float, direction: str) -> bool:
        """
        Identifies an engulfing candle pattern.
        
        Args:
            po, pc (float): Open and Close of the previous candle.
            co, cc (float): Open and Close of the current candle.
            direction (str): 'BUY' (Bullish Engulfing) or 'SELL' (Bearish Engulfing).
            
        Returns:
            bool: True if engulfing pattern is detected.
        """
        if direction == "BUY": return cc > co and pc < po and cc > po and co < pc
        return cc < co and pc > po and cc < po and co > pc

    def analyze(self, symbol: str, h1_candles: 'CandleArray', m15_candles: 'CandleArray', 
                m5_candles_original: 'CandleArray', current_price: float,
                d1_candles: Optional['CandleArray'] = None, session: Optional[str] = None,
                preprocessed: Optional[dict] = None, circuit_breaker_safe: bool = True) -> Tuple[Optional[TradeSignal], str, str]:
        """
        Primary entry point for market analysis. Logic:
        1. Check gatekeepers (Session, Circuit Breaker).
        2. Evaluate M5 data for volume expansion.
        3. Match current price against H1 Institutional Zones (Demand/Supply).
        4. Validate against M15 Bias.
        5. Look for M5 Price Action patterns (Rejection candles, Engulfing).
        
        Args:
            symbol (str): Trading symbol (e.g., 'XAUUSDm').
            h1_candles (CandleArray): H1 timeframe data for zones.
            m15_candles (CandleArray): M15 timeframe data for bias.
            m5_candles_original (CandleArray): M5 timeframe data for entries.
            current_price (float): Current market price (bid/ask).
            d1_candles (Optional[CandleArray]): D1 data (optional).
            session (Optional[str]): Current session.
            preprocessed (Optional[dict]): Cached results from preprocess_history.
            circuit_breaker_safe (bool): Whether circuit breakers allow trading.
            
        Returns:
            Tuple[Optional[TradeSignal], str, str]: (Signal or None, Bias, LogicType).
        """
        
        # Primary trigger timeframe is now M5
        raw_ts = m5_candles_original.time[-1]
        timestamp = datetime.fromtimestamp(float(raw_ts), tz=timezone.utc)
        if self.last_loss_date != timestamp.date():
            self.reset_daily_stats(); self.last_loss_date = timestamp.date()

        gate_status, gate_reason = self._check_gatekeepers(session, current_price, circuit_breaker_safe)
        if not self.research_mode and not gate_status: return None, gate_reason, "NEUTRAL"

        self.trade_counter += 1
        if not self.research_mode and self.trade_counter - self.last_stop_index < self.cooldown_candles: return None, "COOLDOWN", "NEUTRAL"

        m5 = m5_candles_original[-100:]
        if len(m5) < 20: return None, "INSUFFICIENT_DATA", "NEUTRAL"

        # sniper logic: Get precomputed Bias (M15) and Zones (H1)
        bias = preprocessed.get("m_bias", "NEUTRAL") if preprocessed else "NEUTRAL"
        m_high = preprocessed.get("m_high", 999999.0) if preprocessed else 999999.0
        m_low = preprocessed.get("m_low", 0.0) if preprocessed else 0.0
        
        in_demand = preprocessed.get("in_demand", False) if preprocessed else False
        in_supply = preprocessed.get("in_supply", False) if preprocessed else False
        d_depth = preprocessed.get("d_depth", 50) if preprocessed else 50
        s_depth = preprocessed.get("s_depth", 50) if preprocessed else 50
        
        last_m5 = m5[-1]; prev_m5 = m5[-2]; atr = self._calculate_atr(m5, 14)
        signal = None; reason = ""

        # --- M5 SNIPER ENTRY (Session-Aware Precision) ---
        vol_sma = preprocessed.get("vol_sma", 0.0) if preprocessed else 0.0
        
        # Session Tuning: Tokyo needs more volume and deeper zone entries due to low liquidity
        vol_mult = 1.25 if session == "TOKYO" else 1.1
        depth_thresh = 20 if session == "TOKYO" else 30
        
        vol_expansion = last_m5['tick_volume'] > vol_sma * vol_mult if vol_sma > 0 else True
        if self.research_mode: vol_expansion = True # Bypass volume restriction
        
        # Entry Logic (Relaxed for Research Mode)
        bias_bull = (bias in ["BULLISH", "NEUTRAL"]) or self.research_mode
        bias_bear = (bias in ["BEARISH", "NEUTRAL"]) or self.research_mode
        
        # 1. Bulls (H1 Demand Zone OR strong M15 Bullish Bias)
        is_bullish_context = (bias == "BULLISH")
        can_buy = bias_bull and (in_demand or is_bullish_context)
        
        if can_buy and (d_depth < depth_thresh or is_bullish_context):
            if last_m5['low'] < m_low and current_price > m_low and vol_expansion:
                if self._is_rejection_candle(last_m5['open'], last_m5['high'], last_m5['low'], last_m5['close'], "BUY") or \
                   self._is_engulfing(prev_m5['open'], prev_m5['close'], last_m5['open'], last_m5['close'], "BUY"):
                    signal = TradeSignal("BUY", current_price, 0, 0, session=session)
                    reason = "M5 Demand Snipe" if in_demand else "M5 Bullish Flow"

        # 2. Bears (H1 Supply Zone OR strong M15 Bearish Bias)
        is_bearish_context = (bias == "BEARISH")
        can_sell = bias_bear and (in_supply or is_bearish_context)
        
        if not signal and can_sell and (s_depth > (100 - depth_thresh) or is_bearish_context):
            if last_m5['high'] > m_high and current_price < m_high and vol_expansion:
                if self._is_rejection_candle(last_m5['open'], last_m5['high'], last_m5['low'], last_m5['close'], "SELL") or \
                   self._is_engulfing(prev_m5['open'], prev_m5['close'], last_m5['open'], last_m5['close'], "SELL"):
                    signal = TradeSignal("SELL", current_price, 0, 0, session=session)
                    reason = "M5 Supply Sniper" if in_supply else "M5 Bearish Flow"

        if signal:
            # Dynamic Confidence Scaling
            conf = 60.0 # Base
            if bias == ("BULLISH" if signal.direction == "BUY" else "BEARISH"): conf += 20.0
            if vol_expansion: conf += 10.0
            if (signal.direction == "BUY" and in_demand) or (signal.direction == "SELL" and in_supply): conf += 10.0
            
            signal.confidence = conf
            signal.confluence_score = int(conf / 10)
            signal.reasons = [reason, f"Bias: {bias}", f"Conf: {conf}%"]
            signal = self._setup_trade_params(signal, last_m5, atr, m_high, m_low)
            
            # Final Gate: Strategic Confidence Filter
            if signal.confidence < self.min_confidence:
                self._log(f"SIGNAL REJECTED: Low Confidence ({signal.confidence}% < {self.min_confidence}%)")
                return None, "LOW_CONFIDENCE", "NEUTRAL"

            self._log(f"SNIPER SIGNAL: {signal.direction} | {reason} | Conf: {signal.confidence}% | SL: {signal.stop_loss:.2f} | TP: {signal.take_profit:.2f}")

        return signal, bias, "PA_SNIPER"

    def _setup_trade_params(self, signal: TradeSignal, last_candle: dict, atr: float, 
                            sh: float, sl: float) -> TradeSignal:
        """
        Calculates SL, TP, and R/R ratio based on local structure and ATR.
        
        Args:
            signal (TradeSignal): The signal to populate.
            last_candle (dict): The trigger candle data.
            atr (float): Current ATR for buffer calculation.
            sh, sl (float): Swing High/Low levels current.
            
        Returns:
            TradeSignal: The fully parameterized signal.
        """
        # --- STRUCTURAL + VOLATILITY SL ---
        # Gold needs room: Use Structure (M5 Swing) + 1.0 ATR Buffer
        session = signal.session
        vol_buffer_mult = 1.0
        if session in ["TOKYO", "NEW_YORK"]: vol_buffer_mult = 1.5
        
        buffer = atr * vol_buffer_mult
        
        if signal.direction == "BUY":
            # Place SL below the recent Swing Low (sl) or at least 1.0 ATR below entry
            structural_sl = sl - buffer
            min_sl = signal.entry_price - (atr * 4.0) # Risk cap
            signal.stop_loss = max(structural_sl, min_sl)
            
            risk = signal.entry_price - signal.stop_loss
            signal.take_profit = signal.entry_price + (risk * self.fixed_rr)
            # Partial TP Logic: Finance risk early
            signal.tp1_price = signal.entry_price + (risk * 1.0) # 1:1 Partial
            signal.tp2_price = signal.take_profit
        else:
            # Place SL above the recent Swing High (sh) or at least 1.0 ATR above entry
            structural_sl = sh + buffer
            max_sl = signal.entry_price + (atr * 4.0)
            signal.stop_loss = min(structural_sl, max_sl)
            
            risk = signal.stop_loss - signal.entry_price
            signal.take_profit = signal.entry_price - (risk * self.fixed_rr)
            signal.tp1_price = signal.entry_price - (risk * 1.0)
            signal.tp2_price = signal.take_profit
            
        signal.rr_ratio = self.fixed_rr
        return signal

    def _check_gatekeepers(self, session: str, cp: float, cb_safe: bool = True) -> Tuple[bool, str]:
        """
        Validates whether trading is allowed based on session and safety locks.
        
        Args:
            session (str): Current session name.
            cp (float): Current price.
            cb_safe (bool): Whether circuit breakers are clear.
            
        Returns:
            Tuple[bool, str]: (Allowed, Reason).
        """
        if not cb_safe: return False, "CB_TRIPPED"
        if session and session not in self.tradeable_sessions: return False, "SESSION_OFF"
        if session and self.session_cooldown_active.get(session, False): return False, "COOLDOWN"
        return True, "OK"

    def report_trade_result(self, result: str, timestamp: datetime, session: Optional[str] = None):
        """
        Updates internal counters after a trade closes.
        
        Args:
            result (str): 'TP' or 'SL'.
            timestamp (datetime): Close time.
            session (Optional[str]): Trading session.
        """
        with self.lock:
            self.daily_trades += 1
            if result == "SL":
                self.daily_losses += 1; self.last_stop_index = self.trade_counter
                if session:
                    self.consecutive_losses[session] += 1
                    max_losses = self.config.get("risk", {}).get("max_consecutive_losses", 2)
                    if self.consecutive_losses[session] >= max_losses: self.session_cooldown_active[session] = True
            elif result == "TP":
                if session: self.consecutive_losses[session] = 0; self.session_cooldown_active[session] = False

    def reset_daily_stats(self):
        with self.lock:
            self.daily_losses = 0; self.daily_trades = 0
            for s in self.consecutive_losses: self.consecutive_losses[s] = 0; self.session_cooldown_active[s] = False

    def preprocess_history(self, h1: 'CandleArray', m15: 'CandleArray', m5_alias: 'CandleArray', m5_original: 'CandleArray') -> dict:
        """
        Vectorized preprocessing of historical data to speed up the main analyze loop.
        Computes H1 Zones, M15 Biases, and M5 Structural levels in a single pass.
        
        Args:
            h1 (CandleArray): H1 data.
            m15 (CandleArray): M15 data.
            m5_alias (CandleArray): M5 data (unused alias).
            m5_original (CandleArray): Primary M5 data.
            
        Returns:
            dict: Mapped precomputed data for every M5 candle.
        """
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
        m15_is_high = (m15_c == pd.Series(m15_c).rolling(window=7, min_periods=4).max().values)
        m15_is_low = (m15_c == pd.Series(m15_c).rolling(window=7, min_periods=4).min().values)
        m15_biases = []
        last_hi_indices = []
        last_li_indices = []
        for i in range(len(m15)):
            if m15_is_high[i]:
                last_hi_indices.append(i)
                if len(last_hi_indices) > 2: last_hi_indices.pop(0)
            if m15_is_low[i]:
                last_li_indices.append(i)
                if len(last_li_indices) > 2: last_li_indices.pop(0)
            
            if len(last_hi_indices) < 2 or len(last_li_indices) < 2:
                m15_biases.append("NEUTRAL")
                continue
                
            # Bias Logic: Higher Highs + Higher Lows = BULLISH, Lower Highs + Lower Lows = BEARISH
            h_curr, h_prev = m15_c[last_hi_indices[-1]], m15_c[last_hi_indices[-2]]
            l_curr, l_prev = m15_c[last_li_indices[-1]], m15_c[last_li_indices[-2]]
            
            if h_curr > h_prev and l_curr > l_prev: m15_biases.append("BULLISH")
            elif h_curr < h_prev and l_curr < l_prev: m15_biases.append("BEARISH")
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