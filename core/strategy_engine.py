"""
TRADING BOT V3 - PURE PRICE ACTION SNIPER (5M)
Scales sniping logic to M5 for reduced noise and HTF Institutional Zones.
Hierarchy: HTF (Zones) -> M15 (Bias) -> M5 (Entries/Trailing)
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

# Named constants for magic numbers
_SECONDS_PER_DAY = 86400
_SECONDS_PER_5MIN = 300
_HTF_ZONE_LOOKBACK_DAYS = 5

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
    timestamp: Optional[datetime] = None
    rr_ratio: float = 2.0

class StrategyEngine:
    def __init__(self, config: dict, analysis_logger=None, silent: bool = False):
        self.config = config
        self.strategy_config = config.get("strategy_defaults", {})
        self.pa_config = self.strategy_config.get("price_action", {})
        self.analysis_logger = analysis_logger
        self.silent = silent
        self.strategy_type = config.get("strategy_type", "SNIPER") 

        # Configuration Primitives
        self.swing_lookback = int(self.pa_config.get("swing_lookback", 12))
        self.min_wick_pct = float(self.pa_config.get("min_wick_pct", 40.0))
        self.min_body_pct = float(self.pa_config.get("min_body_pct", 15.0))
        self.fixed_rr = float(self.pa_config.get("fixed_rr", 3.0))

        # Legacy/Misc Config
        self.min_confidence = self.strategy_config.get("min_confidence", 65)
        self.cooldown_candles = int(self.strategy_config.get("cooldown_candles", 12))
        self.research_mode = config.get("research_mode", False)
        
        # State tracking
        self.trade_counter = 0
        self.last_stop_index = -999
        self.last_loss_date = None
        self.daily_losses = 0
        self.daily_trades = 0
        self.lock = None # Will be initialized if needed
        self.session_cooldown_active = {s: False for s in _DEFAULT_SESSIONS}
        self.consecutive_losses = {s: 0 for s in _DEFAULT_SESSIONS}

    def _log(self, msg: str):
        if not self.silent: logger.info(msg)

    def get_session_from_hour(self, hour: int, utc_offset: int = 0) -> str:
        """Determines the trading session based on the hour (UTC)."""
        h = (hour + utc_offset) % 24
        if 0 <= h < 8: return "TOKYO"
        if 8 <= h < 13: return "LONDON"
        if 13 <= h < 17: return "LONDON/NY"
        if 17 <= h < 24: return "NEW_YORK"
        return "GLOBAL"

    def _calculate_atr(self, candles: 'CandleArray', period: int = 14) -> float:
        if len(candles) < period + 1: return 0.0
        high = candles.high; low = candles.low; close = candles.close
        tr = np.maximum(high[1:] - low[1:], 
                        np.maximum(np.abs(high[1:] - close[:-1]), 
                                   np.abs(low[1:] - close[:-1])))
        return float(np.mean(tr[-period:]))

    def _is_engulfing(self, po, pc, co, cc, direction: str) -> bool:
        if direction == "BUY": return cc > co and pc < po and cc > po and co < pc
        return cc < co and pc > po and cc < po and co > pc

    def analyze(self, symbol: str, htf_candles: 'CandleArray', m15_candles: 'CandleArray', 
                m5_candles_original: 'CandleArray', current_price: float,
                d1_candles: Optional['CandleArray'] = None, session: Optional[str] = None,
                preprocessed: Optional[dict] = None, circuit_breaker_safe: bool = True) -> Tuple[Optional[TradeSignal], str, str]:
        
        # FIX #5: Guard against empty CandleArrays
        if len(m5_candles_original) < 2:
            return None, "NEUTRAL", "INSUFFICIENT_DATA"
        
        raw_ts = m5_candles_original.time[-1]
        timestamp = datetime.fromtimestamp(float(raw_ts), tz=timezone.utc)
        if self.last_loss_date != timestamp.date():
            self.reset_daily_stats(); self.last_loss_date = timestamp.date()

        gate_status, gate_reason = self._check_gatekeepers(symbol, session, circuit_breaker_safe)
        if not self.research_mode and not gate_status: return None, "NEUTRAL", gate_reason

        if self.strategy_type == "SMC":
            return self._analyze_smc(symbol, m5_candles_original, current_price, session, preprocessed)
        
        return self._analyze_sniper(symbol, htf_candles, m15_candles, m5_candles_original, current_price, session, preprocessed)

    def _analyze_smc(self, symbol: str, m5_candles: 'CandleArray', current_price: float, 
                     session: str, preprocessed: dict) -> Tuple[Optional[TradeSignal], str, str]:
        """Smart Money Concepts strategy: HTF Zone + LTF Rejection + Volume Spike."""
        m5_ctx = preprocessed if preprocessed else {}
        htf_demand = m5_ctx.get("in_htf_demand", False)
        htf_supply = m5_ctx.get("in_htf_supply", False)
        rej_bull = m5_ctx.get("rej_bull", False)
        rej_bear = m5_ctx.get("rej_bear", False)
        bias = m5_ctx.get("m_bias", "NEUTRAL")
        
        # FIX #18: Removed duplicate variable declarations (was declared twice)
        last_m5 = m5_candles[-1]
        atr = self._calculate_atr(m5_candles, 14)
        signal = None
        reason = ""

        vol_sma = m5_ctx.get("vol_sma", 0.0)
        vol_expansion = m5_candles.tick_volume[-1] > vol_sma * 1.25
        sweep_bull = m5_ctx.get("sweep_bull", False)
        sweep_bear = m5_ctx.get("sweep_bear", False)

        # Hardened Logic: In HTF Zone + LTF Rejection + Volume Spike
        if htf_demand and rej_bull and bias != "BEARISH" and vol_expansion:
            # Tokyo Buffer: Require a Liquidity Sweep
            if session != "TOKYO" or sweep_bull:
                signal = TradeSignal("BUY", current_price, last_m5['low'] - (atr*0.3), 0, session=session)
                reason = "HTF Demand + M5 Rej + VolSpike"
                signal.confidence = 85.0
        
        if not signal and htf_supply and rej_bear and bias != "BULLISH" and vol_expansion:
            if session != "TOKYO" or sweep_bear:
                signal = TradeSignal("SELL", current_price, last_m5['high'] + (atr*0.3), 0, session=session)
                reason = "HTF Supply + M5 Rej + VolSpike"
                signal.confidence = 85.0

        if signal:
            signal.take_profit = signal.entry_price + (abs(signal.entry_price - signal.stop_loss) * self.fixed_rr) if signal.direction == "BUY" else \
                                 signal.entry_price - (abs(signal.entry_price - signal.stop_loss) * self.fixed_rr)
            signal.tp1_price = signal.entry_price + (abs(signal.entry_price - signal.stop_loss) * 1.5) if signal.direction == "BUY" else \
                               signal.entry_price - (abs(signal.entry_price - signal.stop_loss) * 1.5)
            signal.reasons = [reason, f"Bias: {bias}"]

        return signal, bias, "SMC_ELITE"

    def _analyze_sniper(self, symbol: str, htf_candles: 'CandleArray', m15_candles: 'CandleArray', 
                        m5_candles_original: 'CandleArray', current_price: float, 
                        session: str, preprocessed: dict) -> Tuple[Optional[TradeSignal], str, str]:
        """Pure Price Action Sniper: M15 bias + M5 liquidity sweep/rejection entries."""
        if not self.research_mode and self.trade_counter - self.last_stop_index < self.cooldown_candles: return None, "NEUTRAL", "COOLDOWN"
        self.trade_counter += 1
        m5 = m5_candles_original[-100:]
        if len(m5) < 20: return None, "NEUTRAL", "INSUFFICIENT_DATA"

        bias = preprocessed.get("m_bias", "NEUTRAL") if preprocessed else "NEUTRAL"
        m_high = preprocessed.get("m_high", 999999.0) if preprocessed else 999999.0
        m_low = preprocessed.get("m_low", -999999.0) if preprocessed else -999999.0
        in_demand = preprocessed.get("in_htf_demand", False) if preprocessed else False
        in_supply = preprocessed.get("in_htf_supply", False) if preprocessed else False
        vol_expansion = m5_candles_original.tick_volume[-1] > preprocessed.get("vol_sma", 0) * 1.5
        
        last_m5 = m5[-1]; prev_m5 = m5[-2]
        atr = self._calculate_atr(m5, 14)
        signal = None; reason = ""

        can_buy = bias != "BEARISH" and self.consecutive_losses.get(session, 0) < 3
        can_sell = bias != "BULLISH" and self.consecutive_losses.get(session, 0) < 3

        if can_buy and (in_demand or bias == "BULLISH"):
            if last_m5['low'] < m_low and current_price > m_low and vol_expansion:
                is_bull_rej, _ = self._is_rejection_candle(last_m5)
                if is_bull_rej or self._is_engulfing(prev_m5['open'], prev_m5['close'], last_m5['open'], last_m5['close'], "BUY"):
                    signal = TradeSignal("BUY", current_price, 0, 0, session=session)
                    reason = "M5 Demand Sniper" if in_demand else "M5 Bullish Flow"

        if not signal and can_sell and (in_supply or bias == "BEARISH"):
            if last_m5['high'] > m_high and current_price < m_high and vol_expansion:
                _, is_bear_rej = self._is_rejection_candle(last_m5)
                if is_bear_rej or self._is_engulfing(prev_m5['open'], prev_m5['close'], last_m5['open'], last_m5['close'], "SELL"):
                    signal = TradeSignal("SELL", current_price, 0, 0, session=session)
                    reason = "M5 Supply Sniper" if in_supply else "M5 Bearish Flow"

        if signal:
            conf = 60.0
            if bias == ("BULLISH" if signal.direction == "BUY" else "BEARISH"): conf += 20.0
            if vol_expansion: conf += 10.0
            if (signal.direction == "BUY" and in_demand) or (signal.direction == "SELL" and in_supply): conf += 10.0
            
            signal.confidence = conf
            signal.confluence_score = int(conf / 10)
            signal.reasons = [reason, f"Bias: {bias}", f"Conf: {conf}%"]
            signal = self._setup_trade_params(signal, last_m5, atr, m_high, m_low)
            
            if signal.confidence < self.min_confidence: return None, "NEUTRAL", "LOW_CONFIDENCE"

        return signal, bias, "PA_SNIPER"

    def _setup_trade_params(self, signal: TradeSignal, last_candle: dict, atr: float, 
                             sh: float, sl: float) -> TradeSignal:
        session = signal.session
        vol_buffer_mult = 1.0
        if session in ["TOKYO", "NEW_YORK"]: vol_buffer_mult = 1.5
        buffer = atr * vol_buffer_mult
        
        if signal.direction == "BUY":
            signal.stop_loss = max(sl - buffer, signal.entry_price - (atr * 4.0))
            risk = signal.entry_price - signal.stop_loss
            signal.take_profit = signal.entry_price + (risk * self.fixed_rr)
            signal.tp1_price = signal.entry_price + (risk * 1.0)
        else:
            signal.stop_loss = min(sh + buffer, signal.entry_price + (atr * 4.0))
            risk = signal.stop_loss - signal.entry_price
            signal.take_profit = signal.entry_price - (risk * self.fixed_rr)
            signal.tp1_price = signal.entry_price - (risk * 1.0)
        return signal

    def _check_gatekeepers(self, symbol: str, session: str, cb_safe: bool) -> Tuple[bool, str]:
        if not cb_safe: return False, "CIRCUIT_BREAKER"
        if not session or session not in _DEFAULT_SESSIONS: return False, "OFF_SESSION"
        
        # 1. Check symbol-specific override
        sym_sessions = self.config.get("symbols_config", {}).get(symbol, {}).get("sessions", {})
        if session in sym_sessions:
            if not sym_sessions[session].get("enabled", True):
                return False, f"SESSION_DISABLED_{symbol}"
        
        # 2. Check global session config
        global_sessions = self.config.get("session_config", {})
        if session in global_sessions:
            if not global_sessions[session].get("enabled", True):
                return False, "SESSION_DISABLED_GLOBAL"

        return True, ""

    def reset_daily_stats(self):
        self.daily_losses = 0; self.daily_trades = 0
        for s in self.consecutive_losses: self.consecutive_losses[s] = 0; self.session_cooldown_active[s] = False

    def _is_rejection_candle(self, c: dict) -> Tuple[bool, bool]:
        """
        Determines if a candle shows a sharp rejection (Pin Bar).
        Hardened: Requires 65% wick-to-range ratio.
        """
        body = abs(c['close'] - c['open'])
        range_val = c['high'] - c['low']
        if range_val == 0: return False, False
        
        # Bullish Rejection: Lower wick > 65% of candle
        is_bull = (min(c['open'], c['close']) - c['low']) / range_val > 0.65 and body / range_val < 0.3
        # Bearish Rejection: Upper wick > 65% of candle
        is_bear = (c['high'] - max(c['open'], c['close'])) / range_val > 0.65 and body / range_val < 0.3
        
        return is_bull, is_bear

    def preprocess_history(self, htf: 'CandleArray', m15: 'CandleArray', m5_alias: 'CandleArray', m5_original: 'CandleArray') -> dict:
        htf_c = htf.close; htf_h = htf.high; htf_l = htf.low; htf_t = htf.time; htf_o = htf.open
        m15_c = m15.close; m15_t = m15.time
        m5_c = m5_original.close; m5_h = m5_original.high; m5_l = m5_original.low; m5_t = m5_original.time; m5_v = m5_original.tick_volume; m5_o = m5_original.open
        
        # HTF Basics
        htf_body = np.abs(htf_c - htf_o)
        htf_atr = pd.Series(htf_body).rolling(20).mean().values
        htf_is_bear = htf_c < htf_o; htf_is_bull = htf_c > htf_o

        # 1. HTF Structural Displacement (BoS/CHoCH)
        htf_sh = pd.Series(htf_h).shift(1).rolling(10).max().values
        htf_sl = pd.Series(htf_l).shift(1).rolling(10).min().values
        
        # Hardened: Zone MUST be accompanied by a structural break
        # A pattern of candles [i, i+1] is only complete after candle i+1 closes.
        demand_mask = (htf_is_bear[:-1] & htf_is_bull[1:] & (htf_c[1:] > htf_sh[:-1]) & (htf_body[1:] > (htf_atr[:-1] * 1.5)))
        supply_mask = (htf_is_bull[:-1] & htf_is_bear[1:] & (htf_c[1:] < htf_sl[:-1]) & (htf_body[1:] > (htf_atr[:-1] * 1.5)))
        
        demand_zones = []; supply_zones = []
        # HTF period in seconds (assuming H1 for zones)
        htf_period = 3600 
        for i in range(len(demand_mask)):
            if demand_mask[i]: 
                # Pattern ends at i+1. It's only confirmed AFTER candle i+1 closes.
                demand_zones.append({"high": htf_h[i], "low": htf_l[i], "confirmed_at": htf_t[i+1] + htf_period})
            if supply_mask[i]: 
                supply_zones.append({"high": htf_h[i], "low": htf_l[i], "confirmed_at": htf_t[i+1] + htf_period})

        m15_is_high = (m15_c == pd.Series(m15_c).rolling(7, min_periods=4).max().values)
        m15_is_low = (m15_c == pd.Series(m15_c).rolling(7, min_periods=4).min().values)
        m15_biases = []
        lhi=[]; lli=[]
        for i in range(len(m15)):
            if m15_is_high[i]: lhi.append(i); 
            if m15_is_low[i]: lli.append(i);
            if len(lhi)<2 or len(lli)<2: m15_biases.append("NEUTRAL"); continue
            hc, hp = m15_c[lhi[-1]], m15_c[lhi[-2]]
            lc, lp = m15_c[lli[-1]], m15_c[lli[-2]]
            if hc > hp and lc > lp: m15_biases.append("BULLISH")
            elif hc < hp and lc < lp: m15_biases.append("BEARISH")
            else: m15_biases.append("NEUTRAL")

        m5_swing_high = pd.Series(m5_h).shift(1).rolling(window=self.swing_lookback).max().values
        m5_swing_low = pd.Series(m5_l).shift(1).rolling(window=self.swing_lookback).min().values
        m5_vol_sma = pd.Series(m5_v).rolling(window=20).mean().values

        m5_precomputed = []
        active_ob_bull = None; active_ob_bear = None
        current_fvg_bull = False; current_fvg_bear = False
        
        for i in range(len(m5_original)):
            t = m5_t[i]; cp = m5_c[i]; cl = m5_l[i]; ch = m5_h[i]
            m15_idx = np.searchsorted(m15_t, t - _SECONDS_PER_5MIN, side='right') - 1; m15_idx = max(0, m15_idx)
            # FIX: Use 'confirmed_at' to ensure zero lookahead
            relevant_demand = [z for z in demand_zones if z["confirmed_at"] <= t and z["confirmed_at"] > t - _SECONDS_PER_DAY * _HTF_ZONE_LOOKBACK_DAYS]
            relevant_supply = [z for z in supply_zones if z["confirmed_at"] <= t and z["confirmed_at"] > t - _SECONDS_PER_DAY * _HTF_ZONE_LOOKBACK_DAYS]
            in_d = any((cl <= z["high"] and ch >= z["low"]) for z in relevant_demand)
            in_s = any((ch >= z["low"] and cl <= z["high"]) for z in relevant_supply)
            
            if i > 5:
                if cl > m5_h[i-2]: current_fvg_bull = True
                if ch < m5_l[i-2]: current_fvg_bear = True
                if cp > m5_swing_high[i] and m5_c[i-1] < m5_o[i-1]: active_ob_bull = {"high": m5_h[i-1], "low": m5_l[i-1]}; current_fvg_bull = False
                if cp < m5_swing_low[i] and m5_c[i-1] > m5_o[i-1]: active_ob_bear = {"high": m5_h[i-1], "low": m5_l[i-1]}; current_fvg_bear = False

            if active_ob_bull and cl < active_ob_bull['low']: active_ob_bull = None 
            if active_ob_bear and ch > active_ob_bear['high']: active_ob_bear = None

            rej_bull, rej_bear = self._is_rejection_candle({"open": m5_o[i], "high": ch, "low": cl, "close": cp})

            sweep_bull = cl < m5_swing_low[i] and cp > m5_swing_low[i]
            sweep_bear = ch > m5_swing_high[i] and cp < m5_swing_high[i]

            m5_precomputed.append({
                "m_bias": m15_biases[min(m15_idx, len(m15_biases)-1)] if m15_biases else "NEUTRAL",
                "m_high": m5_swing_high[i], "m_low": m5_swing_low[i],
                "in_htf_demand": in_d, "in_htf_supply": in_s,
                "vol_sma": m5_vol_sma[i],
                "rej_bull": rej_bull, "rej_bear": rej_bear,
                "sweep_bull": sweep_bull, "sweep_bear": sweep_bear
            })
        return {"m5": m5_precomputed}