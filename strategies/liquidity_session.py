"""
TRADING BOT V4 — Liquidity Session Strategy (Sniper v4.1 Migrated)
================================================================
Institutional grade sweep and rejection logic for session-based liquidity.
"""

import logging
import numpy as np
from datetime import datetime, timezone
from typing import Any, Optional, Tuple

from core.base_strategy import BaseStrategy, MarketData
from core.types import TradeSignal

logger = logging.getLogger("trading_bot.strategy.liquidity")

_ACTIVE_SESSIONS = {"LONDON", "NEW_YORK", "LONDON/NY", "TOKYO"}
_VOL_WINDOW = 30 

class LiquiditySessionStrategy(BaseStrategy):
    """
    Pure Price Action Sniper v4.1 — Migrated to Institutional V4.
    Focuses on liquidity sweeps and rejections at institutional zones.
    """

    def __init__(self, strategy_id: str, config: dict):
        super().__init__(strategy_id, config)
        p = config.get("params", {})

        self.swing_lookback   = int(p.get("swing_lookback", 7))
        self.sl_atr_mult      = float(p.get("sl_atr_mult", 1.5))
        self.sl_max_atr_mult  = float(p.get("sl_max_atr_mult", 3.0))
        self.rej_wick_ratio   = float(p.get("rej_wick_ratio", 0.55))
        self.rej_body_ratio   = float(p.get("rej_body_ratio", 0.40))
        self.vol_mult         = float(p.get("vol_mult", 1.25))
        self.ema_period       = int(p.get("ema_period", 50))
        self.atr_regime_ratio = float(p.get("atr_regime_ratio", 0.60))

        sd = config.get("strategy_defaults", {})
        self.min_confidence   = float(config.get("min_confidence", sd.get("min_confidence", 0.55)))
        self.cooldown_candles = int(config.get("cooldown_candles", sd.get("cooldown_candles", 2)))
        self.research_mode    = config.get("research_mode", False)

        # Per-strategy state (Reset daily in V4)
        self.trade_counter        = 0
        self.last_stop_index      = -999
        self.last_loss_date       = None
        self.session_traded_today = {}
        self.consecutive_losses   = {s: 0 for s in _ACTIVE_SESSIONS}

    def generate_signal(self, market_data: MarketData) -> Optional[TradeSignal]:
        m5      = market_data.m5_candles
        session = market_data.session
        price   = market_data.current_price
        prep    = market_data.preprocessed or {}

        if len(m5) < 60:
            return None
        if not self.research_mode and session not in _ACTIVE_SESSIONS:
            return None

        # Daily reset logic
        raw_ts = float(m5.time[-1])
        today  = datetime.fromtimestamp(raw_ts, tz=timezone.utc).date()
        if self.last_loss_date != today:
            self._reset_daily_stats()
            self.last_loss_date = today

        self.trade_counter += 1

        # Cooldown & Daily Limits
        if not self.research_mode:
            if self.trade_counter - self.last_stop_index < self.cooldown_candles:
                return None
            if self.session_traded_today.get((session, today), 0) >= 2:
                return None
            if self.consecutive_losses.get(session, 0) >= 3:
                return None

        # Trend & Indicators
        m5w = m5[-60:]
        last = m5w[-1]
        prev = m5w[-2]

        atr_14 = self._calculate_atr(m5w, 14)
        atr_50 = self._calculate_atr(m5w, 50)
        
        if atr_14 <= 0 or (atr_50 > 0 and atr_14 < atr_50 * self.atr_regime_ratio):
            return None

        # EMA 50 Filter
        ema50 = np.mean(m5w.close[-self.ema_period:])
        trend_bull = price > ema50
        trend_bear = price < ema50

        # Volume Ok
        tv = m5.tick_volume[-_VOL_WINDOW:]
        vol_sma_live = np.mean(tv[tv > 0]) if np.any(tv > 0) else 1.0
        vol_ok = float(m5.tick_volume[-1]) > vol_sma_live * self.vol_mult

        # Price Action
        m_high     = prep.get("m_high", price + 9999)
        m_low      = prep.get("m_low",  price - 9999)
        sweep_bull = bool(prep.get("sweep_bull", False))
        sweep_bear = bool(prep.get("sweep_bear", False))

        rej_bull, rej_bear = self._is_rejection(last)
        engulf_b = self._is_engulfing(prev, last, "BUY")
        engulf_s = self._is_engulfing(prev, last, "SELL")

        sig = None
        
        # BUY Logic (T1, T2, T3 scaled to 0-1)
        if (sweep_bull or float(last["low"]) < float(m_low)) and (rej_bull or engulf_b):
            sig = self._create_signal("BUY", price, 0.88, "T1:Sweep+Rej")
        elif not sig and rej_bull and vol_ok and trend_bull:
            sig = self._create_signal("BUY", price, 0.72, "T2:Rej+Vol+Bull")
        elif not sig and engulf_b and trend_bull:
            sig = self._create_signal("BUY", price, 0.60, "T3:BullEngulf")

        # SELL Logic
        if not sig:
            if (sweep_bear or float(last["high"]) > float(m_high)) and (rej_bear or engulf_s):
                sig = self._create_signal("SELL", price, 0.88, "T1:Sweep+Rej")
            elif not sig and rej_bear and vol_ok and trend_bear:
                sig = self._create_signal("SELL", price, 0.72, "T2:Rej+Vol+Bear")
            elif not sig and engulf_s and trend_bear:
                sig = self._create_signal("SELL", price, 0.60, "T3:BearEngulf")

        if sig and sig.confidence >= self.min_confidence:
            # Set SL/TP based on Sniper logic
            self._apply_sniper_risk(sig, price, atr_14, m_high, m_low, last, session)
            return sig
            
        return None

    def get_stop_loss(self, signal: TradeSignal, market_data: MarketData) -> float:
        base_sl = getattr(signal, "stop_loss", market_data.current_price * 0.99)
        entry_price = getattr(signal, "price", market_data.current_price)
        if entry_price == 0: return base_sl
        
        # Calculate Current RR
        dist = abs(entry_price - base_sl)
        if dist == 0: return base_sl
        
        profit = (market_data.current_price - entry_price) if signal.direction == "BUY" else (entry_price - market_data.current_price)
        
        # 1. Institutional Breakeven (Protect at 1.5x RR)
        if profit >= dist * 1.5:
            # Move SL to Entry + small profit buffer to cover commissions
            atr = self._calculate_atr(market_data.m5_candles, 14)
            if signal.direction == "BUY":
                return max(base_sl, entry_price + (atr * 0.2))
            else:
                return min(base_sl, entry_price - (atr * 0.2))
                
        return base_sl

    def get_take_profit(self, signal: TradeSignal, market_data: MarketData) -> float:
        return getattr(signal, "take_profit", 0.0)

    def _create_signal(self, direction: str, price: float, confidence: float, reason: str) -> TradeSignal:
        sig = TradeSignal(direction=direction, price=price, confidence=confidence)
        sig.reasons = [reason]
        return sig

    def _apply_sniper_risk(self, sig, price, atr, m_high, m_low, last, session):
        sess_mult = 1.5 if session == "TOKYO" else 1.2 if session == "NEW_YORK" else 1.0
        buf = atr * self.sl_atr_mult * sess_mult
        
        if sig.direction == "BUY":
            raw_sl = min(float(last["low"]), float(m_low)) - buf
            sig.stop_loss = max(raw_sl, price - atr * self.sl_max_atr_mult)
            risk = price - sig.stop_loss
            sig.take_profit = price + risk * 3.0 # Institutional 1:3 target
        else:
            raw_sl = max(float(last["high"]), float(m_high)) + buf
            sig.stop_loss = min(raw_sl, price + atr * self.sl_max_atr_mult)
            risk = sig.stop_loss - price
            sig.take_profit = price - risk * 3.0

    def _is_rejection(self, c) -> Tuple[bool, bool]:
        body = abs(float(c.close) - float(c.open))
        rng = float(c.high) - float(c.low)
        if rng <= 0: return False, False
        bull = ((min(float(c.open), float(c.close)) - float(c.low)) / rng > self.rej_wick_ratio and body / rng < self.rej_body_ratio)
        bear = ((float(c.high) - max(float(c.open), float(c.close))) / rng > self.rej_wick_ratio and body / rng < self.rej_body_ratio)
        return bull, bear

    def _is_engulfing(self, p, c, direction) -> bool:
        if direction == "BUY":
            return c.close > c.open and p.close < p.open and c.close > p.open and c.open < p.close
        return c.close < c.open and p.close > p.open and c.close < p.open and c.open > p.close

    def _calculate_atr(self, candles, period) -> float:
        h, l, c = candles.high, candles.low, candles.close
        tr = np.maximum(h[1:] - l[1:], np.maximum(np.abs(h[1:] - c[:-1]), np.abs(l[1:] - c[:-1])))
        return float(np.mean(tr[-period:]))

    def _reset_daily_stats(self):
        self.session_traded_today = {}
        for s in self.consecutive_losses:
            self.consecutive_losses[s] = 0
