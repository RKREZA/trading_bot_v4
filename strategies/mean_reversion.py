"""
TRADING BOT V4 — Mean Reversion Strategy (SMC V4 Migrated)
========================================================
Institutional Smart Money Concepts: HTF Zones, Liquidity Sweeps, and Rejections.
"""

import logging
import numpy as np
from datetime import datetime, timezone
from typing import Any, Optional, Tuple

from core.base_strategy import BaseStrategy, MarketData
from core.types import TradeSignal

logger = logging.getLogger("trading_bot.strategy.smc")

_SMC_SESSIONS = {"LONDON", "LONDON/NY", "NEW_YORK", "TOKYO"}
_VOL_WINDOW        = 30 
_CANDLE_EXP_RATIO  = 0.70 
_ATR_REGIME_RATIO  = 0.60 

class MeanReversionStrategy(BaseStrategy):
    """
    Smart Money Concepts v4 — Migrated to Institutional V4 Mean Reversion.
    Focuses on HTF zone bounces, sweeps, and high-quality reversals.
    """

    def __init__(self, strategy_id: str, config: dict):
        super().__init__(strategy_id, config)
        p = config.get("params", {})

        self.sl_atr_mult     = float(p.get("sl_atr_mult", 1.5))
        self.sl_max_atr_mult = float(p.get("sl_max_atr_mult", 3.0))
        self.vol_mult        = float(p.get("vol_thresh", 1.10))
        self.rej_wick_ratio  = float(p.get("rej_wick_ratio", 0.48))
        self.rej_body_ratio  = float(p.get("rej_body_ratio", 0.45))
        self.ema_period      = int(p.get("ema_period", 50))
        self.research_mode   = config.get("research_mode", False)

        sd = config.get("strategy_defaults", {})
        self.min_confidence  = float(config.get("min_confidence", sd.get("min_confidence", 0.60)))

        # Per-strategy state
        self.last_loss_date       = None
        self.session_traded_today = {}
        self.consecutive_losses   = {s: 0 for s in _SMC_SESSIONS}

    def generate_signal(self, market_data: MarketData) -> Optional[TradeSignal]:
        m5      = market_data.m5_candles
        session = market_data.session
        price   = market_data.current_price
        prep    = market_data.preprocessed or {}

        if len(m5) < 60:
            return None
        if not self.research_mode and session not in _SMC_SESSIONS:
            return None

        # Daily reset logic
        raw_ts = float(m5.time[-1])
        today  = datetime.fromtimestamp(raw_ts, tz=timezone.utc).date()
        if self.last_loss_date != today:
            self._reset_daily_stats()
            self.last_loss_date = today

        # Daily trade limit
        if not self.research_mode and self.session_traded_today.get((session, today), 0) >= 1:
            return None
        if self.consecutive_losses.get(session, 0) >= 3:
            return None

        # SMC Context
        htf_demand = bool(prep.get("in_htf_demand", False))
        htf_supply = bool(prep.get("in_htf_supply", False))
        bias       = prep.get("m_bias", "NEUTRAL")
        sweep_bull = bool(prep.get("sweep_bull", False))
        sweep_bear = bool(prep.get("sweep_bear", False))

        if not htf_demand and not htf_supply:
            return None

        # Filter: Inside zone, check for Conflicting Bias
        if (htf_demand and bias == "BEARISH") or (htf_supply and bias == "BULLISH"):
            return None

        m5w = m5[-60:]
        last = m5w[-1]

        atr_14 = self._calculate_atr(m5w, 14)
        atr_50 = self._calculate_atr(m5w, 50)
        
        if atr_14 <= 0 or (atr_50 > 0 and atr_14 < atr_50 * _ATR_REGIME_RATIO):
            return None

        # Candle Expansion Check
        candle_range = float(last.high) - float(last.low)
        if candle_range < atr_14 * _CANDLE_EXP_RATIO:
            return None

        # Trend and Volume
        ema50 = np.mean(m5w.close[-self.ema_period:])
        trend_bull = price > ema50
        trend_bear = price < ema50
        
        tv = m5.tick_volume[-_VOL_WINDOW:]
        vol_sma_live = np.mean(tv[tv > 0]) if np.any(tv > 0) else 1.0
        vol_ok = float(m5.tick_volume[-1]) > vol_sma_live * self.vol_mult

        rej_bull, rej_bear = self._is_rejection(last)

        sig = None
        
        # BUY: HTF Demand Zone
        if htf_demand:
            if rej_bull and bias != "BEARISH":
                sig = self._create_signal("BUY", price, 0.90, "HTFDem+Rej")
            elif not sig and sweep_bull and bias != "BEARISH":
                sig = self._create_signal("BUY", price, 0.77, "HTFDem+Sweep")
            elif not sig and vol_ok and bias == "BULLISH" and trend_bull:
                sig = self._create_signal("BUY", price, 0.65, "HTFDem+VolBull")

        # SELL: HTF Supply Zone
        if not sig and htf_supply:
            if rej_bear and bias != "BULLISH":
                sig = self._create_signal("SELL", price, 0.90, "HTFSup+Rej")
            elif not sig and sweep_bear and bias != "BULLISH":
                sig = self._create_signal("SELL", price, 0.77, "HTFSup+Sweep")
            elif not sig and vol_ok and bias == "BEARISH" and trend_bear:
                sig = self._create_signal("SELL", price, 0.65, "HTFSup+VolBear")

        if sig and sig.confidence >= self.min_confidence:
            self._apply_smc_risk(sig, price, atr_14, last, session)
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
        
        # Protect at 2.0x RR (SMC is a lower-winrate, but higher RR strategy)
        if profit >= dist * 2.0:
            atr = self._calculate_atr(market_data.m5_candles, 14)
            if signal.direction == "BUY":
                return max(base_sl, entry_price + (atr * 0.1))
            else:
                return min(base_sl, entry_price - (atr * 0.1))
                
        return base_sl

    def get_take_profit(self, signal: TradeSignal, market_data: MarketData) -> float:
        return getattr(signal, "take_profit", 0.0)

    def _create_signal(self, direction: str, price: float, confidence: float, reason: str) -> TradeSignal:
        sig = TradeSignal(direction=direction, price=price, confidence=confidence)
        sig.reasons = [reason]
        return sig

    def _apply_smc_risk(self, sig, price, atr, last, session):
        sess_mult = 1.3 if session == "TOKYO" else 1.1 if session == "LONDON" else 1.0
        buf = atr * self.sl_atr_mult * sess_mult
        
        if sig.direction == "BUY":
            raw_sl = float(last.low) - buf
            sig.stop_loss = max(raw_sl, price - atr * self.sl_max_atr_mult)
            risk = price - sig.stop_loss
            sig.take_profit = price + risk * 3.5 
        else:
            raw_sl = float(last.high) + buf
            sig.stop_loss = min(raw_sl, price + atr * self.sl_max_atr_mult)
            risk = sig.stop_loss - price
            sig.take_profit = price - risk * 3.5

    def _is_rejection(self, c) -> Tuple[bool, bool]:
        body = abs(float(c.close) - float(c.open))
        rng = float(c.high) - float(c.low)
        if rng <= 0: return False, False
        bull = ((min(float(c.open), float(c.close)) - float(c.low)) / rng > self.rej_wick_ratio and body / rng < self.rej_body_ratio)
        bear = ((float(c.high) - max(float(c.open), float(c.close))) / rng > self.rej_wick_ratio and body / rng < self.rej_body_ratio)
        return bull, bear

    def _calculate_atr(self, candles, period) -> float:
        h, l, c = candles.high, candles.low, candles.close
        tr = np.maximum(h[1:] - l[1:], np.maximum(np.abs(h[1:] - c[:-1]), np.abs(l[1:] - c[:-1])))
        return float(np.mean(tr[-period:]))

    def _reset_daily_stats(self):
        self.session_traded_today = {}
        for s in self.consecutive_losses:
            self.consecutive_losses[s] = 0
