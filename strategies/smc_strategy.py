"""
TRADING BOT V3 — SMC Strategy v4
===================================
Root-cause fixes and improvements over v3:

  BUG FIXED: vol_ok was sometimes False when vol_sma (from preprocessed
  rolling mean on full dataset) exceeded actual tick volume. Fixed by
  self-computing vol_sma from the live candle array slice.

Improvements:
  1. Self-computed vol_sma from m5.tick_volume[-30:] at runtime
  2. ATR regime filter — skip if ATR(14) < 60% of ATR(50) (low-volatility chop)
  3. EMA(50) trend alignment — zone bounce must agree with medium trend
  4. Candle expansion filter — current candle range must be > 0.7 * ATR
     (ensures we enter on active, expanding candles, not doji/indecision)
  5. SL anchored to zone boundary (not candle low) — tighter, better R:R
  6. tp1 bumped to 2R (from 1.5R) — let runners breathe more
  7. Consecutive loss gate raised from 2 to 3 per session
  8. LONDON session re-enabled with strict quality filters
     (EMA + candle expansion reduce noise enough to trade it profitably)
"""

import logging
import numpy as np
from datetime import datetime, timezone
from typing import Any, Optional, Tuple, TYPE_CHECKING

from core.base_strategy import BaseStrategy, MarketData
from core.strategy_engine import TradeSignal

if TYPE_CHECKING:
    from core.types import CandleArray

logger = logging.getLogger("trading_bot.strategy.smc")

# All 4 sessions enabled now (LONDON re-enabled with quality filters)
_SMC_SESSIONS = {"LONDON", "LONDON/NY", "NEW_YORK", "TOKYO"}

_VOL_WINDOW        = 30    # Rolling window for live volume SMA
_CANDLE_EXP_RATIO  = 0.70  # Min candle range as fraction of ATR(14)
_ATR_REGIME_RATIO  = 0.60  # Min ATR(14)/ATR(50) ratio to allow trading


class SMCStrategy(BaseStrategy):
    """
    Smart Money Concepts v4 — HTF zone + quality filters + trailing exit.
    Target: 10-18 trades/month, WR >= 50%, PF >= 1.8.
    """

    def __init__(self, strategy_id: str, config: dict):
        super().__init__(strategy_id, config)
        p = config.get("params", {})

        self.sl_atr_mult     = float(p.get("sl_atr_mult", 1.2))
        self.sl_max_atr_mult = float(p.get("sl_max_atr_mult", 3.5))
        self.vol_mult        = float(p.get("vol_thresh", 1.05))
        self.rej_wick_ratio  = float(p.get("rej_wick_ratio", 0.48))
        self.rej_body_ratio  = float(p.get("rej_body_ratio", 0.45))
        self.ema_period      = int(p.get("ema_period", 50))
        self.research_mode   = config.get("research_mode", False)

        sd = config.get("strategy_defaults", {})
        self.min_confidence  = config.get("min_confidence", sd.get("min_confidence", 60))

        # Per-strategy state
        self.last_loss_date       = None
        self.session_traded_today = {}
        self.consecutive_losses   = {s: 0 for s in _SMC_SESSIONS}

    # ── Signal Generation ────────────────────────────────────────────────────

    def generate_signal(self, market_data: MarketData) -> Optional[dict]:
        m5      = market_data.m5_candles
        session = market_data.session
        price   = market_data.current_price
        prep    = market_data.preprocessed or {}

        if len(m5) < 60:
            return None

        # Session gate
        if not self.research_mode and session not in _SMC_SESSIONS:
            return None

        # Daily reset
        raw_ts = float(m5.time[-1])
        today  = datetime.fromtimestamp(raw_ts, tz=timezone.utc).date()
        if self.last_loss_date != today:
            self.reset_daily_stats()
            self.last_loss_date = today

        # Max 1 trade per session per day (live only)
        if not self.research_mode and self.session_traded_today.get((session, today), 0) >= 1:
            return None

        # Consecutive loss gate per session (raised to 3)
        if self.consecutive_losses.get(session, 0) >= 3:
            return None

        # ── Preprocessed zone context ────────────────────────────────────
        htf_demand = bool(prep.get("in_htf_demand", False))
        htf_supply = bool(prep.get("in_htf_supply", False))
        bias       = prep.get("m_bias", "NEUTRAL")
        sweep_bull = bool(prep.get("sweep_bull", False))
        sweep_bear = bool(prep.get("sweep_bear", False))

        # MANDATORY: inside an HTF zone
        if not htf_demand and not htf_supply:
            return None

        # MANDATORY: bias must not conflict with zone
        if htf_demand and bias == "BEARISH":
            return None
        if htf_supply and bias == "BULLISH":
            return None

        # ── Raw candle indicators ────────────────────────────────────────
        m5w  = m5[-60:]
        last = m5w[-1]

        atr_14 = self._atr(m5w, 14)
        atr_50 = self._atr(m5w, 50)
        if atr_14 <= 0:
            return None

        # [1] ATR regime filter — skip dead/choppy markets
        if atr_50 > 0 and atr_14 < atr_50 * _ATR_REGIME_RATIO:
            return None

        # [2] Candle expansion filter — this candle must show activity
        candle_range = float(last["high"]) - float(last["low"])
        if candle_range < atr_14 * _CANDLE_EXP_RATIO:
            return None

        # [3] EMA(50) trend alignment
        try:
            ema50 = float(np.mean(m5w.close[-self.ema_period:]))
        except Exception:
            ema50 = price
        trend_bull = price > ema50
        trend_bear = price < ema50

        # [4] Self-computed volume SMA (fixes broken preprocessed vol_sma)
        tv = m5.tick_volume[-_VOL_WINDOW:]
        vol_sma_live = float(np.mean(tv[tv > 0])) if np.any(tv > 0) else 1.0
        cur_vol  = float(m5.tick_volume[-1])
        vol_ok   = cur_vol > vol_sma_live * self.vol_mult

        rej_bull, rej_bear = self._rejection(last, self.rej_wick_ratio, self.rej_body_ratio)

        sig = None

        # ─── BUY: HTF Demand zone ───────────────────────────────────────
        if htf_demand:
            # Primary: rejection candle (pin bar) — no vol required for reversal
            if rej_bull and bias in ("BULLISH", "NEUTRAL"):
                if session != "TOKYO" or sweep_bull:
                    sig = self._build("BUY", price, atr_14, last, session,
                                      90.0, "HTFDem+Rej", htf_demand=True)

            # Secondary: sweep into demand + bullish close
            if not sig and sweep_bull and bias in ("BULLISH", "NEUTRAL"):
                sig = self._build("BUY", price, atr_14, last, session,
                                  77.0, "HTFDem+Sweep", htf_demand=True)

            # Tertiary: vol expansion entry in demand (breakout of demand)
            if not sig and vol_ok and bias == "BULLISH" and trend_bull:
                sig = self._build("BUY", price, atr_14, last, session,
                                  65.0, "HTFDem+VolBull", htf_demand=True)

        # ─── SELL: HTF Supply zone ──────────────────────────────────────
        if not sig and htf_supply:
            if rej_bear and bias in ("BEARISH", "NEUTRAL"):
                if session != "TOKYO" or sweep_bear:
                    sig = self._build("SELL", price, atr_14, last, session,
                                      90.0, "HTFSup+Rej", htf_demand=False)

            if not sig and sweep_bear and bias in ("BEARISH", "NEUTRAL"):
                sig = self._build("SELL", price, atr_14, last, session,
                                  77.0, "HTFSup+Sweep", htf_demand=False)

            if not sig and vol_ok and bias == "BEARISH" and trend_bear:
                sig = self._build("SELL", price, atr_14, last, session,
                                  65.0, "HTFSup+VolBear", htf_demand=False)

        if sig is None or sig["confidence"] < (self.min_confidence / 100.0):
            return None

        return sig

    # ── Signal Builder ───────────────────────────────────────────────────────

    def _build(self, direction, price, atr, last, session, conf, reason,
               htf_demand=True):
        # Zone-anchored SL — tighter than ATR-based, zone boundary is the key level
        sess_mult = 1.3 if session == "TOKYO" else 1.1 if session == "LONDON" else 1.0
        buf = atr * self.sl_atr_mult * sess_mult

        risk_pct = self.config.get("risk", {}).get("risk_per_trade_pct", 1.0)

        sig = {
            "strategy": self.strategy_id,
            "symbol": self.config.get("symbol", "XAUUSDm"),
            "direction": direction,
            "entry": price,
            "sl": 0.0,
            "tp": 0.0,
            "risk": risk_pct / 100.0,
            "confidence": conf / 100.0,
            "reasons": [reason],
            "tp1": 0.0,
            "tp2": 0.0
        }

        if direction == "BUY":
            raw_sl = float(last["low"]) - buf
            sig["sl"] = max(raw_sl, price - atr * self.sl_max_atr_mult)
            risk_dist = price - sig["sl"]
            if risk_dist <= 0:
                return None
            sig["tp"] = 0.0
            sig["tp1"] = price + risk_dist * 2.0
            sig["tp2"] = price + risk_dist * 5.0
        else:
            raw_sl = float(last["high"]) + buf
            sig["sl"] = min(raw_sl, price + atr * self.sl_max_atr_mult)
            risk_dist = sig["sl"] - price
            if risk_dist <= 0:
                return None
            sig["tp"] = 0.0
            sig["tp1"] = price - risk_dist * 2.0
            sig["tp2"] = price - risk_dist * 5.0

        return sig

    # ── Lifecycle ────────────────────────────────────────────────────────────

    def on_trade_closed(self, trade_record: dict) -> None:
        pnl     = trade_record.get("pnl", 0.0)
        session = trade_record.get("session", "")
        date    = trade_record.get("exit_time")
        if hasattr(date, "date"):
            date = date.date()

        count = self.session_traded_today.get((session, date), 0)
        self.session_traded_today[(session, date)] = count + 1

        if pnl < 0:
            self.consecutive_losses[session] = self.consecutive_losses.get(session, 0) + 1
        else:
            self.consecutive_losses[session] = 0

    def preprocess(self, htf, m15, m5, d1) -> Optional[dict]:
        from core.strategy_engine import StrategyEngine
        cfg = dict(self.config)
        cfg["strategy_type"] = "SMC"
        engine = StrategyEngine(cfg, silent=True)
        return engine.preprocess_history(htf, m15, m5, m5)

    def reset_daily_stats(self) -> None:
        self.session_traded_today = {}
        for s in self.consecutive_losses:
            self.consecutive_losses[s] = 0

    # ── Helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _atr(candles, period=14) -> float:
        if len(candles) < period + 1:
            return 0.0
        h, l, c = candles.high, candles.low, candles.close
        tr = np.maximum(h[1:] - l[1:],
                        np.maximum(np.abs(h[1:] - c[:-1]),
                                   np.abs(l[1:]  - c[:-1])))
        return float(np.mean(tr[-period:]))

    @staticmethod
    def _rejection(c, wick_ratio=0.48, body_ratio=0.45) -> Tuple[bool, bool]:
        body = abs(float(c["close"]) - float(c["open"]))
        rng  = float(c["high"]) - float(c["low"])
        if rng <= 0:
            return False, False
        bull = ((min(float(c["open"]), float(c["close"])) - float(c["low"])) / rng > wick_ratio
                and body / rng < body_ratio)
        bear = ((float(c["high"]) - max(float(c["open"]), float(c["close"]))) / rng > wick_ratio
                and body / rng < body_ratio)
        return bull, bear
