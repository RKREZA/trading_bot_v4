"""
TRADING BOT V3 — Sniper Strategy v4.1 (Cooldown Bug Fixed)
=====================================
Root-cause fixes from 6-month backtest diagnostic:

  BUG FIXED: trade_counter was NEVER incremented if the cooldown gate returned None.
  This caused a permanent infinite-loop circuit breaker after the first losing trade.
  
Improvements over V3:
  1. Self-computed volume SMA from live candle data (not preprocessed)
  2. T3 no longer requires vol_ok — trend continuation doesn't need vol spike
  3. ATR regime filter — skip if ATR < 60% of ATR(50) (choppy market)
  4. EMA(50) trend alignment — entry must agree with medium-term trend
  5. Backtest mode (research_mode=True) removes session throttles
  6. Partial closes at 1.5R and 4R (give trades more room)
"""

import logging
import numpy as np
from datetime import datetime, timezone
from typing import Any, Optional, Tuple, TYPE_CHECKING

from core.base_strategy import BaseStrategy, MarketData
from core.strategy_engine import TradeSignal

if TYPE_CHECKING:
    from core.types import CandleArray

logger = logging.getLogger("trading_bot.strategy.sniper")

_ACTIVE_SESSIONS = {"LONDON", "NEW_YORK", "LONDON/NY", "TOKYO"}
_CONF_T1 = 88.0
_CONF_T2 = 72.0
_CONF_T3 = 60.0
_VOL_WINDOW = 30   # Rolling window for self-computed volume SMA


class SniperStrategy(BaseStrategy):
    """
    Pure Price Action Sniper v4.1 — tiered entry, trailing-only exit.
    Restored from V4 with cooldown block bug removed.
    """

    def __init__(self, strategy_id: str, config: dict):
        super().__init__(strategy_id, config)
        p = config.get("params", {})

        self.swing_lookback   = int(p.get("swing_lookback", 7))
        self.sl_atr_mult      = float(p.get("sl_atr_mult", 1.0))
        self.sl_max_atr_mult  = float(p.get("sl_max_atr_mult", 3.0))
        self.rej_wick_ratio   = float(p.get("rej_wick_ratio", 0.50))
        self.rej_body_ratio   = float(p.get("rej_body_ratio", 0.42))
        self.vol_mult         = float(p.get("vol_mult", 1.10))
        self.ema_period       = int(p.get("ema_period", 50))
        self.atr_regime_ratio = float(p.get("atr_regime_ratio", 0.60))

        sd = config.get("strategy_defaults", {})
        self.min_confidence   = config.get("min_confidence", sd.get("min_confidence", 55))
        self.cooldown_candles = int(config.get("cooldown_candles", sd.get("cooldown_candles", 2)))
        self.research_mode    = config.get("research_mode", False)

        # Per-strategy mutable state
        self.trade_counter        = 0
        self.last_stop_index      = -999
        self.last_loss_date       = None
        self.session_traded_today = {}
        self.consecutive_losses   = {s: 0 for s in _ACTIVE_SESSIONS}

    # ── Signal Generation ────────────────────────────────────────────────────

    def generate_signal(self, market_data: MarketData) -> Optional[TradeSignal]:
        m5      = market_data.m5_candles
        session = market_data.session
        price   = market_data.current_price
        prep    = market_data.preprocessed or {}

        if len(m5) < 60:
            return None
        if not self.research_mode and session not in _ACTIVE_SESSIONS:
            return None

        # Daily reset
        raw_ts = float(m5.time[-1])
        today  = datetime.fromtimestamp(raw_ts, tz=timezone.utc).date()
        if self.last_loss_date != today:
            self.reset_daily_stats()
            self.last_loss_date = today

        # BUG FIX: Increment trade counter BEFORE checking cooldown!
        self.trade_counter += 1

        # Cooldown after loss (live only)
        if not self.research_mode:
            if self.trade_counter - self.last_stop_index < self.cooldown_candles:
                return None

        # Max 2 trades per session per day (live only) — raised from 1
        sess_day = (session, today)
        if not self.research_mode:
            if self.session_traded_today.get(sess_day, 0) >= 2:
                return None

        # Consecutive loss gate per session (raised from 2 to 3)
        if self.consecutive_losses.get(session, 0) >= 3:
            return None

        # Tokyo no-neutral gate (still useful)
        bias = prep.get("m_bias", "NEUTRAL")
        if session == "TOKYO" and bias == "NEUTRAL":
            return None

        # ── Raw-candle indicators ────────────────────────────────────────
        m5w  = m5[-60:]
        last = m5w[-1]
        prev = m5w[-2]

        atr_14 = self._atr(m5w, 14)
        atr_50 = self._atr(m5w, 50)
        if atr_14 <= 0:
            return None

        # [1] ATR regime filter — skip choppy/dead markets
        if atr_50 > 0 and atr_14 < atr_50 * self.atr_regime_ratio:
            return None

        # [2] EMA(50) trend filter
        try:
            ema50 = float(np.convolve(m5w.close, np.ones(self.ema_period) / self.ema_period, mode='valid')[-1])
        except Exception:
            ema50 = float(np.mean(m5w.close[-self.ema_period:]))
        trend_bull = price > ema50
        trend_bear = price < ema50

        # [3] Daily VWAP Filter
        # Find index of the first candle of "today"
        today_ts = datetime(today.year, today.month, today.day, tzinfo=timezone.utc).timestamp()
        idx_today = int(np.searchsorted(m5.time, today_ts, side="left"))
        idx_today = max(0, min(idx_today, len(m5.time) - 1))
        today_candles = m5[idx_today:]
        
        vwap = price
        if len(today_candles) > 0:
            tp = (today_candles.high + today_candles.low + today_candles.close) / 3.0
            vol = today_candles.tick_volume
            if np.sum(vol) > 0:
                vwap = float(np.sum(tp * vol) / np.sum(vol))

        vwap_bull = price > vwap
        vwap_bear = price < vwap

        # [4] Self-computed volume SMA
        tv = m5.tick_volume[-_VOL_WINDOW:]
        vol_sma_live = float(np.mean(tv[tv > 0])) if np.any(tv > 0) else 1.0
        cur_vol  = float(m5.tick_volume[-1])
        vol_ok   = cur_vol > vol_sma_live * self.vol_mult

        # [5] LONDON Session Choppiness Gate
        # If London, the signal candle must expand significantly to prove institutional flow.
        candle_range = float(last["high"]) - float(last["low"])
        if session == "LONDON" and atr_14 > 0:
            if candle_range < atr_14 * 0.80:
                return None

        # [4] Preprocessed signals (zone context and swing data)
        m_high     = prep.get("m_high", price + 9999)
        m_low      = prep.get("m_low",  price - 9999)
        sweep_bull = bool(prep.get("sweep_bull", False))
        sweep_bear = bool(prep.get("sweep_bear", False))

        rej_bull, rej_bear = self._rejection(last, self.rej_wick_ratio, self.rej_body_ratio)
        engulf_b = self._engulf(prev, last, "BUY")
        engulf_s = self._engulf(prev, last, "SELL")

        sig = None

        # ─── BUY signals ────────────────────────────────────────────────
        if bias in ("BULLISH", "NEUTRAL"):
            # T1: Sweep + rejection (require VWAP alignment to prevent fading massive selloffs)
            if (sweep_bull or float(last["low"]) < float(m_low)) and (rej_bull or engulf_b) and vwap_bull:
                sig = self._build("BUY", price, atr_14, m_high, m_low, last,
                                  session, _CONF_T1, "T1:Sweep+Rej")

            # T2: Rejection + volume + BULLISH bias + EMA agreement + VWAP
            elif not sig and rej_bull and vol_ok and bias == "BULLISH" and trend_bull and vwap_bull:
                sig = self._build("BUY", price, atr_14, m_high, m_low, last,
                                  session, _CONF_T2, "T2:Rej+Vol+Bull")

            # T3: Bullish engulf + BULLISH bias + EMA + VWAP
            elif not sig and bias == "BULLISH" and engulf_b and trend_bull and vwap_bull:
                sig = self._build("BUY", price, atr_14, m_high, m_low, last,
                                  session, _CONF_T3, "T3:BullEngulf")

        # ─── SELL signals ───────────────────────────────────────────────
        if not sig and bias in ("BEARISH", "NEUTRAL"):
            if (sweep_bear or float(last["high"]) > float(m_high)) and (rej_bear or engulf_s) and vwap_bear:
                sig = self._build("SELL", price, atr_14, m_high, m_low, last,
                                  session, _CONF_T1, "T1:Sweep+Rej")
            elif not sig and rej_bear and vol_ok and bias == "BEARISH" and trend_bear and vwap_bear:
                sig = self._build("SELL", price, atr_14, m_high, m_low, last,
                                  session, _CONF_T2, "T2:Rej+Vol+Bear")
            elif not sig and bias == "BEARISH" and engulf_s and trend_bear and vwap_bear:
                sig = self._build("SELL", price, atr_14, m_high, m_low, last,
                                  session, _CONF_T3, "T3:BearEngulf")

        if sig is None or sig.confidence < self.min_confidence:
            return None

        return sig

    # ── Signal Builder ───────────────────────────────────────────────────────

    def _build(self, direction, price, atr, m_high, m_low, last, session, conf, reason):
        sess_mult = 1.5 if session == "TOKYO" else 1.2 if session == "NEW_YORK" else 1.0
        buf = atr * self.sl_atr_mult * sess_mult

        sig = TradeSignal(direction, price, 0.0, 0.0, session=session)
        sig.confidence = conf
        sig.reasons    = [reason]

        if direction == "BUY":
            raw_sl        = min(float(last["low"]), float(m_low)) - buf
            sig.stop_loss = max(raw_sl, price - atr * self.sl_max_atr_mult)
            risk          = price - sig.stop_loss
            if risk <= 0:
                return None
            sig.take_profit = 0.0
            sig.tp1_price   = price + risk * 1.0   # 50% partial at 1.0R to guarantee safety
            sig.tp2_price   = price + risk * 10.0  # Massive runner up to 10.0R
        else:
            raw_sl        = max(float(last["high"]), float(m_high)) + buf
            sig.stop_loss = min(raw_sl, price + atr * self.sl_max_atr_mult)
            risk          = sig.stop_loss - price
            if risk <= 0:
                return None
            sig.take_profit = 0.0
            sig.tp1_price   = price - risk * 1.0
            sig.tp2_price   = price - risk * 10.0

        sig.rr_ratio        = 0.0
        sig.confluence_score = int(conf / 10)
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
            self.last_stop_index = self.trade_counter
            self.consecutive_losses[session] = self.consecutive_losses.get(session, 0) + 1
        else:
            self.consecutive_losses[session] = 0

    def preprocess(self, htf, m15, m5, d1) -> Optional[dict]:
        from core.strategy_engine import StrategyEngine
        engine = StrategyEngine(self.config, silent=True)
        engine.swing_lookback = self.swing_lookback
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
    def _rejection(c, wick_ratio=0.50, body_ratio=0.42) -> Tuple[bool, bool]:
        body = abs(float(c["close"]) - float(c["open"]))
        rng  = float(c["high"]) - float(c["low"])
        if rng <= 0:
            return False, False
        bull = ((min(float(c["open"]), float(c["close"])) - float(c["low"])) / rng > wick_ratio
                and body / rng < body_ratio)
        bear = ((float(c["high"]) - max(float(c["open"]), float(c["close"]))) / rng > wick_ratio
                and body / rng < body_ratio)
        return bull, bear

    @staticmethod
    def _engulf(p, c, direction) -> bool:
        po = float(p["open"]); pc = float(p["close"])
        co = float(c["open"]); cc = float(c["close"])
        if direction == "BUY":
            return cc > co and pc < po and cc > po and co < pc
        return cc < co and pc > po and cc < po and co > pc
