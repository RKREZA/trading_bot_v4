import logging
import numpy as np
from typing import Optional, Dict, Any

from core.base_strategy import BaseStrategy, MarketData
from core.common.types import TradeSignal, CandleArray

logger = logging.getLogger("trading_bot.smc_strategy")


class SMCStrategy(BaseStrategy):

    def __init__(self, strategy_id: str = "SMC", config: dict = None):
        super().__init__(strategy_id, config or {})
        strat_cfg = self.get_strat_config()
        self.lookback = int(strat_cfg.get("lookback", 50))
        self.bos_window = int(strat_cfg.get("bos_window", 15))
        self.fvg_scan_depth = int(strat_cfg.get("fvg_scan_depth", 15))
        self.rr_target = float(strat_cfg.get("rr_target", 5.0))
        self.ob_scan_depth = int(strat_cfg.get("ob_scan_depth", 10))

    def _get_primary_candles(self, market_data: MarketData) -> CandleArray:
        tf = self.config.get("backtest", {}).get("timeframe", "M5")
        if tf in ("M15", "M30"):
            return market_data.m15_candles
        if tf in ("H1", "H4"):
            return market_data.htf_candles
        return market_data.m5_candles

    def generate_signal(self, market_data: MarketData) -> Optional[TradeSignal]:
        candles = self._get_primary_candles(market_data)
        if len(candles) < self.lookback:
            return None

        if not self.is_spread_safe(market_data):
            return None
        if not self.is_volatility_safe(market_data):
            return None

        bos = self._detect_last_bos(candles)
        if not bos:
            return None

        direction = bos["direction"]
        fvg = self._detect_fvg(candles, direction)
        if not fvg:
            return None

        ob = self._detect_order_block(candles, direction)

        if direction == "BUY":
            entry = fvg["top"]
            sl_base = fvg["bottom"]
            if ob and ob["low"] < sl_base:
                sl_base = ob["low"]
            buffer = (entry - sl_base) * 0.1
            sl = sl_base - buffer
            tp = entry + (entry - sl) * self.rr_target
        else:
            entry = fvg["bottom"]
            sl_base = fvg["top"]
            if ob and ob["high"] > sl_base:
                sl_base = ob["high"]
            buffer = (sl_base - entry) * 0.1
            sl = sl_base + buffer
            tp = entry - (sl - entry) * self.rr_target

        risk = abs(entry - sl)
        if risk <= 0:
            return None
        rr = abs(entry - tp) / risk
        if rr < self.min_rr:
            self.last_rejection_reason = f"RR {rr:.2f} < {self.min_rr}"
            return None

        confidence = self._calc_confidence(market_data, direction, fvg, ob, bos)
        if confidence < self.min_confidence:
            self.last_rejection_reason = f"Confidence {confidence:.2f} < {self.min_confidence}"
            return None

        reasons = [f"BOS_{direction}", "FVG_confluence"]
        if ob:
            reasons.append("OB_confluence")

        return TradeSignal(
            direction=direction,
            symbol=market_data.symbol,
            price=entry,
            stop_loss=sl,
            take_profit=tp,
            confidence=confidence,
            timestamp=market_data.timestamp,
            session=market_data.session,
            rr_ratio=rr,
            execution_id=self.get_execution_id(market_data.timestamp),
            strategy_id=self.strategy_id,
            reasons=reasons,
            metadata={"bos_level": bos["level"], "fvg_range": [fvg["bottom"], fvg["top"]]},
        )

    def get_metrics(self, market_data: MarketData) -> Dict[str, Any]:
        candles = self._get_primary_candles(market_data)
        bos = self._detect_last_bos(candles) if len(candles) >= self.lookback else None
        return {
            "strategy_id": self.strategy_id,
            "enabled": self.enabled,
            "last_bos": bos,
            "spread_safe": self.is_spread_safe(market_data),
            "volatility_safe": self.is_volatility_safe(market_data),
            "last_rejection": self.last_rejection_reason,
        }

    def _detect_last_bos(self, candles: CandleArray) -> Optional[Dict[str, Any]]:
        n = len(candles)
        if n < self.bos_window + 1:
            return None

        window_high = np.max(candles.h[n - self.bos_window - 1 : n - 1])
        window_low = np.min(candles.l[n - self.bos_window - 1 : n - 1])
        last_close = candles.c[n - 1]

        if last_close > window_high:
            return {"direction": "BUY", "level": float(window_high)}
        if last_close < window_low:
            return {"direction": "SELL", "level": float(window_low)}
        return None

    def _detect_fvg(self, candles: CandleArray, direction: str) -> Optional[Dict[str, float]]:
        n = len(candles)
        scan_end = max(2, n - self.fvg_scan_depth)

        for i in range(n - 1, scan_end, -1):
            if i < 2:
                break
            c1_high = candles.h[i - 2]
            c1_low = candles.l[i - 2]
            c3_high = candles.h[i]
            c3_low = candles.l[i]

            if direction == "BUY" and c3_low > c1_high:
                return {"bottom": float(c1_high), "top": float(c3_low)}
            if direction == "SELL" and c3_high < c1_low:
                return {"top": float(c1_low), "bottom": float(c3_high)}

        return None

    def _detect_order_block(self, candles: CandleArray, direction: str) -> Optional[Dict[str, float]]:
        n = len(candles)
        scan_end = max(2, n - self.ob_scan_depth)

        for i in range(n - 2, scan_end, -1):
            if direction == "BUY":
                if candles.c[i] < candles.o[i] and candles.c[i + 1] > candles.o[i + 1]:
                    body = abs(candles.c[i + 1] - candles.o[i + 1])
                    prev_body = abs(candles.c[i] - candles.o[i])
                    if body > prev_body * 1.5:
                        return {"high": float(candles.h[i]), "low": float(candles.l[i])}
            else:
                if candles.c[i] > candles.o[i] and candles.c[i + 1] < candles.o[i + 1]:
                    body = abs(candles.c[i + 1] - candles.o[i + 1])
                    prev_body = abs(candles.c[i] - candles.o[i])
                    if body > prev_body * 1.5:
                        return {"high": float(candles.h[i]), "low": float(candles.l[i])}
        return None

    def get_parameter_grid(self) -> Dict[str, list]:
        return {
            "rr_target": [4.0, 5.0, 6.0, 7.0],
            "bos_window": [10, 15, 20],
            "fvg_scan_depth": [10, 15, 20],
        }

    def _calc_confidence(self, market_data: MarketData, direction: str, fvg: dict, ob: Optional[dict] = None, bos: Optional[dict] = None) -> float:
        score = 0.55

        trend = self.get_ema_trend(market_data.m15_candles)
        if (direction == "BUY" and trend == 1) or (direction == "SELL" and trend == -1):
            score += 0.12

        if self.check_mtf_consensus(market_data):
            score += 0.10

        fvg_size = abs(fvg["top"] - fvg["bottom"])
        if market_data.point > 0:
            fvg_pips = fvg_size / market_data.point
            if fvg_pips > 20:
                score += 0.05
            if fvg_pips > 50:
                score += 0.05

        if ob:
            score += 0.08

        if bos and market_data.point > 0:
            bos_dist = abs(market_data.current_price - bos["level"]) / market_data.point
            if bos_dist > 50:
                score += 0.05

        candles = self._get_primary_candles(market_data)
        if len(candles) > 20:
            recent_vol = np.mean(candles.v[-20:])
            last_vol = candles.v[-1]
            if last_vol > recent_vol * 1.3:
                score += 0.05

        rsi = market_data.m5_candles.rsi(14)
        if len(rsi) > 0 and not np.isnan(rsi[-1]):
            rsi_val = rsi[-1]
            if direction == "BUY" and rsi_val < 45:
                score += 0.05
            elif direction == "SELL" and rsi_val > 55:
                score += 0.05

        return min(1.0, score)
