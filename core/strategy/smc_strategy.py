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
        self.lookback = int(strat_cfg.get("lookback", 100))
        self.bos_window = int(strat_cfg.get("bos_window", 20))
        self.fvg_scan_depth = int(strat_cfg.get("fvg_scan_depth", 10))
        self.rr_target = float(strat_cfg.get("rr_target", 2.5))

    def generate_signal(self, market_data: MarketData) -> Optional[TradeSignal]:
        candles = market_data.m5_candles
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

        if direction == "BUY":
            entry = fvg["top"]
            sl = fvg["bottom"]
            tp = entry + (entry - sl) * self.rr_target
        else:
            entry = fvg["bottom"]
            sl = fvg["top"]
            tp = entry - (sl - entry) * self.rr_target

        rr = abs(entry - tp) / abs(entry - sl) if abs(entry - sl) > 0 else 0
        if rr < self.min_rr:
            self.last_rejection_reason = f"RR {rr:.2f} < {self.min_rr}"
            return None

        confidence = self._calc_confidence(market_data, direction, fvg)
        if confidence < self.min_confidence:
            self.last_rejection_reason = f"Confidence {confidence:.2f} < {self.min_confidence}"
            return None

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
            reasons=[f"BOS_{direction}", "FVG_confluence"],
            metadata={"bos_level": bos["level"], "fvg_range": [fvg["bottom"], fvg["top"]]},
        )

    def get_metrics(self, market_data: MarketData) -> Dict[str, Any]:
        candles = market_data.m5_candles
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

    def _calc_confidence(self, market_data: MarketData, direction: str, fvg: dict) -> float:
        score = 0.5

        trend = self.get_ema_trend(market_data.m15_candles)
        if (direction == "BUY" and trend == 1) or (direction == "SELL" and trend == -1):
            score += 0.15

        if self.check_mtf_consensus(market_data):
            score += 0.15

        fvg_size = abs(fvg["top"] - fvg["bottom"])
        if market_data.point > 0:
            fvg_pips = fvg_size / market_data.point
            if fvg_pips > 30:
                score += 0.1

        rsi = market_data.m5_candles.rsi(14)
        if len(rsi) > 0 and not np.isnan(rsi[-1]):
            rsi_val = rsi[-1]
            if direction == "BUY" and rsi_val < 40:
                score += 0.1
            elif direction == "SELL" and rsi_val > 60:
                score += 0.1

        return min(1.0, score)
