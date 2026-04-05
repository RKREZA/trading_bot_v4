from enum import Enum
import numpy as np
import logging

class MarketRegime(Enum):
    TREND           = "TREND"
    RANGE           = "RANGE"
    HIGH_VOLATILITY = "HIGH_VOLATILITY"
    LOW_VOLATILITY  = "LOW_VOLATILITY"
    UNCERTAIN       = "UNCERTAIN"

class RegimeInfo:
    def __init__(self, regime_type: MarketRegime, confidence: float, adx_val: float, atr_val: float):
        self.type = regime_type
        self.confidence = confidence
        self.adx = adx_val
        self.atr = atr_val

    def __repr__(self):
        return f"<Regime:{self.type.value} Conf:{self.confidence:.2f} ADX:{self.adx:.1f} ATR:{self.atr:.5f}>"

class RegimeDetector:
    """
    Institutional Regime Detection Layer.
    Uses ADX for Trend vs Range and ATR for volatility classification.
    """

    def __init__(self, adx_period: int = 14, atr_period: int = 14):
        self.adx_period = adx_period
        self.atr_period = atr_period
        self.logger = logging.getLogger("trading_bot.regime_detector")

    def detect(self, candles) -> RegimeInfo:
        if len(candles) < self.adx_period * 2:
            return RegimeInfo(MarketRegime.UNCERTAIN, 0.0, 0.0, 0.0)

        adx = self._calculate_adx(candles)
        atr = self._calculate_atr(candles)
        
        all_tr = self._calculate_all_tr(candles)
        atr_series = self._ema(all_tr, self.atr_period)
        avg_atr = np.mean(atr_series[-100:]) if len(atr_series) >= 100 else atr
        
        vol_ratio = atr / avg_atr if avg_atr > 0 else 1.0

        regime = MarketRegime.UNCERTAIN
        conf = 0.5
        
        if adx >= 25:
            regime = MarketRegime.TREND
            conf = min(1.0, (adx - 20) / 30)
        elif adx <= 20:
            regime = MarketRegime.RANGE
            conf = min(1.0, (25 - adx) / 10)
        
        if vol_ratio > 1.8:
            regime = MarketRegime.HIGH_VOLATILITY
            conf = 0.9
        elif vol_ratio < 0.6:
            regime = MarketRegime.LOW_VOLATILITY
            conf = 0.8

        return RegimeInfo(regime, conf, adx, atr)

    def _calculate_adx(self, candles) -> float:
        h, l, c = candles.high, candles.low, candles.close
        up_move = h[1:] - h[:-1]
        down_move = l[:-1] - l[1:]
        
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
        
        tr = np.maximum(h[1:] - l[1:], np.maximum(np.abs(h[1:] - c[:-1]), np.abs(l[1:] - c[:-1])))
        
        atr = self._ema(tr, self.adx_period)
        plus_di = 100 * self._ema(plus_dm, self.adx_period) / (atr + 1e-10)
        minus_di = 100 * self._ema(minus_dm, self.adx_period) / (atr + 1e-10)
        
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        adx = self._ema(dx, self.adx_period)
        return float(adx[-1])

    def _calculate_atr(self, candles) -> float:
        h, l, c = candles.high, candles.low, candles.close
        tr = np.maximum(h[1:] - l[1:], np.maximum(np.abs(h[1:] - c[:-1]), np.abs(l[1:] - c[:-1])))
        return float(np.mean(tr[-self.atr_period:]))

    def _calculate_all_tr(self, candles):
        h, l, c = candles.high, candles.low, candles.close
        tr = np.maximum(h[1:] - l[1:], np.maximum(np.abs(h[1:] - c[:-1]), np.abs(l[1:] - c[:-1])))
        return tr

    def _ema(self, data, period):
        alpha = 2.0 / (period + 1)
        ema = np.zeros_like(data)
        ema[0] = data[0]
        for i in range(1, len(data)):
            ema[i] = (data[i] - ema[i-1]) * alpha + ema[i-1]
        return ema
