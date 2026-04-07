from core.common.types import MarketRegime, VolatilityStatus
import numpy as np
import logging

class RegimeInfo:
    def __init__(self, market_type: MarketRegime, volatility: VolatilityStatus, confidence: float, adx_val: float, atr_val: float):
        self.market_type = market_type
        self.volatility = volatility
        self.confidence = confidence
        self.adx = adx_val
        self.atr = atr_val

    def __repr__(self):
        return f"<Regime:{self.market_type.value} Vol:{self.volatility.value} Conf:{self.confidence:.2f} ADX:{self.adx:.1f} ATR:{self.atr:.5f}>"

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
            return RegimeInfo(MarketRegime.UNCERTAIN, VolatilityStatus.NORMAL, 0.0, 0.0, 0.0)

        # Optimization: Use pre-calculated indicators from IndicatorEngine if available
        adx = candles.get_indicator(f"adx_{self.adx_period}")[-1] if f"adx_{self.adx_period}" in candles.indicators else self._calculate_adx(candles)
        atr = candles.get_indicator(f"atr_{self.atr_period}")[-1] if f"atr_{self.atr_period}" in candles.indicators else self._calculate_atr(candles)
        
        # We still need the series for volatility ratio
        atr_series = candles.get_indicator(f"atr_{self.atr_period}") if f"atr_{self.atr_period}" in candles.indicators else self._ema(self._calculate_all_tr(candles), self.atr_period)
        avg_atr = np.mean(atr_series[-100:]) if len(atr_series) >= 100 else atr
        
        vol_ratio = atr / avg_atr if avg_atr > 0 else 1.0

        # Metric 1: Directional Type (ADX)
        market_type = MarketRegime.UNCERTAIN
        conf_type = 0.5
        if adx >= 25:
            market_type = MarketRegime.TREND
            conf_type = min(1.0, (adx - 20) / 30)
        elif adx <= 20:
            market_type = MarketRegime.RANGE
            conf_type = min(1.0, (25 - adx) / 10)
        
        # Metric 2: Volatility Status (ATR Ratio)
        vol_status = VolatilityStatus.NORMAL
        if vol_ratio > 1.8:
            vol_status = VolatilityStatus.HIGH
        elif vol_ratio < 0.6:
            vol_status = VolatilityStatus.LOW

        return RegimeInfo(market_type, vol_status, conf_type, adx, atr)

    def _calculate_adx(self, candles) -> float:
        h, l, c = candles.h, candles.l, candles.c
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
        h, l, c = candles.h, candles.l, candles.c
        tr = np.maximum(h[1:] - l[1:], np.maximum(np.abs(h[1:] - c[:-1]), np.abs(l[1:] - c[:-1])))
        return float(np.mean(tr[-self.atr_period:]))

    def _calculate_all_tr(self, candles):
        h, l, c = candles.h, candles.l, candles.c
        tr = np.maximum(h[1:] - l[1:], np.maximum(np.abs(h[1:] - c[:-1]), np.abs(l[1:] - c[:-1])))
        return tr

    def _ema(self, data: np.ndarray, period: int) -> np.ndarray:
        ema = np.zeros_like(data)
        # Initialize with mean of first `period` data points for unbiased start
        ema[0] = np.mean(data[:min(period, len(data))])
        multiplier = 2.0 / (period + 1)
        for i in range(1, len(data)):
            ema[i] = (data[i] - ema[i-1]) * multiplier + ema[i-1]
        return ema
