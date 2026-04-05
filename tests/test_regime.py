import numpy as np

from core.regime_detector import RegimeDetector, MarketRegime
from core.types import CandleArray


def _candles(n=120, step=0.1):
    time = np.arange(n, dtype=np.int64) * 300 + 1700000000
    close = 100 + np.arange(n) * step
    return CandleArray(
        time=time,
        open=close - 0.05,
        high=close + 0.10,
        low=close - 0.10,
        close=close,
        tick_volume=np.full(n, 100),
    )


def test_short_series_is_uncertain():
    det = RegimeDetector()
    info = det.detect(_candles(n=10))
    assert info.type == MarketRegime.UNCERTAIN


def test_trending_series_detected_as_trend_or_volatility_regime():
    det = RegimeDetector()
    info = det.detect(_candles(n=140, step=0.2))
    assert info.type in {MarketRegime.TREND, MarketRegime.HIGH_VOLATILITY, MarketRegime.LOW_VOLATILITY}
    assert isinstance(info.confidence, float)
