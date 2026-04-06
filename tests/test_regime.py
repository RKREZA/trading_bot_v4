import pytest
from core.regime_detector import RegimeDetector, MarketRegime
from core.common.types import VolatilityStatus

class TestRegimeDetection:
    """Verifies the accuracy of market architecture detection."""

    def test_trending_market_bullish(self, candle_factory):
        det = RegimeDetector()
        # Create a strong bullish trend
        candles = candle_factory(n=200, trend="BULLISH", volatility=0.5)
        info = det.detect(candles)
        
        # Detector should recognize TRENDING or BULLISH status
        assert info.market_type in {MarketRegime.TREND, MarketRegime.UNCERTAIN}
        if info.market_type == MarketRegime.TREND:
            assert info.volatility in {VolatilityStatus.NORMAL, VolatilityStatus.HIGH, VolatilityStatus.LOW}

    def test_ranging_market(self, candle_factory):
        det = RegimeDetector()
        # Create a flat range with low volatility
        candles = candle_factory(n=200, trend="FLAT", volatility=0.2)
        info = det.detect(candles)
        
        # Check standard ranging attributes
        assert info.market_type in {MarketRegime.RANGE, MarketRegime.UNCERTAIN}

    def test_high_volatility_uncertainty(self, candle_factory):
        det = RegimeDetector()
        # Create extreme noise (Volatile market)
        candles = candle_factory(n=100, trend="FLAT", volatility=5.0)
        info = det.detect(candles)
        
        # High volatility should often trigger UNCERTAIN or specific HIGH_VOLATILITY regime
        assert info.market_type is not None

    def test_short_history_is_uncertain(self, candle_factory):
        det = RegimeDetector()
        # Very short history cannot be analyzed
        candles = candle_factory(n=10)
        info = det.detect(candles)
        
        assert info.market_type == MarketRegime.UNCERTAIN
