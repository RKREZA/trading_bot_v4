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
        # Optimization: Strictly use pre-calculated indicators from IndicatorEngine (Step 4.2)
        adx_series = candles.get_indicator(f"adx_{self.adx_period}")
        atr_series = candles.get_indicator(f"atr_{self.atr_period}")
        
        if len(adx_series) == 0 or len(atr_series) == 0:
            return RegimeInfo(MarketRegime.UNCERTAIN, VolatilityStatus.NORMAL, 0.0, 0.0, 0.0)

        # 1. Trend Strength (ADX)
        adx = adx_series[-1]
        
        # 2. Volatility (ATR Ratio)
        atr = atr_series[-1]
        # Institutional Standard: Compare current ATR to its own 100-bar baseline
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
