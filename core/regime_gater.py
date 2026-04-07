import logging
from typing import Dict, Any, Optional
from core.common.types import MarketRegime, VolatilityStatus
from core.regime_detector import RegimeDetector, RegimeInfo

logger = logging.getLogger("trading_bot.gating")

class RegimeGater:
    """
    Institutional Gating Layer.
    Controls strategy activation and risk parameters based on the current Market Regime.
    """

    @staticmethod
    def is_strategy_allowed(strategy_type: str, market_type: MarketRegime) -> bool:
        """
        Implements the activation logic based on Directional Type:
        - TREND: Enables Trend Following + Breakout
        - RANGE: Enables Mean Reversion + Liquidity/Session
        """
        st_type = strategy_type.upper().replace("_", "")
        
        if market_type == MarketRegime.TREND:
            return "TREND" in st_type or "BREAKOUT" in st_type
        
        if market_type == MarketRegime.RANGE:
            return "MEANREVERSION" in st_type or "LIQUIDITY" in st_type or "SNIPER" in st_type
            
        return True # Default to enabled if UNCERTAIN or other

    @staticmethod
    def get_risk_multiplier(volatility: VolatilityStatus) -> float:
        """
        Implements risk reduction based on Volatility Status:
        - HIGH: Reduce risk (0.5x)
        """
        if volatility == VolatilityStatus.HIGH:
            return 0.5
        return 1.0

    @staticmethod
    def get_confidence_buffer(volatility: VolatilityStatus) -> float:
        """
        Implements frequency reduction based on Volatility Status:
        - LOW: Increase required confidence (+0.15)
        """
        if volatility == VolatilityStatus.LOW:
            return 0.15
        return 0.0

    @staticmethod
    def is_drawdown_gated(max_dd: float, threshold: float = 15.0) -> bool:
        """
        Hard Institutional Gate: Disable strategy if Max DD exceeds threshold.
        Default threshold: 15.0%
        """
        if max_dd > threshold:
            return True
        return False
