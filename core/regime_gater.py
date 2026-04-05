import logging
from typing import Dict, Any, Optional
from core.common.types import MarketRegime
from core.regime_detector import RegimeDetector, RegimeInfo

logger = logging.getLogger("trading_bot.gating")

class RegimeGater:
    """
    Institutional Gating Layer.
    Controls strategy activation and risk parameters based on the current Market Regime.
    """

    @staticmethod
    def is_strategy_allowed(strategy_type: str, regime: MarketRegime) -> bool:
        """
        Implements the activation logic:
        - TREND: Enables Trend Following + Breakout
        - RANGE: Enables Mean Reversion + Liquidity/Session
        """
        st_type = strategy_type.upper().replace("_", "")
        
        if regime == MarketRegime.TREND:
            return "TREND" in st_type or "BREAKOUT" in st_type
        
        if regime == MarketRegime.RANGE:
            return "MEANREVERSION" in st_type or "LIQUIDITY" in st_type or "SNIPER" in st_type
            
        return True # Default to enabled if UNCERTAIN or other

    @staticmethod
    def get_risk_multiplier(regime: MarketRegime) -> float:
        """
        Implements risk reduction:
        - HIGH_VOL: Reduce risk (0.5x)
        """
        if regime == MarketRegime.HIGH_VOLATILITY:
            return 0.5
        return 1.0

    @staticmethod
    def get_confidence_buffer(regime: MarketRegime) -> float:
        """
        Implements frequency reduction:
        - LOW_VOL: Increase required confidence (+0.15)
        """
        if regime == MarketRegime.LOW_VOLATILITY:
            return 0.15
        return 0.0
