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
        Institutional Regime Routing.
        
        V4-ULTRA Policy: Strategies are self-gating via their own ADX, RSI, and
        volatility filters. The regime gater provides advisory routing only —
        it no longer hard-blocks strategies to avoid redundant double-gating.
        
        Each strategy's generate_signal() is the final authority.
        """
        # All strategies are allowed to evaluate — they self-reject if conditions
        # don't match their internal institutional filters.
        return True

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
