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

    # Institutional Regime Contract Mapping (Priority 2)
    # Strategies MUST exist in these buckets to execute under the specific regime flag
    REGIME_CONTRACT: Dict[MarketRegime, list] = {
        MarketRegime.TREND: ["TrendFollowing", "LiquiditySweepBreakout", "Diagnostic", "LiquiditySession", "PureBreakoutOneMinute"],
        MarketRegime.RANGE: ["SmartMeanReversion", "RangeBounce", "Diagnostic", "LiquiditySession", "PureBreakoutOneMinute"],
        MarketRegime.UNCERTAIN: ["Diagnostic", "PureBreakoutOneMinute", "LiquiditySession", "TrendFollowing", "LiquiditySweepBreakout", "SmartMeanReversion", "RangeBounce"] # Allow all in uncertain markets
    }

    @classmethod
    def is_strategy_allowed(cls, strategy_name: str, market_type: MarketRegime) -> bool:
        """
        Institutional Regime Deterministic Routing.
        Strictly enforces centroid assignments against active model names protecting against drift.
        """
        allowed_list = cls.REGIME_CONTRACT.get(market_type, [])

        # Robust case-insensitive search
        for allowed in allowed_list:
            if allowed.lower() in strategy_name.lower():
                return True
                
        # If market type isn't defined or strategy misses contract -> Hard block
        return False

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
        - LOW: Increase required confidence (+0.05)
        """
        if volatility == VolatilityStatus.LOW:
            return 0.05
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
