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
    # Strategies MUST match these class names exactly to execute under specific regimes
    REGIME_CONTRACT: Dict[MarketRegime, set] = {
        MarketRegime.TREND: {
            "TrendFollowing", "TrendFollowingStrategy", "Diagnostic", "LiquiditySession",
            "PureBreakoutOneMinute", "LiquiditySweepBreakout", "LiquiditySweepBreakoutStrategy"
        },
        MarketRegime.RANGE: {
            "SmartMeanReversion", "SmartMeanReversionStrategy", "RangeBounce",
            "MeanReversionStrategy", "Diagnostic", "LiquiditySession",
            "TrendFollowing", "TrendFollowingStrategy",
            "LiquiditySweepBreakout", "LiquiditySweepBreakoutStrategy"
        },
        MarketRegime.LIQUIDITY_EVENT: {
            "LiquiditySweepBreakout", "LiquiditySweepBreakoutStrategy",
            "LiquiditySweepStrategy", "TrendFollowing", "TrendFollowingStrategy", "Diagnostic"
        },
        MarketRegime.EXPANSION: {
            "PureBreakoutOneMinute", "TrendFollowing", "TrendFollowingStrategy",
            "LiquiditySweepBreakout", "LiquiditySweepBreakoutStrategy", "Diagnostic"
        },
        MarketRegime.TRANSITION: {
            "TrendFollowing", "TrendFollowingStrategy",
            "SmartMeanReversion", "SmartMeanReversionStrategy",
            "LiquiditySweepBreakout", "LiquiditySweepBreakoutStrategy", "Diagnostic"
        }
    }

    @classmethod
    def is_strategy_allowed(cls, strategy_name: str, regime_info: RegimeInfo) -> bool:
        """
        Institutional Regime Deterministic Routing.
        Strictly enforces centroid assignments against active model names protecting against drift.
        """
        # Hard suppression: Do not trade if overall regime confidence is weak
        # Includes volatility-aware confidence buffer (e.g. higher req for LOW vol)
        required_confidence = 0.45 + cls.get_confidence_buffer(regime_info.volatility)
        if regime_info.confidence < required_confidence:
            return False
            
        allowed_set = cls.REGIME_CONTRACT.get(regime_info.market_type, set())
        
        # Strict identity matching against registry
        if strategy_name not in allowed_set:
            return False
            
        # Archetypal Volatility Suppression
        # Hard block MeanReversion during HIGH volatility regardless of ADX
        is_mean_rev = "MeanReversion" in strategy_name or "SmartMeanReversion" in strategy_name
        if is_mean_rev and regime_info.volatility == VolatilityStatus.HIGH:
            return False
            
        # Note: LOW volatility hard block removed. SMC strategies can trade
        # structural entries during consolidation — this is by design.
            
        # TRANSITION regime: allow all strategies included in the contract (already gated above)
            
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
