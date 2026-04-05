"""
TRADING BOT V4 — Strategy Registry
==================================
Exposes the institutional-grade micro-service strategies.
"""

from .trend_following import TrendFollowingStrategy
from .mean_reversion import MeanReversionStrategy
from .breakout import BreakoutStrategy
from .liquidity_session import LiquiditySessionStrategy

STRATEGY_REGISTRY = {
    "TREND_FOLLOWING": TrendFollowingStrategy,
    "MEAN_REVERSION": MeanReversionStrategy,
    "BREAKOUT": BreakoutStrategy,
    "LIQUIDITY_SESSION": LiquiditySessionStrategy
}

def create_strategy(strategy_id: str, config: dict):
    """
    Factory function to instantiate strategies by ID.
    
    Args:
        strategy_id (str): ID corresponding to the registry keys.
        config (dict): Strategy-specific configuration.
        
    Returns:
        BaseStrategy: Instantiated strategy object.
    """
    if strategy_id not in STRATEGY_REGISTRY:
        raise ValueError(f"Strategy ID '{strategy_id}' not found in registry.")
    
    return STRATEGY_REGISTRY[strategy_id](strategy_id, config)

__all__ = [
    "TrendFollowingStrategy",
    "MeanReversionStrategy",
    "BreakoutStrategy",
    "LiquiditySessionStrategy",
    "STRATEGY_REGISTRY",
    "create_strategy"
]
