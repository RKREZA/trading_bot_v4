"""Trading Bot V3 — Strategies Package"""

from .sniper_strategy import SniperStrategy
from .smc_strategy import SMCStrategy

__all__ = ["SniperStrategy", "SMCStrategy"]

# Strategy Registry: Maps type string to class
STRATEGY_REGISTRY = {
    "SNIPER": SniperStrategy,
    "SMC": SMCStrategy,
}


def create_strategy(strategy_id: str, strategy_type: str, config: dict):
    """
    Factory function to create a strategy by type name.
    
    Args:
        strategy_id: Unique identifier for this strategy instance
        strategy_type: Type key (e.g. "SNIPER", "SMC")
        config: Strategy-specific configuration dict
        
    Returns:
        BaseStrategy instance
        
    Raises:
        ValueError: If strategy_type is not registered
    """
    cls = STRATEGY_REGISTRY.get(strategy_type.upper())
    if cls is None:
        raise ValueError(
            f"Unknown strategy type '{strategy_type}'. "
            f"Available: {list(STRATEGY_REGISTRY.keys())}"
        )
    return cls(strategy_id, config)
