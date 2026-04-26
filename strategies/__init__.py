"""
TRADING BOT V5 — Dynamic Strategy Discovery
===========================================
Automatically discovers and registers all BASE_STRATEGY subclasses in the strategies/ directory.
"""

import os
import pkgutil
import importlib
import logging
from typing import Dict, Type
from core.base_strategy import BaseStrategy

logger = logging.getLogger("trading_bot.strategies")

STRATEGY_REGISTRY: Dict[str, Type[BaseStrategy]] = {}

def _discover_strategies():
    """Scan the current package for strategy implementations."""
    path = os.path.dirname(__file__)
    for loader, module_name, is_pkg in pkgutil.iter_modules([path]):
        if is_pkg or module_name == "__init__":
            continue
            
        try:
            module = importlib.import_module(f".{module_name}", package=__package__)
            for attribute_name in dir(module):
                attribute = getattr(module, attribute_name)
                if (isinstance(attribute, type) and 
                    issubclass(attribute, BaseStrategy) and 
                    attribute is not BaseStrategy):
                    
                    # Store by Class Name (Upper) as the Registry Type
                    strat_type = attribute_name.replace("Strategy", "").upper()
                    STRATEGY_REGISTRY[strat_type] = attribute
                    logger.debug(f"Registered dynamic strategy type: {strat_type}")
        except Exception as e:
            logger.error(f"Failed to load strategy module {module_name}: {e}")

# Initial Discovery Run
_discover_strategies()

def create_strategy(strategy_id: str, st_type: str = None, config: dict = None):
    """
    Enhanced Factory function with dynamic resolution.
    
    Args:
        strategy_id (str): The unique ID for this instance (e.g., 'sniper_v1').
        st_type (str, optional): The strategy class type (e.g., 'LIQUIDITY_SESSION').
        config (dict): The global configuration.
        
    Returns:
        BaseStrategy: Instantiated strategy object.
    """
    if config is None: config = {}
    
    # Auto-resolve type if not provided
    if not st_type:
        # Heuristic resolution based on common ID patterns
        if "pattern" in strategy_id.lower() or "grid" in strategy_id.lower(): st_type = "NPATTERNGRID"
        else:
            raise ValueError(f"Could not auto-resolve strategy type for ID '{strategy_id}'. Please provide it explicitly.")

    # Normalization
    st_type = st_type.upper().replace("_", "")
    
    if st_type not in STRATEGY_REGISTRY:
        raise ValueError(f"Strategy Type '{st_type}' not found. Available: {list(STRATEGY_REGISTRY.keys())}")
    
    return STRATEGY_REGISTRY[st_type](strategy_id, config)

__all__ = ["STRATEGY_REGISTRY", "create_strategy"]
