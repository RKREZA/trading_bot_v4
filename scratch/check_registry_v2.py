import sys
import os
import json
sys.path.append(os.getcwd())
from strategies import STRATEGY_REGISTRY
from core.config.loader import ConfigLoader

loader = ConfigLoader()
config = loader.global_config

for st_type, st_class in STRATEGY_REGISTRY.items():
    print(f"Checking {st_type}...")
    try:
        # Match backtest.py logic
        pascal_name = st_type.title().replace("_", "")
        obj = st_class(pascal_name, config=config)
        print(f"  ID: {obj.strategy_id}")
        print(f"  Enabled: {obj.enabled}")
        print(f"  Symbol XAUUSDm allowed: {obj.is_symbol_allowed('XAUUSDm')}")
    except Exception as e:
        print(f"  Error: {e}")
