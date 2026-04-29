import sys
import os
sys.path.append(os.getcwd())
from strategies import STRATEGY_REGISTRY
print(f"Registry: {list(STRATEGY_REGISTRY.keys())}")
