
from core.config.loader import ConfigLoader
import json

loader = ConfigLoader(environment="backtest")
cfg = loader.get_symbol_config("XAUUSDm")

print("--- Root Keys ---")
print(list(cfg.keys()))

print("\n--- LiquidityPriceAction Block ---")
if "LiquidityPriceAction" in cfg:
    print(json.dumps(cfg["LiquidityPriceAction"], indent=2))
else:
    print("NOT FOUND")

print("\n--- Strategies Block ---")
if "strategies" in cfg:
    print(json.dumps(cfg["strategies"].get("LiquidityPriceAction", {}), indent=2))
