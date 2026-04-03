import json
import os

try:
    with open("config.json", "r") as f:
        config = json.load(f)
    print("SUCCESS: config.json is valid JSON.")
    print(f"fixed_rr: {config['strategy_defaults']['price_action']['fixed_rr']}")
except Exception as e:
    print(f"FAILED: config.json parsing error: {e}")
