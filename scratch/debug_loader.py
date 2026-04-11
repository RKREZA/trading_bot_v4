from core.config.loader import ConfigLoader
import json

def debug_config():
    loader = ConfigLoader()
    symbol = "XAUUSDm"
    merged = loader.get_symbol_config(symbol)
    
    print(f"--- DEBUG MERGED CONFIG FOR {symbol} ---")
    print(json.dumps(merged, indent=2))
    
    # Check specific strategy keys
    strategies = ["TrendFollowing", "LiquiditySweepBreakout", "SmartMeanReversion"]
    for s in strategies:
        print(f"\n{s} config exists: {s in merged}")
        if s in merged:
            print(f"{s} enabled: {merged[s].get('enabled')}")

if __name__ == "__main__":
    debug_config()
