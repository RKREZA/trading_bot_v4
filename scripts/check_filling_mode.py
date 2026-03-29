import MetaTrader5 as mt5
import os
import json
from dotenv import load_dotenv

def check_filling():
    load_dotenv()
    
    # Load config to get symbols
    try:
        with open("config.json", "r") as f:
            config = json.load(f)
    except Exception as e:
        print(f"Error loading config: {e}")
        return

    login = int(os.environ.get("MT5_LOGIN", 0))
    password = os.environ.get("MT5_PASSWORD", "")
    server = os.environ.get("MT5_SERVER", "")

    if not mt5.initialize(login=login, password=password, server=server):
        print(f"MT5 initialize failed: {mt5.last_error()}")
        return

    symbols = config.get("symbols", ["XAUUSDm", "BTCUSDm"])
    print("-" * 50)
    print(f"{'Symbol':<15} | {'Filling Mode Bitmask':<20} | {'Recommended Mode'}")
    print("-" * 50)

    for symbol in symbols:
        info = mt5.symbol_info(symbol)
        if info is None:
            print(f"{symbol:<15} | Not found")
            continue
        
        filling = info.filling_mode
        
        mode_str = "UNKNOWN"
        if filling & 1: # SYMBOL_FILLING_FOK
            mode_str = "ORDER_FILLING_FOK"
        elif filling & 2: # SYMBOL_FILLING_IOC
            mode_str = "ORDER_FILLING_IOC"
        else:
            mode_str = "ORDER_FILLING_RETURN"

        print(f"{symbol:<15} | {filling:<20} | {mode_str}")

    mt5.shutdown()

if __name__ == "__main__":
    check_filling()
