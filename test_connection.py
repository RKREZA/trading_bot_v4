"""MT5 Connection Test Script"""
import MetaTrader5 as mt5
import time

MT5_LOGIN = 463171405
MT5_PASSWORD = "Mftx@1234"
MT5_SERVER = "Exness-MT5Trial17"

print("=" * 50)
print("MT5 CONNECTION TEST")
print("=" * 50)

print("\n[1] Checking MT5...")
try:
    version = mt5.version()
    if version:
        print(f"    + MT5 Version: {version[0]}")
    else:
        print("    X MT5 not found!")
        exit(1)
except:
    print("    X MT5 not installed!")
    exit(1)

print(f"\n[2] Connecting to {MT5_SERVER}...")
mt5.shutdown()
time.sleep(1)

for attempt in range(3):
    print(f"\nAttempt {attempt + 1}/3...")
    if mt5.initialize(login=MT5_LOGIN, password=MT5_PASSWORD, server=MT5_SERVER, timeout=30000):
        print("    + Connected!")
        info = mt5.account_info()
        if info:
            print(f"\n[3] Account:")
            print(f"    + Login: {info.login}")
            print(f"    + Balance: ${info.balance:,.2f}")
            print(f"    + Equity: ${info.equity:,.2f}")
            print(f"\n[4] Testing data fetch...")
            rates = mt5.copy_rates_from_pos("BTCUSDm", mt5.TIMEFRAME_M30, 0, 10)
            if rates is not None and len(rates) > 0:
                print(f"    + Fetched {len(rates)} candles")
            mt5.shutdown()
            print("\n" + "=" * 50)
            print("CONNECTION TEST PASSED!")
            print("=" * 50)
            print("\nNow run: python main.py --backtest --symbol BTCUSDm")
            exit(0)
        else:
            print(f"    X Account failed: {mt5.last_error()}")
    else:
        error = mt5.last_error()
        print(f"    X Failed: {error}")
        if error[0] == -10005:
            print("    -> MT5 terminal not running! Open MT5 first.")
    time.sleep(2)

print("\n" + "=" * 50)
print("CONNECTION TEST FAILED")
print("=" * 50)
print("\nTROUBLESHOOTING:")
print("1. Open MT5 terminal and login manually")
print("2. Enable Algo Trading in MT5 settings")
print("3. Check internet connection")
mt5.shutdown()
