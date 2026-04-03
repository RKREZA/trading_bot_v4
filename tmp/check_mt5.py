import MetaTrader5 as mt5

if not mt5.initialize():
    print(f"FAILED: MT5 initialize() failed, error code={mt5.last_error()}")
    quit()

print("SUCCESS: MT5 connected.")
mt5.shutdown()
