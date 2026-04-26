import MetaTrader5 as mt5
import datetime

if not mt5.initialize():
    print("Failed to initialize MT5")
    quit()

symbol = "XAUUSDm"
mt5.symbol_select(symbol, True)

# Check total bars available
count_m5 = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M5, 0, 10000)
count_m1 = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M1, 0, 10000)

print(f"MT5 REPORT for {symbol}:")
print(f"M5 Bars Available: {len(count_m5) if count_m5 is not None else 0}")
print(f"M1 Bars Available: {len(count_m1) if count_m1 is not None else 0}")

if count_m5 is not None and len(count_m5) > 0:
    first_bar = datetime.datetime.fromtimestamp(count_m5[0][0])
    last_bar = datetime.datetime.fromtimestamp(count_m5[-1][0])
    print(f"M5 Range: {first_bar} to {last_bar}")

mt5.shutdown()
