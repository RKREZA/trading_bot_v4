import sys
import os
import MetaTrader5 as mt5
from datetime import datetime, timezone

if not mt5.initialize():
    print("Initialize failed")
    sys.exit()

symbol = "XAUUSDm"
timeframe = mt5.TIMEFRAME_M1
start_date = datetime(2026, 4, 18, tzinfo=timezone.utc)
end_date = datetime(2026, 4, 28, tzinfo=timezone.utc)

rates = mt5.copy_rates_range(symbol, timeframe, start_date, end_date)
if rates is None:
    print(f"No rates for {symbol} on M1")
else:
    print(f"Retrieved {len(rates)} M1 candles for {symbol}")
    if len(rates) > 0:
        print(f"First: {datetime.fromtimestamp(rates[0]['time'], tz=timezone.utc)}")
        print(f"Last: {datetime.fromtimestamp(rates[-1]['time'], tz=timezone.utc)}")

mt5.shutdown()
