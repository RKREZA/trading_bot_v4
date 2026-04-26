import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.getcwd())
from strategies.n_pattern_grid import NPatternGridStrategy
from core.common.types import CandleArray
from datetime import datetime

df = pd.read_csv(r"c:\xampp\htdocs\trading_bot_v3\scratch\XAUUSDm_M1_data.csv")
time_arr = pd.to_datetime(df['time']).values.astype('datetime64[s]').astype(int)
m1_data = CandleArray(
    df['open'].values, df['high'].values, df['low'].values, df['close'].values,
    df['tick_volume'].values, time_arr, df['spread'].values
)

strat = NPatternGridStrategy("NPatternGrid", {"NPatternGrid": {"enabled": True}})

# Let's inspect the first 1000 bars
for i in range(50, 1000):
    m1_data.set_limit(i)
    avg_body = np.mean(np.abs(m1_data.o[-50:-1] - m1_data.c[-50:-1]))
    
    # Just check for Big Candles first
    o, h, l, c = m1_data.o[-1], m1_data.h[-1], m1_data.l[-1], m1_data.c[-1]
    if strat._is_big_candle(o, h, l, c, avg_body):
        print(f"[{df['time'].iloc[i-1]}] BIG CANDLE: Body={abs(c-o):.2f}, Avg={avg_body:.2f}")
        
        # Now track if it EVER retraces 90% in the next 30 bars
        is_bullish = c > o
        c_range = h - l
        retrace_level = h - (c_range * 0.9) if is_bullish else l + (c_range * 0.9)
        
        for j in range(i, min(i + 30, len(df))):
            curr_l = df['low'].iloc[j]
            curr_h = df['high'].iloc[j]
            if is_bullish and curr_l <= retrace_level:
                print(f"  -> HIT RETRACE at bar {j} ({df['time'].iloc[j]})")
                break
            elif not is_bullish and curr_h >= retrace_level:
                print(f"  -> HIT RETRACE at bar {j} ({df['time'].iloc[j]})")
                break
