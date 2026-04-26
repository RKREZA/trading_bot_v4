import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.getcwd())
from strategies.n_pattern_grid import NPatternGridStrategy
from core.common.types import CandleArray

df = pd.read_csv(r"c:\xampp\htdocs\trading_bot_v3\scratch\XAUUSDm_M1_data.csv")
time_arr = pd.to_datetime(df['time']).values.astype('datetime64[s]').astype(int)
# CORRECT ORDER: time, open, high, low, close, tick_volume, spread
m1_data = CandleArray(
    time_arr, 
    df['open'].values, df['high'].values, df['low'].values, df['close'].values,
    df['tick_volume'].values, df['spread'].values
)

strat = NPatternGridStrategy("NPatternGrid", {"NPatternGrid": {"enabled": True}})

for i in range(200, 300):
    m1_data.set_limit(i)
    avg_body = np.mean(np.abs(m1_data.o[-50:-1] - m1_data.c[-50:-1]))
    o, h, l, c = m1_data.o[-1], m1_data.h[-1], m1_data.l[-1], m1_data.c[-1]
    body = abs(c-o)
    rng = h-l
    ratio = body/rng if rng > 0 else 0
    is_large = body > avg_body*1.5
    if is_large and ratio > 0.7:
        print(f"Bar {i}: BIG CANDLE! Body={body:.3f}, Avg={avg_body:.3f}, Ratio={ratio:.2f}")
