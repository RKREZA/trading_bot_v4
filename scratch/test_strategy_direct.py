import pandas as pd
import numpy as np
import logging
import sys
import os

sys.path.append(os.getcwd())
from strategies.n_pattern_grid import NPatternGridStrategy
from core.common.types import CandleArray
from datetime import datetime

# Setup data
df = pd.read_csv(r"c:\xampp\htdocs\trading_bot_v3\scratch\XAUUSDm_M1_data.csv")
time_arr = pd.to_datetime(df['time']).values.astype('datetime64[s]').astype(int)
m1_data = CandleArray(
    time_arr,
    df['open'].values, df['high'].values, df['low'].values, df['close'].values,
    df['tick_volume'].values, df['spread'].values
)

strat = NPatternGridStrategy("NPatternGrid", {"NPatternGrid": {"enabled": True}})

class MockMarketData:
    def __init__(self, m1):
        self.m1_candles = m1
        self.current_price = m1.c[-1]
        self.symbol = "XAUUSDm"
        self.point = 0.01
        self.session = "LONDON"
        self.timestamp = datetime.fromtimestamp(m1.time[-1])

trades = 0
for i in range(200, len(df)):
    m1_data.set_limit(i)
    md = MockMarketData(m1_data)
    sig = strat.generate_signal(md)
    if sig:
        print(f"[{df['time'].iloc[i-1]}] SIGNAL: {sig.direction} at {sig.price}")
        trades += 1

print(f"Total Trades: {trades}")
