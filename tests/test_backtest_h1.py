import sys
sys.path.insert(0, '.')
import logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

from core.data.parquet_store import ParquetStore
from backtesting.backtester import PortfolioBacktester
from strategies import create_strategy
from datetime import datetime, timezone
import json
import numpy as np

with open('config.json') as f:
    config = json.load(f)

config['backtest']['debug_signals'] = True
config['backtest']['adaptive_strategy'] = False

store = ParquetStore(base_path='data_cache')
m5 = store.load('XAUUSDm', 'M5')
h1 = store.load('XAUUSDm', 'H1')

print(f'M5 bars: {len(m5)}, time: {datetime.fromtimestamp(m5.time[0], tz=timezone.utc)} to {datetime.fromtimestamp(m5.time[-1], tz=timezone.utc)}')
print(f'H1 bars: {len(h1)}, time: {datetime.fromtimestamp(h1.time[0], tz=timezone.utc)} to {datetime.fromtimestamp(h1.time[-1], tz=timezone.utc)}')

# Create strategy
strat = create_strategy('LSB1', 'LIQUIDITYSWEEPBREAKOUT', config)

# Generate synthetic M1
from core.common.types import CandleArray
m1_times = []
m1_opens = []
m1_highs = []
m1_lows = []
m1_closes = []
m1_volumes = []
m1_spreads = []

for i in range(len(m5.time)):
    base_time = int(m5.time[i])
    for j in range(5):
        minute_time = base_time + j * 60
        m1_times.append(minute_time)
        m1_opens.append(m5.open[i] if j == 0 else m5.close[i-1] if i > 0 else m5.open[i])
        m1_highs.append(m5.high[i])
        m1_lows.append(m5.low[i])
        m1_closes.append(m5.close[i])
        m1_volumes.append(int(m5.tick_volume[i] / 5) if m5.tick_volume[i] > 0 else 0)
        m1_spreads.append(m5.spread[i] / 5 if m5.spread[i] > 0 else 10)

m1 = CandleArray(
    time=np.array(m1_times, dtype=np.int64),
    open=np.array(m1_opens, dtype=np.float64),
    high=np.array(m1_highs, dtype=np.float64),
    low=np.array(m1_lows, dtype=np.float64),
    close=np.array(m1_closes, dtype=np.float64),
    tick_volume=np.array(m1_volumes, dtype=np.int64),
    spread=np.array(m1_spreads, dtype=np.float64)
)

# Limit to Feb 2026 - use proper time-based slicing
start_ts = datetime(2026, 2, 1).timestamp()
end_ts = datetime(2026, 3, 1).timestamp()

m5_start_idx = int(np.searchsorted(m5.time, start_ts))
m5_end_idx = int(np.searchsorted(m5.time, end_ts))

# Get H1 data that covers this period
h1_start_ts = m5.time[m5_start_idx]
h1_end_ts = m5.time[m5_end_idx]
h1_start_idx = int(np.searchsorted(h1.time, h1_start_ts))
h1_end_idx = int(np.searchsorted(h1.time, h1_end_ts))

print(f'Sliced M5: {m5_end_idx - m5_start_idx} bars from idx {m5_start_idx}')
print(f'Sliced H1: {h1_end_idx - h1_start_idx} bars from idx {h1_start_idx}')
print()

# Run backtester with proper slices
bt = PortfolioBacktester(config)
history, equity_history = bt.run('XAUUSDm', [strat], m5[m5_start_idx:m5_end_idx], h1[h1_start_idx:h1_end_idx], m5[m5_start_idx:m5_end_idx], m5[m5_start_idx:m5_end_idx], m1, resume=False)

print()
print(f'Trades: {len(history)}')
for trade in history[:5]:
    print(f'  {trade["direction"]} @ {trade["fill_price"]:.2f} -> PnL: {trade["pnl"]:.2f}')
