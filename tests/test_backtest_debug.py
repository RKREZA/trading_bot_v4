import sys
sys.path.insert(0, '.')
import json
import logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

from core.data.parquet_store import ParquetStore
from backtesting.backtester import PortfolioBacktester
from strategies import create_strategy
from core.common.types import CandleArray
from datetime import datetime, timezone
import numpy as np

with open('config.json') as f:
    config = json.load(f)

# Enable debug
config['backtest']['debug_signals'] = True
config['backtest']['adaptive_strategy'] = False

store = ParquetStore(base_path='data_cache')
m5 = store.load('XAUUSDm', 'M5')
h1 = store.load('XAUUSDm', 'H1')

# Generate synthetic M1
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

# Create strategy
strat = create_strategy('LiquiditySweepBreakout', 'LIQUIDITYSWEEPBREAKOUT', config)

# Limit to Feb 2026
start_ts = datetime(2026, 2, 1).timestamp()
end_ts = datetime(2026, 2, 5).timestamp()

m5_start = int(np.searchsorted(m5.time, start_ts))
m5_end = int(np.searchsorted(m5.time, end_ts))

# Get aligned H1 data
h1_start = int(np.searchsorted(h1.time, m5.time[m5_start]))
h1_end = int(np.searchsorted(h1.time, m5.time[m5_end]))

print(f"Config debug_signals: {config['backtest'].get('debug_signals')}")

# Run backtest
bt = PortfolioBacktester(config)
history, equity = bt.run('XAUUSDm', [strat], m5[m5_start:m5_end], h1[h1_start:h1_end], m5[m5_start:m5_end], m5[m5_start:m5_end], m1)

print(f"\n=== RESULT ===")
print(f"Trades: {len(history)}")
for t in history[:5]:
    print(f"  {t['direction']} @ {t['fill_price']:.2f} -> PnL: {t['pnl']:.2f}")
