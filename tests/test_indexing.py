import sys
sys.path.insert(0, '.')
from core.data.parquet_store import ParquetStore
from core.common.types import Candle
from datetime import datetime
import numpy as np

store = ParquetStore(base_path='data_cache')
m5 = store.load('XAUUSDm', 'M5')

start_ts = datetime(2026, 1, 1).timestamp()
idx_start = int(np.searchsorted(m5.time, start_ts))

print(f'Original data at idx_start:')
print(f'  close[81182]: {m5.close[81182]}')
print(f'  close[81183]: {m5.close[81183]}')

m5.set_limit(81184)
print(f'\nAfter set_limit(81184):')
print(f'  len: {len(m5)}')
print(f'  limit: {m5.limit}')

# Test indexing
last = m5[-1]
print(f'  m5[-1].close: {last.close}')
print(f'  m5[-1].time: {datetime.fromtimestamp(last.time)}')

# The issue: m5[-1] should be bar 81183, not bar 81184
print(f'\nExpected: bar 81183 (close={m5.close[81183]:.3f})')
print(f'Got: bar with close={last.close:.3f}')

# Check what bar m5[-1] corresponds to
for i in range(81180, 81190):
    if abs(m5.close[i] - last.close) < 0.001:
        print(f'Found match at index {i}: close={m5.close[i]:.3f}, time={datetime.fromtimestamp(m5.time[i])}')
