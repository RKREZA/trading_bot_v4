import sys
sys.path.insert(0, '.')
from core.data.parquet_store import ParquetStore
from datetime import datetime
import numpy as np

store = ParquetStore(base_path='data_cache')
m5 = store.load('XAUUSDm', 'M5')

print(f'Original len: {len(m5)}')
print(f'Original limit: {m5.limit}')

# Simulate backtest loop
start_ts = datetime(2026, 1, 1).timestamp()
idx_start = int(np.searchsorted(m5.time, start_ts))

for i in range(idx_start, idx_start + 5):
    m5.set_limit(i)
    print(f'After set_limit({i}): len(m5)={len(m5)}, m5.limit={m5.limit}, m5[-1]={m5.close[-1]}, m5[-lookback-1:-1] has {len(m5[-21:-1])} elements')

# Reset and check
m5._limit = None
print(f'\nAfter reset: len(m5)={len(m5)}, m5.limit={m5.limit}')
