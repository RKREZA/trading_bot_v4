import sys
sys.path.insert(0, '.')
from core.data.parquet_store import ParquetStore
from datetime import datetime
import numpy as np

store = ParquetStore(base_path='data_cache')
m5 = store.load('XAUUSDm', 'M5')

start_ts = datetime(2026, 1, 1).timestamp()
idx_start = int(np.searchsorted(m5.time, start_ts))

print(f'Loop index vs data:')
for i in [81182, 81183, 81184, 81185]:
    m5.set_limit(i)
    last = m5[-1]
    print(f'i={i}: set_limit({i}), m5[-1].close={last.close:.3f} (bar {i-1})')
    print(f'      m5.open[{i}]={m5.open[i]:.3f} (bar {i})')
    print()
