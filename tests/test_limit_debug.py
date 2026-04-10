import sys
sys.path.insert(0, '.')
from core.data.parquet_store import ParquetStore
from datetime import datetime
import numpy as np

store = ParquetStore(base_path='data_cache')
m5 = store.load('XAUUSDm', 'M5')

# Check actual data
start_ts = datetime(2026, 1, 1).timestamp()
idx_start = int(np.searchsorted(m5.time, start_ts))

print(f'Original data (full):')
print(f'  len: {len(m5)}')
print(f'  time[81182]: {datetime.fromtimestamp(m5.time[81182])}')
print(f'  time[81187]: {datetime.fromtimestamp(m5.time[81187])}')
print(f'  close[81182]: {m5.close[81182]}')
print(f'  close[81187]: {m5.close[81187]}')

# Set limit
m5.set_limit(81187)
print(f'\nAfter set_limit(81187):')
print(f'  len: {len(m5)}')
print(f'  limit property: {m5.limit}')
print(f'  close[-1]: {m5.close[-1]}')
print(f'  close[-2]: {m5.close[-2]}')
print(f'  c property (limited): {m5.c[-1]}')

# Reset and check
m5._limit = None
print(f'\nAfter reset:')
print(f'  len: {len(m5)}')
print(f'  close[-1]: {m5.close[-1]}')
