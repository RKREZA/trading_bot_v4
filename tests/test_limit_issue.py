import sys
sys.path.insert(0, '.')
from core.data.parquet_store import ParquetStore
from datetime import datetime
import numpy as np

store = ParquetStore(base_path='data_cache')
m5_full = store.load('XAUUSDm', 'M5')

# Check actual data
start_ts = datetime(2026, 1, 1).timestamp()
idx_start = int(np.searchsorted(m5_full.time, start_ts))

print(f'idx_start: {idx_start}')
print(f'Full data at idx_start: time={datetime.fromtimestamp(m5_full.time[idx_start])}, close={m5_full.close[idx_start]}')
print(f'Full data at idx_start+5: time={datetime.fromtimestamp(m5_full.time[idx_start+5])}, close={m5_full.close[idx_start+5]}')

# Now with limit
m5_full.set_limit(idx_start + 5)
print(f'\nAfter set_limit({idx_start+5}):')
print(f'len: {len(m5_full)}')
print(f'close[-1]: {m5_full.close[-1]} (should be bar {idx_start+5})')

# The issue is: when we set limit to i, len(m5) = i, so m5[-1] = m5[i-1]
# But we want to simulate bar i, where the "current" bar is the one at index i
# In the backtest, the loop goes to bar i, and at that point, the current bar should be bar i

# Let me check what the backtester does
print(f'\nIn backtester, i is the loop index:')
print(f'At i={idx_start+5}, we want bar {idx_start+5} to be the "current" bar')
print(f'But with set_limit({idx_start+5}), the accessible bars are 0 to {idx_start+4}')
print(f'So m5[-1] would be bar {idx_start+4}, not {idx_start+5}')
