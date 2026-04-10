import sys
sys.path.insert(0, '.')
from core.data.parquet_store import ParquetStore
from datetime import datetime
import numpy as np

store = ParquetStore(base_path='data_cache')
m5 = store.load('XAUUSDm', 'M5')

# Check what properties return
start_ts = datetime(2026, 1, 1).timestamp()
idx_start = int(np.searchsorted(m5.time, start_ts))

m5.set_limit(81184)
print(f'After set_limit(81184):')
print(f'  len: {len(m5)}')
print(f'  m5.open[81184] (absolute): {m5.open[81184]}')  # This is the unlimited access
print(f'  m5.o (limited): {m5.o[-1]}')  # This is the limited access

# The issue: m5.open uses __getitem__ which accesses absolute index
# But m5.o uses the limited property

print(f'\nFor bar 81184, we want:')
print(f'  open: {m5.open[81184]}')  # Correct
print(f'  close: {m5.close[81184]}')  # This is the close of bar 81184
print(f'\nBut m5[-1] returns:')
print(f'  close: {m5[-1].close}')  # This is close of bar 81183 (last limited element)

print(f'\nThe backtester does: current_price = target_tf_data.open[i]')
print(f'Where i is the loop index. So at i=81184, current_price = open[81184]')
print(f'But the strategy sees m5[-1] = close[81183]')
print(f'This is comparing open[81184] with range based on close[81183]')
