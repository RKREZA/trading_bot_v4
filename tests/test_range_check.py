import sys
sys.path.insert(0, '.')
from core.data.parquet_store import ParquetStore
from datetime import datetime
import numpy as np

store = ParquetStore(base_path='data_cache')
m5 = store.load('XAUUSDm', 'M5')

# Simulate backtest loop
start_ts = datetime(2026, 1, 1).timestamp()
idx_start = int(np.searchsorted(m5.time, start_ts))

lookback = 20
for i in range(idx_start, idx_start + 10):
    m5.set_limit(i)
    
    prev_range = m5[-lookback-1:-1]
    r_high = np.max(prev_range.high)
    r_low = np.min(prev_range.low)
    price = m5.close[-1]
    
    status = "INSIDE" if r_low <= price <= r_high else "BREAKOUT"
    direction = "BUY" if price > r_high else "SELL"
    
    print(f'Bar {i}: Price={price:.2f}, Range=[{r_low:.2f}, {r_high:.2f}], {status} ({direction})')
