import sys
sys.path.insert(0, '.')
from core.data.parquet_store import ParquetStore
from datetime import datetime, timezone
import numpy as np

store = ParquetStore(base_path='data_cache')
m5 = store.load('XAUUSDm', 'M5')
h1 = store.load('XAUUSDm', 'H1')

lookback = 20
body_thresh = 0.55
h1_strength_thresh = 0.45

start_ts = datetime(2026, 2, 1).timestamp()
idx_start = int(np.searchsorted(m5.time, start_ts))

reasons = {}
signals = 0

for i in range(idx_start + lookback + 1, min(idx_start + 2000, len(m5) - 1)):
    price = m5.close[i-1]
    prev_high = np.max(m5.high[i-lookback-1:i-1])
    prev_low = np.min(m5.low[i-lookback-1:i-1])
    
    if prev_low <= price <= prev_high:
        reasons['inside_range'] = reasons.get('inside_range', 0) + 1
        continue
    
    m5_range = m5.high[i-1] - m5.low[i-1]
    m5_strength = abs(m5.close[i-1] - m5.open[i-1]) / m5_range if m5_range > 0 else 0
    
    if m5_strength < body_thresh:
        reasons['m5_weak'] = reasons.get('m5_weak', 0) + 1
        continue
    
    h1_idx = np.searchsorted(h1.time, m5.time[i])
    if h1_idx >= len(h1): h1_idx = len(h1) - 1
    h_high = h1.high[h1_idx]
    h_low = h1.low[h1_idx]
    h_range = h_high - h_low
    h1_strength = abs(h1.close[h1_idx] - h1.open[h1_idx]) / h_range if h_range > 0 else 0
    
    if h1_strength < h1_strength_thresh:
        reasons['h1_weak'] = reasons.get('h1_weak', 0) + 1
        continue
    
    signals += 1
    if signals <= 3:
        dt = datetime.fromtimestamp(m5.time[i], tz=timezone.utc)
        direction = "BUY" if price > prev_high else "SELL"
        print(f"{dt}: {direction} @ {price:.2f}")

print(f"Signals: {signals}")
for r, c in sorted(reasons.items(), key=lambda x: -x[1])[:8]:
    print(f"{r}: {c}")
