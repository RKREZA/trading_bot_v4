import sys
sys.path.insert(0, '.')
from core.data.parquet_store import ParquetStore
from core.session_detector import SessionDetector
from datetime import datetime, timezone
import numpy as np

store = ParquetStore(base_path='data_cache')
m5 = store.load('XAUUSDm', 'M5')
h1 = store.load('XAUUSDm', 'H1')

start_ts = datetime(2026, 1, 1).timestamp()
idx_start = int(np.searchsorted(m5.time, start_ts))

lookback = 20
body_thresh = 0.40
h1_strength_thresh = 0.35

print('Checking signal generation:')
signals = 0
for i in range(idx_start + lookback + 1, idx_start + 500):
    dt = datetime.fromtimestamp(m5.time[i], tz=timezone.utc)
    
    # Session check
    if not SessionDetector.is_session_active(dt, allowed_sessions=['TOKYO', 'LONDON', 'LONDON/NY', 'GLOBAL', 'NEW_YORK']):
        continue
    
    # Price and range
    price = m5.close[i-1]
    prev_range = m5[i-lookback-1:i-1]
    r_high = np.max(prev_range.high)
    r_low = np.min(prev_range.low)
    
    # M5 strength
    last = m5[i-2]
    m5_range = last.high - last.low
    m5_strength = abs(last.close - last.open) / m5_range if m5_range > 0 else 0
    
    if price <= r_high and price >= r_low:
        continue
    if m5_strength < body_thresh:
        continue
    
    # H1 check
    h1_idx = int(np.searchsorted(h1.time, m5.time[i]))
    if h1_idx >= len(h1):
        continue
    h1_c = h1[h1_idx]
    h1_range = h1_c.high - h1_c.low
    h1_strength = abs(h1_c.close - h1_c.open) / h1_range if h1_range > 0 else 0
    h1_vol = h1_c.tick_volume
    
    if h1_strength < h1_strength_thresh:
        continue
    
    # Volume check
    h1_v_arr = h1.v
    h1_vol_sma = np.mean(h1_v_arr[-21:-1]) if len(h1_v_arr) > 21 else np.mean(h1_v_arr)
    minutes_into_hour = dt.minute
    completion_pct = max(0.05, (minutes_into_hour + 1) / 60.0)
    dynamic_threshold = h1_vol_sma * completion_pct * 0.50
    vol_confirmed = h1_vol > dynamic_threshold or h1_vol > h1_vol_sma * 0.40
    
    if not vol_confirmed:
        continue
    
    # Valid signal!
    signals += 1
    direction = "BUY" if price > r_high else "SELL"
    if signals <= 5:
        print(f"  {dt}: {direction} @ {price:.2f}, M5_str={m5_strength:.2f}, H1_str={h1_strength:.2f}")

print(f"\nTotal signals in 500 bars: {signals}")
