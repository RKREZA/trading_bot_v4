import sys
sys.path.insert(0, '.')
from core.data.parquet_store import ParquetStore
from core.session_detector import SessionDetector
from strategies import create_strategy
from datetime import datetime, timezone
import numpy as np
import json

with open('config.json') as f:
    config = json.load(f)

store = ParquetStore(base_path='data_cache')
m5 = store.load('XAUUSDm', 'M5')
h1 = store.load('XAUUSDm', 'H1')

strat = create_strategy('LSB1', 'LIQUIDITYSWEEPBREAKOUT', config)

start_ts = datetime(2026, 1, 1).timestamp()
idx_start = int(np.searchsorted(m5.time, start_ts))

lookback = 20
body_thresh = strat.body_thresh
h1_strength_thresh = strat.h1_strength_thresh

print(f'Config: body_thresh={body_thresh}, h1_strength_thresh={h1_strength_thresh}')

reasons = {}
signals = 0
for i in range(idx_start + lookback + 1, idx_start + 5000):
    dt = datetime.fromtimestamp(m5.time[i], tz=timezone.utc)
    
    # Session check
    allowed = config.get('LiquiditySweepBreakout', {}).get('allowed_sessions', [])
    if not SessionDetector.is_session_active(dt, allowed_sessions=allowed):
        reasons['Out of session'] = reasons.get('Out of session', 0) + 1
        continue
    
    # Price and range
    price = m5.close[i-1]
    prev_high = np.max(m5.high[i-lookback-1:i-1])
    prev_low = np.min(m5.low[i-lookback-1:i-1])
    
    if prev_low <= price <= prev_high:
        reasons['Price inside range'] = reasons.get('Price inside range', 0) + 1
        continue
    
    # M5 strength
    last_high = m5.high[i-2]
    last_low = m5.low[i-2]
    last_open = m5.open[i-2]
    last_close = m5.close[i-2]
    m5_range = last_high - last_low
    m5_strength = abs(last_close - last_open) / m5_range if m5_range > 0 else 0
    
    if m5_strength < body_thresh:
        reasons[f'M5 Strength too low ({m5_strength:.2f})'] = reasons.get(f'M5 Strength too low ({m5_strength:.2f})', 0) + 1
        continue
    
    # H1 check
    h1_idx = int(np.searchsorted(h1.time, m5.time[i]))
    if h1_idx >= len(h1):
        h1_idx = len(h1) - 1
    h1_high = h1.high[h1_idx]
    h1_low = h1.low[h1_idx]
    h1_open = h1.open[h1_idx]
    h1_close = h1.close[h1_idx]
    h1_vol = h1.tick_volume[h1_idx]
    h1_range = h1_high - h1_low
    h1_strength = abs(h1_close - h1_open) / h1_range if h1_range > 0 else 0
    
    if h1_strength < h1_strength_thresh:
        reasons[f'H1 Strength too low ({h1_strength:.2f})'] = reasons.get(f'H1 Strength too low ({h1_strength:.2f})', 0) + 1
        continue
    
    # Volume check
    h1_v_arr = h1.tick_volume
    h1_vol_sma = np.mean(h1_v_arr[-21:-1]) if len(h1_v_arr) > 21 else np.mean(h1_v_arr)
    minutes_into_hour = dt.minute
    completion_pct = max(0.05, (minutes_into_hour + 1) / 60.0)
    dynamic_threshold = h1_vol_sma * completion_pct * 0.50
    vol_confirmed = h1_vol > dynamic_threshold or h1_vol > h1_vol_sma * 0.40
    
    if not vol_confirmed:
        reasons['Volume not confirmed'] = reasons.get('Volume not confirmed', 0) + 1
        continue
    
    # Valid signal!
    signals += 1
    if signals <= 5:
        direction = "BUY" if price > prev_high else "SELL"
        print(f"  {dt}: {direction} @ {price:.2f}")

print(f"\nTotal signals in 5000 bars: {signals}")
print("\nTop rejection reasons:")
for r, c in sorted(reasons.items(), key=lambda x: -x[1])[:10]:
    print(f"  {r}: {c}")
