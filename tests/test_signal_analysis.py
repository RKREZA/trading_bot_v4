import sys
sys.path.insert(0, '.')

from core.data.parquet_store import ParquetStore
from strategies import create_strategy
from core.regime_gater import RegimeGater
from core.regime_detector import RegimeDetector
from core.session_detector import SessionDetector
from datetime import datetime, timezone
import json
import numpy as np

with open('config.json') as f:
    config = json.load(f)

store = ParquetStore(base_path='data_cache')
m5 = store.load('XAUUSDm', 'M5')
h1 = store.load('XAUUSDm', 'H1')
m15 = m5  # Use m5 as proxy for m15

strat = create_strategy('LSB1', 'LIQUIDITYSWEEPBREAKOUT', config)
regime_detector = RegimeDetector()

lookback = strat.lookback
body_thresh = strat.body_thresh
h1_strength_thresh = strat.h1_strength_thresh

start_ts = datetime(2026, 2, 1).timestamp()
idx_start = int(np.searchsorted(m5.time, start_ts))

reasons = {}
signals = 0
trades = 0

for i in range(idx_start + lookback + 1, min(idx_start + 5000, len(m5.time) - 1)):
    dt = datetime.fromtimestamp(m5.time[i], tz=timezone.utc)
    
    # Check regime
    m5_limited = m5[:i]
    regime_info = regime_detector.detect(m5_limited)
    if not RegimeGater.is_strategy_allowed('LiquiditySweepBreakoutStrategy', regime_info.market_type):
        reasons['Regime blocked'] = reasons.get('Regime blocked', 0) + 1
        continue
    
    # Session check
    allowed = config.get('LiquiditySweepBreakout', {}).get('allowed_sessions', [])
    if not SessionDetector.is_session_active(dt, allowed_sessions=allowed):
        reasons['Out of session'] = reasons.get('Out of session', 0) + 1
        continue
    
    # Price and range (using closed data)
    price = m5.close[i-1]
    prev_high = np.max(m5.high[i-lookback-1:i-1])
    prev_low = np.min(m5.low[i-lookback-1:i-1])
    
    if prev_low <= price <= prev_high:
        reasons['Price inside range'] = reasons.get('Price inside range', 0) + 1
        continue
    
    # M5 strength
    last_high = m5.high[i-1]
    last_low = m5.low[i-1]
    last_open = m5.open[i-1]
    last_close = m5.close[i-1]
    m5_range = last_high - last_low
    m5_strength = abs(last_close - last_open) / m5_range if m5_range > 0 else 0
    
    if m5_strength < body_thresh:
        reasons[f'M5 Strength < {body_thresh:.2f}'] = reasons.get(f'M5 Strength < {body_thresh:.2f}', 0) + 1
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
        reasons[f'H1 Strength < {h1_strength_thresh:.2f}'] = reasons.get(f'H1 Strength < {h1_strength_thresh:.2f}', 0) + 1
        continue
    
    # Volume check
    h1_v_arr = h1.tick_volume
    h1_vol_sma = np.mean(h1_v_arr[max(0, h1_idx-21):h1_idx])
    minutes_into_hour = dt.minute
    completion_pct = max(0.05, (minutes_into_hour + 1) / 60.0)
    dynamic_threshold = h1_vol_sma * completion_pct * 0.50
    vol_confirmed = h1_vol > dynamic_threshold or h1_vol > h1_vol_sma * 0.40
    
    if not vol_confirmed:
        reasons['Volume not confirmed'] = reasons.get('Volume not confirmed', 0) + 1
        continue
    
    # M15 trend check (simplified - just check if EMA is trending)
    m15_ema = m5.close[max(0, i-50):i]
    if len(m15_ema) > 20:
        ema_50 = np.mean(m15_ema[-50:]) if len(m15_ema) >= 50 else m15_ema[-1]
        m15_trend = 1 if price > ema_50 else -1
    else:
        m15_trend = 0
    
    # H1 direction
    h1_dir = 1 if h1_close > h1_open else -1
    
    # Check trend alignment
    if price > prev_high:
        if h1_dir != 1:
            reasons['H1/M5 Dir mismatch (BUY)'] = reasons.get('H1/M5 Dir mismatch (BUY)', 0) + 1
            continue
        if m15_trend == -1:
            reasons['M15 trend opposes BUY'] = reasons.get('M15 trend opposes BUY', 0) + 1
            continue
    
    if price < prev_low:
        if h1_dir != -1:
            reasons['H1/M5 Dir mismatch (SELL)'] = reasons.get('H1/M5 Dir mismatch (SELL)', 0) + 1
            continue
        if m15_trend == 1:
            reasons['M15 trend opposes SELL'] = reasons.get('M15 trend opposes SELL', 0) + 1
            continue
    
    # Valid signal!
    signals += 1
    if signals <= 5:
        direction = "BUY" if price > prev_high else "SELL"
        print(f"  {dt}: {direction} @ {price:.2f} (RSI area)")

print(f"\nTotal potential signals in sample: {signals}")
print("\nTop rejection reasons:")
for r, c in sorted(reasons.items(), key=lambda x: -x[1])[:15]:
    pct = c / 5000 * 100
    print(f"  {r}: {c} ({pct:.1f}%)")
