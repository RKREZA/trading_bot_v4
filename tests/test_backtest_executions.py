import sys
sys.path.insert(0, '.')
from core.data.parquet_store import ParquetStore
from strategies import create_strategy
from core.risk.risk_guardian import RiskGuardian
from core.execution.order_manager import OrderManager
from core.session_detector import SessionDetector
from datetime import datetime, timezone
import json
import numpy as np

with open('config.json') as f:
    config = json.load(f)

store = ParquetStore(base_path='data_cache')
m5 = store.load('XAUUSDm', 'M5')
h1 = store.load('XAUUSDm', 'H1')

strat = create_strategy('LSB1', 'LIQUIDITYSWEEPBREAKOUT', config)
risk_guardian = RiskGuardian(config)
order_manager = OrderManager(config)

symbol_cfg = config.get('symbols_config', {}).get('XAUUSDm', {})

lookback = strat.lookback
body_thresh = strat.body_thresh
h1_strength_thresh = strat.h1_strength_thresh
sl_atr = strat.sl_atr

start_ts = datetime(2026, 2, 1).timestamp()
idx_start = int(np.searchsorted(m5.time, start_ts))

balance = 1000.0  # initial balance per strategy

reasons = {}
signals = 0
executions = 0

for i in range(idx_start + lookback + 1, min(idx_start + 2000, len(m5) - 1)):
    dt = datetime.fromtimestamp(m5.time[i], tz=timezone.utc)
    
    # Session check
    allowed = config.get('LiquiditySweepBreakout', {}).get('allowed_sessions', [])
    if not SessionDetector.is_session_active(dt, allowed_sessions=allowed):
        reasons['out_of_session'] = reasons.get('out_of_session', 0) + 1
        continue
    
    price = m5.close[i-1]
    prev_high = np.max(m5.high[i-lookback-1:i-1])
    prev_low = np.min(m5.low[i-lookback-1:i-1])
    
    if prev_low <= price <= prev_high:
        reasons['inside_range'] = reasons.get('inside_range', 0) + 1
        continue
    
    # M5 strength
    last_high = m5.high[i-1]
    last_low = m5.low[i-1]
    last_open = m5.open[i-1]
    last_close = m5.close[i-1]
    m5_range = last_high - last_low
    m5_strength = abs(last_close - last_open) / m5_range if m5_range > 0 else 0
    
    if m5_strength < body_thresh:
        reasons['m5_weak'] = reasons.get('m5_weak', 0) + 1
        continue
    
    # H1 check
    h1_idx = int(np.searchsorted(h1.time, m5.time[i]))
    if h1_idx >= len(h1):
        h1_idx = len(h1) - 1
    h_high = h1.high[h1_idx]
    h_low = h1.low[h1_idx]
    h_open = h1.open[h1_idx]
    h_close = h1.close[h1_idx]
    h_vol = h1.tick_volume[h1_idx]
    h_range = h_high - h_low
    h1_strength = abs(h_close - h_open) / h_range if h_range > 0 else 0
    
    if h1_strength < h1_strength_thresh:
        reasons['h1_weak'] = reasons.get('h1_weak', 0) + 1
        continue
    
    # M15 trend check
    m15_trend = 0
    if len(m5) >= i + 50:
        ema_50 = np.mean(m5.close[i-50:i])
        m15_trend = 1 if price > ema_50 else -1
    
    # H1 direction
    h1_dir = 1 if h_close > h_open else -1
    
    # Check direction alignment
    if price > prev_high:
        if h1_dir != 1:
            reasons['h1_dir_mismatch_buy'] = reasons.get('h1_dir_mismatch_buy', 0) + 1
            continue
        if m15_trend == -1:
            reasons['m15_trend_opposes_buy'] = reasons.get('m15_trend_opposes_buy', 0) + 1
            continue
    elif price < prev_low:
        if h1_dir != -1:
            reasons['h1_dir_mismatch_sell'] = reasons.get('h1_dir_mismatch_sell', 0) + 1
            continue
        if m15_trend == 1:
            reasons['m15_trend_opposes_sell'] = reasons.get('m15_trend_opposes_sell', 0) + 1
            continue
    
    # Volume check
    h1_v_arr = h1.tick_volume
    h1_vol_sma = np.mean(h1_v_arr[max(0, h1_idx-21):h1_idx])
    minutes_into_hour = dt.minute
    completion_pct = max(0.05, (minutes_into_hour + 1) / 60.0)
    dynamic_threshold = h1_vol_sma * completion_pct * 0.50
    vol_confirmed = h_vol > dynamic_threshold or h_vol > h1_vol_sma * 0.40
    
    if not vol_confirmed:
        reasons['volume_not_confirmed'] = reasons.get('volume_not_confirmed', 0) + 1
        continue
    
    # Valid signal!
    signals += 1
    direction = "BUY" if price > prev_high else "SELL"
    
    # Calculate SL
    atr_vals = m5.atr(14) if hasattr(m5, 'atr') else np.array([5.0])
    atr = atr_vals[-1] if len(atr_vals) > 0 else 5.0
    sl_dist = atr * sl_atr
    
    if sl_dist <= 0:
        reasons['sl_dist_zero'] = reasons.get('sl_dist_zero', 0) + 1
        continue
    
    # Calculate lot size
    point = symbol_cfg.get('point', 0.01)
    tick_value = symbol_cfg.get('tick_value', 1.0)
    risk_pct = config.get('risk_governance', {}).get('risk_per_trade_pct', 1.0)
    risk_amount = balance * (risk_pct / 100.0)
    points_dist = sl_dist / point
    denominator = points_dist * tick_value
    lot_size = risk_amount / denominator if denominator > 0 else 0.0
    
    # Normalize lot
    min_lot = symbol_cfg.get('min_lot', 0.01)
    max_lot = symbol_cfg.get('max_lot', 50.0)
    lot_size = max(min_lot, min(max_lot, lot_size))
    
    if lot_size < 0.01:
        reasons['lot_too_small'] = reasons.get('lot_too_small', 0) + 1
        continue
    
    # Execute trade
    executions += 1
    if executions <= 3:
        print(f"{dt}: {direction} @ {price:.2f} | SL dist: {sl_dist:.2f} | Lot: {lot_size:.3f}")
    
    reasons['executed'] = reasons.get('executed', 0) + 1

print(f"\nSignals: {signals}, Executions: {executions}")
for r, c in sorted(reasons.items(), key=lambda x: -x[1])[:12]:
    print(f"  {r}: {c}")
