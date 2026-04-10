import sys
sys.path.insert(0, '.')

from core.data.parquet_store import ParquetStore
from strategies import create_strategy
from core.base_strategy import MarketData
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

lookback = strat.lookback

start_ts = datetime(2026, 2, 1).timestamp()
idx_start = int(np.searchsorted(m5.time, start_ts))

# Precompute H1 indices for M5 times
h1_times = h1.time
m5_times = m5.time

reasons = {}
signals = 0

for i in range(idx_start + lookback + 1, min(idx_start + 5000, len(m5) - 1)):
    dt = datetime.fromtimestamp(m5_times[i], tz=timezone.utc)
    
    # Get H1 index for current M5 time
    h1_idx = np.searchsorted(h1_times, m5_times[i], side='right')
    h1_idx = max(1, min(h1_idx, len(h1)))
    
    # Build limited data
    m5_limited = m5[:i]
    h1_limited = h1[:h1_idx]
    m15_limited = m5[:i]  # Use M5 as M15 proxy
    
    current_price = m5.close[i-1]
    
    market_data = MarketData(
        symbol='XAUUSDm',
        htf_candles=h1_limited,
        m15_candles=m15_limited,
        m5_candles=m5_limited,
        d1_candles=None,
        current_price=current_price,
        bid=current_price,
        ask=current_price + 0.5,
        spread=0.5,
        session=SessionDetector.get_session(dt, 3),
        timestamp=dt
    )
    
    signal = strat.generate_signal(market_data)
    
    if signal is None:
        reason = getattr(strat, 'last_rejection_reason', 'Unknown')
        reasons[reason] = reasons.get(reason, 0) + 1
    else:
        signals += 1
        if signals <= 5:
            print(f"  {dt}: SIGNAL {signal.direction} @ {signal.price:.2f}")

print(f"\nTotal signals: {signals}")
for r, c in sorted(reasons.items(), key=lambda x: -x[1])[:10]:
    print(f"  {r}: {c}")
