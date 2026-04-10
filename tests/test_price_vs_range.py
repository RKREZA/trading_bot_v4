import sys
sys.path.insert(0, '.')
from core.data.parquet_store import ParquetStore
from strategies import create_strategy
from core.base_strategy import MarketData
from datetime import datetime, timezone
import numpy as np
import json

with open('config.json') as f:
    config = json.load(f)

store = ParquetStore(base_path='data_cache')
m5_full = store.load('XAUUSDm', 'M5')
h1_full = store.load('XAUUSDm', 'H1')

strat = create_strategy('LSB1', 'LIQUIDITYSWEEPBREAKOUT', config)

# Test at various indices
for idx in [87200, 87400, 87600, 87800, 88000, 88200, 88400, 88600, 88800, 89000]:
    dt = datetime.fromtimestamp(m5_full.time[idx], tz=timezone.utc)
    
    h1_idx = np.searchsorted(h1_full.time, m5_full.time[idx], side='right')
    h1_idx = max(1, min(h1_idx, len(h1_full)))
    h1_data = h1_full[:h1_idx]
    m5_data = m5_full[:idx]
    
    price = m5_full.close[idx-1]
    lookback = strat.lookback
    prev_high = np.max(m5_full.high[idx-lookback-1:idx-1])
    prev_low = np.min(m5_full.low[idx-lookback-1:idx-1])
    
    in_range = prev_low <= price <= prev_high
    date_str = dt.strftime('%m-%d %H:%M')
    print(f"{idx} ({date_str}): price={price:.1f}, range=[{prev_low:.1f},{prev_high:.1f}], in_range={in_range}")
