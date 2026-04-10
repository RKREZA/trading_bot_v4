import sys
sys.path.insert(0, '.')
from strategies import create_strategy
from core.data.parquet_store import ParquetStore
from core.base_strategy import MarketData
from core.session_detector import SessionDetector
from datetime import datetime, timezone
import numpy as np
import json

with open('config.json') as f:
    config = json.load(f)

store = ParquetStore(base_path='data_cache')
m5 = store.load('XAUUSDm', 'M5')
h1 = store.load('XAUUSDm', 'H1')

strat = create_strategy('LSB1', 'LIQUIDITYSWEEPBREAKOUT', config)

# Test at multiple times
test_times = [
    datetime(2026, 1, 2, 5, 5),
    datetime(2026, 1, 2, 5, 10),
    datetime(2026, 1, 2, 5, 15),
    datetime(2026, 1, 2, 5, 20),
]

for test_dt in test_times:
    test_ts = test_dt.timestamp()
    m5_idx = int(np.searchsorted(m5.time, test_ts))
    h1_idx = int(np.searchsorted(h1.time, test_ts))
    
    market_data = MarketData(
        symbol='XAUUSDm',
        htf_candles=h1[:h1_idx+1],
        m15_candles=h1[:h1_idx+1],
        m5_candles=m5[:m5_idx+1],
        d1_candles=None,
        current_price=float(m5.close[m5_idx]),
        bid=float(m5.close[m5_idx]),
        ask=float(m5.close[m5_idx]) + 0.5,
        spread=0.5,
        session='TOKYO',
        timestamp=test_dt.replace(tzinfo=timezone.utc)
    )
    
    signal = strat.generate_signal(market_data)
    min_conf = getattr(strat, "min_confidence", 0.6)
    if signal:
        print(f'{test_dt}: Signal={signal.direction}, Conf={signal.confidence:.3f}, Min={min_conf}, Pass={signal.confidence >= min_conf}')
    else:
        print(f'{test_dt}: No signal - {strat.last_rejection_reason}')
