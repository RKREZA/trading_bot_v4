import sys
sys.path.insert(0, '.')
import json
import numpy as np
from datetime import datetime, timezone
from core.data.parquet_store import ParquetStore
from core.base_strategy import MarketData
from core.session_detector import SessionDetector
from core.regime_detector import RegimeDetector
from core.regime_gater import RegimeGater
from strategies import create_strategy

with open('config.json') as f:
    config = json.load(f)

store = ParquetStore(base_path='data_cache')
m5 = store.load('XAUUSDm', 'M5')
h1 = store.load('XAUUSDm', 'H1')

# Get data for Jan 2026
start_ts = datetime(2026, 1, 1).timestamp()
end_ts = datetime(2026, 1, 3).timestamp()
idx_start = int(np.searchsorted(m5.time, start_ts))
idx_end = int(np.searchsorted(m5.time, end_ts))

strat = create_strategy('LSB1', 'LIQUIDITYSWEEPBREAKOUT', config)

regime_detector = RegimeDetector(config)
signals_found = 0
rejected_reasons = {}

print(f"Testing backtest loop from bar {max(100, idx_start)} to {idx_end}")

for i in range(max(100, idx_start), idx_end):
    t = m5.time[i]
    dt = datetime.fromtimestamp(t, tz=timezone.utc)
    
    # Regime detection
    regime_info = regime_detector.detect(m5[:i+1])
    regime = regime_info.market_type
    conf_buffer = RegimeGater.get_confidence_buffer(regime_info.volatility)
    
    # Get H1 data
    h1_idx = int(np.searchsorted(h1.time, t))
    
    # Build market data
    market_data = MarketData(
        symbol='XAUUSDm',
        htf_candles=h1[:h1_idx+1],
        m15_candles=h1[:h1_idx+1],
        m5_candles=m5[:i+1],
        d1_candles=None,
        current_price=float(m5.close[i]),
        bid=float(m5.close[i]),
        ask=float(m5.close[i]) + 0.5,
        spread=0.5,
        session=SessionDetector.get_session(dt, 3),
        timestamp=dt
    )
    
    # Generate signal
    signal = strat.generate_signal(market_data)
    
    if signal and signal.direction != 'NONE':
        min_conf = getattr(strat, "min_confidence", 0.6)
        if signal.confidence >= (min_conf + conf_buffer):
            signals_found += 1
            if signals_found <= 3:
                print(f"BAR {i}: {dt} - SIGNAL {signal.direction} @ {signal.confidence:.3f}")
        else:
            reason = f"Conf {signal.confidence:.3f} < {min_conf + conf_buffer:.3f}"
            rejected_reasons[reason] = rejected_reasons.get(reason, 0) + 1
    else:
        reason = strat.last_rejection_reason or "Unknown"
        rejected_reasons[reason] = rejected_reasons.get(reason, 0) + 1

print(f"\nTotal signals found: {signals_found}")
print("\nTop rejection reasons:")
for reason, count in sorted(rejected_reasons.items(), key=lambda x: -x[1])[:10]:
    print(f"  {reason}: {count}")
