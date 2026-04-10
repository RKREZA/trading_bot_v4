import sys
sys.path.insert(0, '.')
import logging

# Set up detailed logging
logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')

from core.data.parquet_store import ParquetStore
from strategies import create_strategy
from core.base_strategy import MarketData
from core.session_detector import SessionDetector
from core.regime_gater import RegimeGater
from datetime import datetime, timezone
import json
import numpy as np

with open('config.json') as f:
    config = json.load(f)

config['backtest']['debug_signals'] = True

store = ParquetStore(base_path='data_cache')
m5 = store.load('XAUUSDm', 'M5')
h1 = store.load('XAUUSDm', 'H1')

# Create strategy with the correct ID
strat = create_strategy('LiquiditySweepBreakout', 'LIQUIDITYSWEEPBREAKOUT', config)
print(f"Strategy created: {strat.strategy_id}, enabled={strat.enabled}")

# Simulate the backtester loop manually
lookback = strat.lookback
start_idx = 87000  # Around Feb 2026

print(f"\nTesting from idx {start_idx}")

for i in range(start_idx, start_idx + 500):
    dt = datetime.fromtimestamp(m5.time[i], tz=timezone.utc)
    
    # Simulate set_limit
    m5_limited = m5[:i]
    h1_idx = np.searchsorted(h1.time, m5.time[i], side='right')
    h1_idx = max(1, min(h1_idx, len(h1)))
    h1_limited = h1[:h1_idx]
    
    # Check regime
    regime_check = RegimeGater.is_strategy_allowed(strat.__class__.__name__, None)
    print(f"[{i}] Regime check for None regime: {regime_check}")
    
    # Build market data
    market_data = MarketData(
        symbol='XAUUSDm',
        htf_candles=h1_limited,
        m15_candles=m5_limited,
        m5_candles=m5_limited,
        d1_candles=None,
        current_price=m5.close[i-1],
        bid=m5.close[i-1],
        ask=m5.close[i-1] + 0.5,
        spread=0.5,
        session='LONDON',
        timestamp=dt
    )
    
    # Generate signal
    signal = strat.generate_signal(market_data)
    
    if signal:
        print(f"  [{i}] SIGNAL: {signal.direction} @ {signal.price:.2f}")
    else:
        reason = getattr(strat, 'last_rejection_reason', 'Unknown')
        if 'H1' in reason or 'Breakout' in reason:
            print(f"  [{i}] REJECTED: {reason} (H1 bars: {len(h1_limited)})")

print("\nDone")
