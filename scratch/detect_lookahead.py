import numpy as np
import pandas as pd
from core.common.types import CandleArray
from core.indicator_engine import IndicatorEngine

def test_lookahead():
    # 1. Create a dummy dataset
    size = 1000
    times = np.arange(size)
    # Price is a simple trend
    close = np.linspace(10, 20, size) + np.random.normal(0, 0.1, size)
    
    ca = CandleArray(
        time=times,
        open=close-0.1,
        high=close+0.1,
        low=close-0.2,
        close=close,
        tick_volume=np.ones(size),
        spread=np.ones(size)
    )
    
    # 2. Precalculate FULL indicators
    ca.indicators = IndicatorEngine.precalculate_all("TEST", "M5", ca)
    
    # 3. Test values at various limits
    test_idx = 500
    
    # View A: Limit set to 500
    ca.set_limit(test_idx)
    ema_limited = ca.ema(20)
    val_at_limit_view = ema_limited[-1]
    
    # View B: Full array access (Potential Leak)
    # If the user is right, someone might be doing:
    val_full_leak = ca.indicators['ema_20'][test_idx - 1]
    
    print(f"Index: {test_idx - 1}")
    print(f"Value from get_indicator (Limited): {val_at_limit_view}")
    print(f"Value from direct indicators dict: {val_full_leak}")
    
    if np.isclose(val_at_limit_view, val_full_leak):
        print("PASS: Accessing via get_indicator(limit) matches direct index.")
    
    # 4. Detect TRUE lookahead (Does precalculation change based on future data?)
    # Calculate indicators for a SMALL subset
    ca_small = ca.slice(0, 500)
    ema_small = IndicatorEngine.precalculate_all("TEST", "M5", ca_small)['ema_20']
    
    print(f"Value calculated on 0:500: {ema_small[-1]}")
    print(f"Value calculated on 0:1000 at index 499: {val_full_leak}")
    
    if not np.isclose(ema_small[-1], val_full_leak):
        print("FAIL: Indicators change when more data is added (LOOKAHEAD DETECTED!)")
    else:
        print("PASS: Causal indicators detected.")

if __name__ == "__main__":
    test_lookahead()
