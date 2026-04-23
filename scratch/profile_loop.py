import sys
import os
sys.path.append(os.getcwd())
import time
import numpy as np
from datetime import datetime, timezone
from core.common.types import CandleArray, MarketRegime
from core.regime_detector import RegimeDetector, RegimeState
from core.regime_store import MemoryRegimeStore
from core.indicator_engine import IndicatorEngine
from core.session_detector import SessionDetector, SessionType

def profile_loop():
    print("Generating synthetic data...")
    size = 10000
    t = np.arange(size) * 300
    o = np.random.randn(size) + 2000
    h = o + 2
    l = o - 2
    c = o + 1
    v = np.random.randint(100, 1000, size)
    s = np.random.randint(1, 5, size)
    
    candles = CandleArray(t, o, h, l, c, v, s)
    
    print("Pre-calculating indicators...")
    indicators = IndicatorEngine.precalculate_all("TEST", "M5", candles)
    candles._indicators = indicators
    
    detector = RegimeDetector()
    store = MemoryRegimeStore()
    
    print(f"Starting Profile Loop for {size} bars...")
    start_time = time.time()
    
    # Simulate shim
    class Shim: pass
    shim = Shim()
    shim.m5_candles = candles
    shim.htf_candles = candles
    
    for i in range(500, 1500): # Small window for profile
        t_val = candles.time[i] # Use raw time array
        dt = datetime.fromtimestamp(t_val, tz=timezone.utc)
        
        # 1. Session lookup
        s1 = time.time()
        session = SessionDetector.get_session(dt, 0)
        shim.session = session
        shim.timestamp = dt
        e1 = time.time()
        
        # 2. Limit setting
        candles.set_limit(i)
        
        # 3. Detection
        s2 = time.time()
        state = store.load("TEST")
        res, new_state, trace = detector.detect(shim, state, f"EXEC_{i}", "TEST")
        store.save("TEST", new_state)
        e2 = time.time()
        
        if i % 100 == 0:
            print(f"Bar {i}: Session={e1-s1:.6f}s, Detect={e2-s2:.6f}s")
            
    end_time = time.time()
    print(f"Total time for 1000 bars: {end_time - start_time:.4f}s")
    print(f"Projected time for 75,000 bars: {(end_time - start_time) * 75:.2f}s")

if __name__ == "__main__":
    profile_loop()
