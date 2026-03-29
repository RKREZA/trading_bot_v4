
import time
import numpy as np
import os
import sys

# Add root to sys.path
sys.path.append(".")

from core.strategy_engine import StrategyEngine

def test_performance():
    config = {
        "strategy": {
            "min_confluence_score": 2,
            "min_confidence": 60,
            "pullback_distance_pct": 0.5,
            "atr_period": 14,
            "swing_lookback": 3,
            "ema_fast": 20,
            "ema_slow": 50,
            "min_candle_body_pct": 20,
            "sl_atr_buffer": 0.8,
            "volatility_filter": {"enabled": True, "atr_multiplier_high": 4.0, "atr_multiplier_low": 0.2, "lookback": 200}
        }
    }
    strategy = StrategyEngine(config, silent=True)
    
    # Mock candles (reflecting real sizes)
    n_m5 = 54000
    n_m30 = 18000
    n_h1 = 9000
    n_h4 = 2500
    
    m5_candles = [{"time": i, "open": 100, "high": 101, "low": 99, "close": 100} for i in range(n_m5)]
    m30_candles = [{"time": i*6, "open": 100, "high": 101, "low": 99, "close": 100} for i in range(n_m30)]
    h1_candles = [{"time": i*12, "open": 100, "high": 101, "low": 99, "close": 100} for i in range(n_h1)]
    h4_candles = [{"time": i*48, "open": 100, "high": 101, "low": 99, "close": 100} for i in range(n_h4)]
    
    print(f"Testing StrategyEngine.analyze with real-world scales...")
    start_time = time.time()
    
    # Just test 1000 iterations at the end (where it's slowest)
    test_range = range(50000, 51000)
    for i in test_range:
        m5_slice = m5_candles[:i+1]
        m30_slice = m30_candles[:i//6 + 1]
        h1_slice = h1_candles[:i//12 + 1]
        h4_slice = h4_candles[:i//48 + 1]
        
        strategy.analyze("TEST", h4_slice, h1_slice, m30_slice, m5_slice, 100.0)
    
    end_time = time.time()
    per_candle = (end_time - start_time) / len(test_range)
    print(f"Time per candle at the end: {per_candle:.4f} seconds")
    print(f"Estimated total time for 54,000 candles: {per_candle * 54000 / 2 / 60:.2f} minutes")

if __name__ == "__main__":
    test_performance()
