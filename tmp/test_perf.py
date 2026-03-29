
import time
import numpy as np
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
    
    # Mock candles
    n = 5000
    m5_candles = [{"time": i, "open": 100, "high": 101, "low": 99, "close": 100} for i in range(n)]
    m30_candles = [{"time": i*6, "open": 100, "high": 101, "low": 99, "close": 100} for i in range(n//6)]
    h1_candles = [{"time": i*12, "open": 100, "high": 101, "low": 99, "close": 100} for i in range(n//12)]
    h4_candles = [{"time": i*48, "open": 100, "high": 101, "low": 99, "close": 100} for i in range(n//48)]
    
    print(f"Testing StrategyEngine.analyze with {n} candles...")
    start_time = time.time()
    
    for i in range(200, n):
        if i % 1000 == 0:
            print(f"Processed {i}/{n} candles...")
        m5_slice = m5_candles[:i+1]
        # We use the same slices for simplicity in this test
        strategy.analyze("TEST", h4_candles[:100], h1_candles[:100], m30_candles[:100], m5_slice, 100.0)
    
    end_time = time.time()
    print(f"Total time for {n} candles: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    test_performance()
