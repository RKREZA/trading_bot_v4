
import time
import numpy as np
import os
import sys

# Add project root
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from core.strategy_engine import StrategyEngine

def benchmark():
    config = {"strategy_defaults": {}, "session_config": {}}
    engine = StrategyEngine(config, silent=True)
    
    # Generate 400 sample prices
    prices = np.random.normal(2000, 10, 400)
    
    # Warm up
    engine._calculate_rsi(prices, 14)
    engine._calculate_ema_series(prices, 20)
    
    n_iters = 1000
    
    start_rsi = time.perf_counter()
    for _ in range(n_iters):
        engine._calculate_rsi(prices, 14)
    end_rsi = time.perf_counter()
    
    start_ema = time.perf_counter()
    for _ in range(n_iters):
        engine._calculate_ema_series(prices, 20)
    end_ema = time.perf_counter()
    
    print(f"RSI NumPy (1000 iters of 400 prices): {(end_rsi - start_rsi)*1000:.2f}ms")
    print(f"EMA NumPy (1000 iters of 400 prices): {(end_ema - start_ema)*1000:.2f}ms")

if __name__ == "__main__":
    benchmark()
