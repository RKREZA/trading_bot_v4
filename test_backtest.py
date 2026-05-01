import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from core.backtest.engine import BacktestEngine
from core.strategy.smc_strategy import SMCStrategy
from core.time.time_service import time_service

def generate_dummy_data(bars=200):
    """Generates a dummy OHLC dataframe for testing."""
    dates = [datetime(2024, 1, 1, tzinfo=timezone.utc) + timedelta(minutes=5*i) for i in range(bars)]
    
    # Generate some price action
    close = 2000.0 + np.cumsum(np.random.randn(bars) * 2)
    high = close + np.random.rand(bars) * 2
    low = close - np.random.rand(bars) * 2
    open_price = np.roll(close, 1)
    open_price[0] = close[0]

    df = pd.DataFrame({
        'open': open_price,
        'high': high,
        'low': low,
        'close': close
    }, index=dates)
    return df

def test_smc_backtest():
    print("Initializing Time Service...")
    time_service.set_broker_offset(0)
    
    print("Generating dummy M5 data...")
    df = generate_dummy_data(300)
    
    print("Initializing SMC Strategy...")
    strategy = SMCStrategy(symbol="XAUUSDm", lookback=50, risk_reward=3.0)
    
    print("Initializing Backtest Engine (Spread: 1.5 pips, Slippage: 0.5 pips)...")
    engine = BacktestEngine(strategy=strategy, data=df, spread_pips=1.5, slippage_pips=0.5)
    
    print("Running Backtest...")
    results = engine.run(pip_size=0.1) # XAU pip size typically 0.1 or 0.01 depending on broker
    
    print("\n--- Backtest Results ---")
    for k, v in results.items():
        print(f"{k}: {v}")
        
    print("\nTest completed successfully!")

if __name__ == "__main__":
    test_smc_backtest()
