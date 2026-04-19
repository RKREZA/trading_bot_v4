import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime

# Add project root to sys.path
sys.path.append(os.getcwd())

from core.ai.features import FeatureEngineer
from core.ai.model import AIModeWrapper
from core.common.types import CandleArray

def test_ai_upgrade():
    print("--- Testing AI Edge Upgrade ---")
    
    # 1. Create dummy candles
    # Need at least 20+ to avoid FeatureEngineer short-circuit
    times = [datetime.now().timestamp() - i * 60 for i in range(50)][::-1]
    opens = [1.0 + np.random.normal(0, 0.001) for _ in range(50)]
    closes = [o + np.random.normal(0, 0.001) for o in opens]
    highs = [max(o, c) + abs(np.random.normal(0, 0.001)) for o, c in zip(opens, closes)]
    lows = [min(o, c) - abs(np.random.normal(0, 0.001)) for o, c in zip(opens, closes)]
    spreads = [0.0002 for _ in range(50)]
    
    tick_volumes = np.array([100] * 50)
    
    candles = CandleArray(
        time=np.array(times),
        open=np.array(opens),
        high=np.array(highs),
        low=np.array(lows),
        close=np.array(closes),
        tick_volume=tick_volumes,
        spread=np.array(spreads)
    )
    
    # Mock Indicators (IndicatorEngine would normally do this)
    # CandleArray needs cached dict entries
    candles._indicators["adx_14"] = np.array([25.0] * 50)
    candles._indicators["atr_14"] = np.array([0.001] * 50)
    
    # 2. Extract Features
    print("Extracting features with AR-3 lags...")
    features = FeatureEngineer.extract_features(candles, "BUY", 50.0)
    
    lag_keys = [k for k in features.keys() if "lag" in k]
    print(f"Detected {len(lag_keys)} lagged features: {lag_keys}")
    
    if len(lag_keys) < 9: # 3 lags * (body, trend, size)
        print("FAILED: Missing lagged features.")
        return
    else:
        print("SUCCESS: AR-3 lagged features extracted correctly.")

    # 3. Test Model Training (Dummy Data)
    print("\nTesting Model Training (HGB)...")
    wrapper = AIModeWrapper(model_dir="scratch/ai_test_weights")
    
    # Create dummy training data
    X = pd.DataFrame([features] * 10)
    y = pd.Series([1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
    
    try:
        wrapper.train(X, y)
        print("SUCCESS: Model trained and saved.")
    except Exception as e:
        print(f"FAILED: Training error: {e}")
        return

    # 4. Test Prediction
    print("\nTesting Prediction...")
    prob = wrapper.predict_probability(features)
    print(f"Prediction Probability: {prob:.4f}")
    
    if 0 <= prob <= 1:
        print("SUCCESS: Prediction successful.")
    else:
        print("FAILED: Invalid probability value.")

if __name__ == "__main__":
    test_ai_upgrade()
