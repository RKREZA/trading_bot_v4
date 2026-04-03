import numpy as np
from core.types import CandleArray

def test_boolean_masking():
    # Create dummy data
    times = np.array([1000, 2000, 3000, 4000, 5000])
    candles = CandleArray(
        time=times,
        open=np.random.rand(5),
        high=np.random.rand(5),
        low=np.random.rand(5),
        close=np.random.rand(5),
        tick_volume=np.random.randint(0, 100, 5)
    )
    
    # Mask: time > 2500
    mask = times > 2500
    sliced = candles[mask]
    
    print(f"Original length: {len(candles)}")
    print(f"Mask: {mask}")
    print(f"Sliced length: {len(sliced)}")
    print(f"Sliced times: {sliced.time}")
    
    if len(sliced) == 3 and np.array_equal(sliced.time, [3000, 4000, 5000]):
        print("SUCCESS: Boolean masking works.")
    else:
        print("FAILED: Boolean masking result incorrect.")

if __name__ == "__main__":
    test_boolean_masking()
