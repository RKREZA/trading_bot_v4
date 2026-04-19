from core.common.types import CandleArray, MarketRegime, VolatilityStatus
from core.regime_detector import RegimeDetector
import numpy as np

def test_regime():
    print("Initializing Regime Detector...")
    r = RegimeDetector()
    print("Init successful.")
    # More tests can be added here if needed

if __name__ == "__main__":
    test_regime()
