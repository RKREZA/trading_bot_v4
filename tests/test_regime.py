import pytest
from core.regime import MarketRegime

def test_regime_classification_trending():
    # Create an artificially perfectly trending market (straight up)
    candles = []
    for i in range(100):
        # Highs and lows follow a tight spread around the perfect close
        candles.append({
            'open': 100 + i,
            'high': 100 + i + 0.5,
            'low': 100 + i - 0.5,
            'close': 101 + i,
            'tick_volume': 100
        })
    # This should yield exceptionally high Efficiency Ratio -> Trending
    regime = MarketRegime.classify(candles, lookback=50)
    assert regime == MarketRegime.TRENDING

def test_regime_classification_low_liquidity():
    # Normal volume then abrupt drop
    candles = []
    for i in range(95):
        candles.append({
            'open': 100, 'high': 101, 'low': 99, 'close': 100.5, 'tick_volume': 1000
        })
    for i in range(5):
        candles.append({
            'open': 100, 'high': 100.1, 'low': 99.9, 'close': 100, 'tick_volume': 10 # 1% of avg
        })
    
    regime = MarketRegime.classify(candles, lookback=50)
    assert regime == MarketRegime.LOW_LIQUIDITY

def test_regime_classification_ranging():
    # Zig zag market with efficiency ratio near 0
    candles = []
    for i in range(100):
        close = 101 if i % 2 == 0 else 99
        candles.append({
            'open': 100, 'high': 102, 'low': 98, 'close': close, 'tick_volume': 100
        })
    
    regime = MarketRegime.classify(candles, lookback=50)
    # The ATR might be moderate, but ER is practically 0
    assert regime == MarketRegime.RANGING
