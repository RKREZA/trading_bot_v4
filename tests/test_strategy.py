"""
Unit tests for StrategyEngine.
Tests can run without MT5 terminal — they use synthetic candle data.
Aligns with Trading Bot V3 Roadmap updates (Phase 2 & 3).
"""

import sys
import os
import pytest
import numpy as np
from datetime import datetime, timezone

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.strategy_engine import StrategyEngine, TradeSignal
from core.regime import MarketRegime
from core.types import CandleArray

# ---------------------------------------------------------------------------
# Helpers — generate synthetic candle data
# ---------------------------------------------------------------------------

def make_candles(n: int, base_price: float = 100.0, trend: str = "flat",
                  step: float = 0.5) -> CandleArray:
    """
    Generate synthetic candles and returns a CandleArray.
    trend: 'up', 'down', 'flat'
    """
    candles = []
    price = base_price
    for i in range(n):
        if trend == "up":
            price += step
        elif trend == "down":
            price -= step

        o = price
        c = price + (step * 0.3 if trend == "up" else -step * 0.3 if trend == "down" else 0.1)
        h = max(o, c) + abs(step * 0.4)
        l = min(o, c) - abs(step * 0.4)
        candles.append({
            "time": 1700000000 + i * 300,  # 5-min intervals
            "open": o,
            "high": h,
            "low": l,
            "close": c,
            "tick_volume": 1000 + i * 10,
        })
    return CandleArray.from_dicts(candles)


@pytest.fixture
def config():
    return {
        "strategy_defaults": {
            "min_confluence_score": 1, 
            "min_confidence": 30, 
            "cooldown_candles": 3,
            "atr_period": 14,
            "ema_fast": 20,
            "ema_slow": 50,
            "sl_atr_buffer": 0.8
        },
        "session_config": {
            "LONDON": {"enabled": True},
            "NEW_YORK": {"enabled": True},
            "TOKYO": {"enabled": True}
        },
        "ai_advisor": {"bias": 0.0}
    }


@pytest.fixture
def engine(config):
    return StrategyEngine(config, silent=True)


# ---------------------------------------------------------------------------
# Indicator Tests
# ---------------------------------------------------------------------------

class TestIndicators:
    def test_atr_calculation(self, engine):
        candles = make_candles(30, trend="up", step=1.0)
        atr = engine._calculate_atr(candles, period=14)
        assert isinstance(atr, float)
        assert atr > 0
        # In our make_candles with step 1.0, TR is roughly 1.0 + 0.8 = 1.8
        assert 1.0 < atr < 3.0


# ---------------------------------------------------------------------------
# Regime Tests
# ---------------------------------------------------------------------------

class TestMarketRegime:
    def test_trending_regime(self):
        candles = make_candles(100, trend="up", step=1.0)
        regime = MarketRegime.classify(candles)
        assert regime == MarketRegime.TRENDING

    def test_ranging_regime(self):
        candles = make_candles(100, trend="flat", step=0.01)
        regime = MarketRegime.classify(candles)
        assert regime == MarketRegime.RANGING


# ---------------------------------------------------------------------------
# Strategy Analysis Tests
# ---------------------------------------------------------------------------

class TestStrategyLogic:
    def test_analyze_uptrend_signal(self, engine):
        # Setup alignment: H4 Up, H1 Up, M30 Up, M5 breakout
        h4 = make_candles(100, trend="up", step=2.0)
        h1 = make_candles(100, trend="up", step=1.0)
        m15 = make_candles(100, trend="up", step=0.5)
        m5 = make_candles(100, trend="up", step=0.2)
        
        current_price = m5.close[-1] + 0.1 # Force a small breakout
        
        # provide required preprocessed dict
        pre = {
            "m_bias": "BULLISH", "m_high": m5.high[-2], "m_low": m5.low[-2],
            "in_demand": True, "d_depth": 10.0, "vol_sma": 1000
        }
        
        signal, trend, regime = engine.analyze(
            "TEST", h4, h1, m15, m5, current_price, session="LONDON", preprocessed=pre
        )
        
        assert trend == "BULLISH"
        # In the context of PA_SNIPER, regime is the type of logic triggered
        assert regime == "PA_SNIPER"

    def test_session_filter_skips_untradeable(self, engine):
        h4 = make_candles(50, trend="up")
        h1 = make_candles(50, trend="up")
        m15 = make_candles(50, trend="up")
        m5 = make_candles(50, trend="up")
        
        pre = {"m_bias": "NEUTRAL", "m_high": 200, "m_low": 50, "vol_sma": 1000}
        
        # Test a session not in our config
        signal, _, _ = engine.analyze("TEST", h4, h1, m15, m5, 100.0, session="SYDNEY", preprocessed=pre)
        assert signal is None
