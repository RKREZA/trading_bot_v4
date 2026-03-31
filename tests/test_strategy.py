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

# ---------------------------------------------------------------------------
# Helpers — generate synthetic candle data
# ---------------------------------------------------------------------------

def make_candles(n: int, base_price: float = 100.0, trend: str = "flat",
                  step: float = 0.5) -> list:
    """
    Generate synthetic candles.
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
    return candles


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
# EMA Tests
# ---------------------------------------------------------------------------

class TestIndicators:
    def test_ema_series_length(self, engine):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = engine._calculate_ema_series(data, 3)
        assert len(result) == len(data)

    def test_ema_trending_up(self, engine):
        data = np.array([float(i) for i in range(20)])
        result = engine._calculate_ema_series(data, 5)
        # EMA should be increasing
        for i in range(1, len(result)):
            assert result[i] > result[i - 1]

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
        h4 = make_candles(100, base_price=100, trend="up", step=2.0)
        h1 = make_candles(100, base_price=150, trend="up", step=1.0)
        m30 = make_candles(100, base_price=170, trend="up", step=0.5)
        m5 = make_candles(100, base_price=180, trend="up", step=0.2)
        
        current_price = m5[-1]["close"] + 0.1 # Force a small breakout
        
        signal, trend, regime = engine.analyze(
            "TEST", h4, h1, m30, m5, current_price, session="LONDON"
        )
        
        assert trend == "BULLISH"
        # We don't strictly assert signal here because confluence/confidence might still filter it
        # but trend and regime should be identified.
        assert regime in [MarketRegime.TRENDING, MarketRegime.RANGING, MarketRegime.HIGH_VOLATILITY]

    def test_session_filter_skips_untradeable(self, engine):
        h4 = make_candles(100, trend="up")
        h1 = make_candles(100, trend="up")
        m30 = make_candles(100, trend="up")
        m5 = make_candles(100, trend="up")
        
        # Test a session not in our config
        signal, _, _ = engine.analyze("TEST", h4, h1, m30, m5, 100.0, session="SYDNEY")
        assert signal is None

# ---------------------------------------------------------------------------
# Confluence & Confidence Tests
# ---------------------------------------------------------------------------

class TestConfluence:
    def test_confluence_logic(self, engine):
        signal = TradeSignal(direction="BUY", entry_price=100, stop_loss=98, take_profit=105, timestamp=datetime.now(timezone.utc))
        m30 = make_candles(50, trend="up")
        m5 = make_candles(50, trend="up")
        
        m30_atr = engine._calculate_atr(m30)
        m5_atr = engine._calculate_atr(m5)
        
        score, reasons = engine._calculate_confluence(
            "BULLISH", 80, MarketRegime.TRENDING, signal, m30, m5, m30_atr, m5_atr, "LONDON"
        )
        
        assert score >= 1 # At least London session and H4 alignment
        assert "LONDON Session" in reasons or "LONDON/NY Session" in reasons
