"""
Unit tests for StrategyEngine.
Tests can run without MT5 terminal — they use synthetic candle data.
"""

import sys
import os
import pytest
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.strategy_engine import StrategyEngine, TradeSignal, TRADEABLE_SESSIONS


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
            "time": 1700000000 + i * 1800,  # 30-min intervals
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
        "strategy": {"min_confluence_score": 3, "min_confidence": 40, "cooldown_candles": 3},
        "symbols_config": {"TEST": {"point": 0.01, "contract_size": 1, "lot": 0.01}},
    }


@pytest.fixture
def engine(config):
    return StrategyEngine(config)


# ---------------------------------------------------------------------------
# EMA Tests
# ---------------------------------------------------------------------------

class TestEMA:
    def test_ema_length_matches_input(self, engine):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = engine._ema(data, 3)
        assert len(result) == len(data)

    def test_ema_first_value_equals_input(self, engine):
        data = np.array([10.0, 20.0, 30.0])
        result = engine._ema(data, 2)
        assert result[0] == 10.0

    def test_ema_trending_up(self, engine):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        result = engine._ema(data, 3)
        # EMA should be increasing for upward data
        for i in range(1, len(result)):
            assert result[i] > result[i - 1]

    def test_ema_constant_data(self, engine):
        data = np.array([5.0] * 10)
        result = engine._ema(data, 3)
        np.testing.assert_allclose(result, 5.0)


# ---------------------------------------------------------------------------
# ATR Tests
# ---------------------------------------------------------------------------

class TestATR:
    def test_atr_returns_float(self, engine):
        candles = make_candles(20, trend="up")
        result = engine._calculate_atr(candles)
        assert isinstance(result, float)
        assert result > 0

    def test_atr_insufficient_data_returns_default(self, engine):
        candles = make_candles(5)
        result = engine._calculate_atr(candles, period=14)
        assert result == 100.0  # default fallback

    def test_atr_flat_market_is_small(self, engine):
        candles = make_candles(30, base_price=100.0, trend="flat", step=0.01)
        result = engine._calculate_atr(candles)
        assert result < 1.0  # flat market should have small ATR


# ---------------------------------------------------------------------------
# H4 Trend Tests
# ---------------------------------------------------------------------------

class TestH4Trend:
    def test_strong_uptrend(self, engine):
        candles = make_candles(60, base_price=100.0, trend="up", step=1.0)
        trend, strength = engine._get_h4_trend(candles)
        assert trend == "BULLISH"
        assert strength >= 60

    def test_strong_downtrend(self, engine):
        candles = make_candles(60, base_price=200.0, trend="down", step=1.0)
        trend, strength = engine._get_h4_trend(candles)
        assert trend == "BEARISH"
        assert strength >= 60

    def test_flat_market_is_ranging(self, engine):
        candles = make_candles(60, base_price=100.0, trend="flat", step=0.001)
        trend, _ = engine._get_h4_trend(candles)
        assert trend == "RANGING"

    def test_insufficient_data(self, engine):
        candles = make_candles(10)
        trend, strength = engine._get_h4_trend(candles)
        assert trend == "RANGING"
        assert strength == 0


# ---------------------------------------------------------------------------
# Session Filter Tests
# ---------------------------------------------------------------------------

class TestSessionFilter:
    def test_tradeable_sessions_constant(self):
        assert "LONDON" in TRADEABLE_SESSIONS
        assert "NEW_YORK" in TRADEABLE_SESSIONS
        assert "LONDON/NY" in TRADEABLE_SESSIONS
        assert "CLOSED" not in TRADEABLE_SESSIONS
        assert "TOKYO" not in TRADEABLE_SESSIONS

    def test_analyze_skips_closed_session(self, engine):
        h4 = make_candles(60, trend="up", step=1.0)
        m30 = make_candles(110, trend="up", step=0.5)
        m15 = make_candles(110, trend="up", step=0.3)
        price = m30[-1]["close"]

        result = engine.analyze("TEST", h4, m30, m15, price, session="CLOSED")
        assert result is None

    def test_analyze_skips_tokyo_session(self, engine):
        h4 = make_candles(60, trend="up", step=1.0)
        m30 = make_candles(110, trend="up", step=0.5)
        m15 = make_candles(110, trend="up", step=0.3)
        price = m30[-1]["close"]

        result = engine.analyze("TEST", h4, m30, m15, price, session="TOKYO")
        assert result is None


# ---------------------------------------------------------------------------
# Confluence & Confidence Tests
# ---------------------------------------------------------------------------

class TestConfluence:
    def test_max_confluence_score(self, engine):
        signal = TradeSignal(
            direction="BUY", entry_price=100, stop_loss=95, take_profit=110,
            confidence=0, confluence_score=0, rejection_type="BREAKOUT",
        )
        # Bullish M30 candles (last 2 are green)
        m30 = make_candles(10, trend="up", step=1.0)
        score, reasons = engine._calculate_confluence("BULLISH", 80, "BULLISH", signal, m30)
        # H4 strong (2) + M30 aligned (2) + Breakout (2) + Momentum (1) = 7
        assert score == 7
        assert len(reasons) == 4

    def test_confidence_bounds(self, engine):
        signal = TradeSignal(
            direction="BUY", entry_price=100, stop_loss=95, take_profit=110,
            confidence=0, confluence_score=0, rr_ratio=2.5,
        )
        conf = engine._calculate_confidence(7, 90, signal)
        assert 30 <= conf <= 95


# ---------------------------------------------------------------------------
# Determine Trend Alias Test
# ---------------------------------------------------------------------------

class TestDetermineTrend:
    def test_alias_returns_string(self, engine):
        candles = make_candles(60, trend="up", step=1.0)
        result = engine._determine_trend(candles)
        assert result in ("BULLISH", "BEARISH", "RANGING")
