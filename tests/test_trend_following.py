"""
test_trend_following.py — Strategy Logic Test Suite
===================================================
Proves session overrides, indicator alignment, RSI overextension
blocking, and cooldown gating in the TrendFollowingStrategy.

V5-INSIGNIA Institutional Certification.
"""

import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from datetime import datetime, timezone, timedelta
from strategies.trend_following import TrendFollowingStrategy
from core.common.types import CandleArray, TradeSignal
from core.base_strategy import MarketData


# ============================================================================
# HELPERS
# ============================================================================

def _make_m15_candles(n=260, trend="BULLISH", base_price=2000.0):
    """Creates M15 CandleArray with pre-populated indicators."""
    t = (np.arange(n) * 900 + 1700000000).astype(np.int64)
    rng = np.random.default_rng(42)

    if trend == "BULLISH":
        close = base_price + np.linspace(0, 20, n) + rng.normal(0, 0.3, n)
    elif trend == "BEARISH":
        close = base_price - np.linspace(0, 20, n) + rng.normal(0, 0.3, n)
    else:
        close = base_price + rng.normal(0, 1.0, n)

    candles = CandleArray(
        time=t,
        open=(close - 0.3).astype(np.float64),
        high=(close + 2.0).astype(np.float64),
        low=(close - 2.0).astype(np.float64),
        close=close.astype(np.float64),
        tick_volume=np.full(n, 300, dtype=np.int64),
        spread=np.full(n, 15, dtype=np.int64),
    )
    return candles


def _inject_indicators(candles, ema50, ema100, ema200, adx, st_val, st_dir, rsi=50.0, atr=5.0):
    """Injects mock indicator arrays into a CandleArray."""
    n = len(candles.time)
    candles._indicators = {
        "ema_50": np.full(n, ema50),
        "ema_100": np.full(n, ema100),
        "ema_200": np.full(n, ema200),
        "adx_14": np.full(n, adx),
        "supertrend_val": np.full(n, st_val),
        "supertrend_dir": np.full(n, st_dir),
        "rsi_14": np.full(n, rsi),
        "atr_14": np.full(n, atr),
    }
    return candles


def _make_market_data(m15, session="LONDON", current_price=2010.0):
    """Wraps M15 candles in a MarketData object."""
    # Create a proper UTC datetime during the target session
    if session == "LONDON":
        dt = datetime(2026, 4, 14, 10, 0, 0, tzinfo=timezone.utc)  # 10:00 UTC = LONDON
    elif session == "NEW_YORK":
        dt = datetime(2026, 4, 14, 17, 0, 0, tzinfo=timezone.utc)  # 17:00 UTC = NY
    elif session == "LONDON/NY":
        dt = datetime(2026, 4, 14, 14, 0, 0, tzinfo=timezone.utc)  # 14:00 UTC = overlap
    else:
        dt = datetime(2026, 4, 14, 3, 0, 0, tzinfo=timezone.utc)   # 03:00 UTC = TOKYO

    h1 = _make_m15_candles(n=80)
    _inject_indicators(h1, 2010, 2005, 2000, 30, 1990, 1, atr=5.0)

    return MarketData(
        symbol="XAUUSDm",
        htf_candles=h1,
        m15_candles=m15,
        m5_candles=_make_m15_candles(n=260),
        d1_candles=None,
        current_price=current_price,
        bid=current_price,
        ask=current_price + 0.60,
        spread=0.60,
        point=0.01,
        session=session,
        timestamp=dt,
    )


# ============================================================================
# 1. SESSION OVERRIDES
# ============================================================================

class TestSessionOverrides:
    """Verifies session-specific parameter adjustments."""

    def test_session_override_london(self):
        """
        LONDON session: adx_offset=0, conf_floor=0.65.
        """
        config = {
            "TrendFollowing": {
                "enabled": True,
                "max_spread_points": 1000,
                "adx_threshold": 25,
                "min_confidence": 0.65,
            },
            "risk_governance": {"min_tick_density": 1},
        }
        strat = TrendFollowingStrategy("TrendFollowing", config)

        adj = strat.session_adjustments["LONDON"]
        assert adj["adx_offset"] == 0, "LONDON should have no ADX offset"
        assert adj["conf_floor"] == 0.65, "LONDON confidence floor should be 0.65"

    def test_session_override_newyork(self):
        """
        NEW_YORK session: adx_offset=-5, conf_floor=0.60.
        """
        config = {
            "TrendFollowing": {"enabled": True, "adx_threshold": 25, "min_confidence": 0.65},
            "risk_governance": {"min_tick_density": 1},
        }
        strat = TrendFollowingStrategy("TrendFollowing", config)

        adj = strat.session_adjustments["NEW_YORK"]
        assert adj["adx_offset"] == -5, "NY should reduce ADX threshold by 5"
        assert adj["conf_floor"] == 0.60, "NY confidence floor should be 0.60"

        # Effective threshold = 25 + (-5) = 20
        effective = strat.adx_threshold + adj["adx_offset"]
        assert effective == 20, "Effective ADX threshold in NY should be 20"


# ============================================================================
# 2. INDICATOR ALIGNMENT
# ============================================================================

class TestIndicatorAlignment:
    """Verifies signal generation with perfectly aligned indicators."""

    def test_triple_ema_bullish_supertrend_buy(self):
        """
        Triple EMA bullish (50 > 100 > 200) + SuperTrend=1 + ADX>25
        → BUY signal must be generated.
        """
        config = {
            "TrendFollowing": {
                "enabled": True,
                "max_spread_points": 1000,
                "adx_threshold": 25,
                "min_confidence": 0.05,
                "min_bars_between_signals": 0,
                "allowed_sessions": ["LONDON", "NEW_YORK", "LONDON/NY"],
            },
            "risk_governance": {"min_tick_density": 1},
            "max_spread_points": 1000,
        }
        strat = TrendFollowingStrategy("TrendFollowing", config)

        m15 = _make_m15_candles(n=260, trend="BULLISH")
        # Inject perfectly aligned bullish indicators
        _inject_indicators(
            m15,
            ema50=2020.0,    # 50 > 100 > 200 → bullish cloud
            ema100=2015.0,
            ema200=2010.0,
            adx=30.0,        # > 25 threshold
            st_val=1990.0,   # SuperTrend value (support)
            st_dir=1,        # SuperTrend direction = LONG
            rsi=55.0,        # Not overbought
        )

        md = _make_market_data(m15, session="LONDON", current_price=2020.0)
        signal = strat.generate_signal(md)

        assert signal is not None, "Aligned bullish indicators should produce a signal"
        assert signal.direction == "BUY"
        assert signal.confidence > 0.0

    def test_triple_ema_bearish_supertrend_sell(self):
        """
        Triple EMA bearish (50 < 100 < 200) + SuperTrend=-1 + ADX>25
        → SELL signal must be generated.
        """
        config = {
            "TrendFollowing": {
                "enabled": True,
                "max_spread_points": 1000,
                "adx_threshold": 25,
                "min_confidence": 0.05,
                "min_bars_between_signals": 0,
                "allowed_sessions": ["LONDON", "NEW_YORK", "LONDON/NY"],
            },
            "risk_governance": {"min_tick_density": 1},
            "max_spread_points": 1000,
        }
        strat = TrendFollowingStrategy("TrendFollowing", config)

        m15 = _make_m15_candles(n=260, trend="BEARISH")
        _inject_indicators(
            m15,
            ema50=1980.0,    # 50 < 100 < 200 → bearish cloud
            ema100=1985.0,
            ema200=1990.0,
            adx=30.0,
            st_val=2010.0,   # SuperTrend value (resistance)
            st_dir=-1,       # SuperTrend direction = SHORT
            rsi=45.0,        # Not oversold
        )

        md = _make_market_data(m15, session="LONDON", current_price=1975.0)
        signal = strat.generate_signal(md)

        assert signal is not None, "Aligned bearish indicators should produce a signal"
        assert signal.direction == "SELL"

    def test_ema_mismatch_no_signal(self):
        """
        EMA cloud bullish but SuperTrend bearish → NO signal (mismatch).
        """
        config = {
            "TrendFollowing": {
                "enabled": True,
                "max_spread_points": 1000,
                "adx_threshold": 25,
                "min_confidence": 0.05,
                "min_bars_between_signals": 0,
                "allowed_sessions": ["LONDON", "NEW_YORK", "LONDON/NY"],
            },
            "risk_governance": {"min_tick_density": 1},
            "max_spread_points": 1000,
        }
        strat = TrendFollowingStrategy("TrendFollowing", config)

        m15 = _make_m15_candles(n=260)
        _inject_indicators(
            m15,
            ema50=2020.0,    # Bullish cloud
            ema100=2015.0,
            ema200=2010.0,
            adx=30.0,
            st_val=2030.0,
            st_dir=-1,       # SuperTrend says SHORT → mismatch
            rsi=55.0,
        )

        md = _make_market_data(m15, session="LONDON")
        signal = strat.generate_signal(md)

        assert signal is None, "Cloud/SuperTrend mismatch should produce no signal"


# ============================================================================
# 3. RSI OVEREXTENSION
# ============================================================================

class TestRSIOverextension:
    """Verifies RSI blocks overextended entries."""

    def test_rsi_overbought_blocks_buy(self):
        """
        RSI > 75 should block BUY signals even with perfect alignment.
        """
        config = {
            "TrendFollowing": {
                "enabled": True,
                "max_spread_points": 1000,
                "adx_threshold": 25,
                "min_confidence": 0.05,
                "min_bars_between_signals": 0,
                "allowed_sessions": ["LONDON", "NEW_YORK", "LONDON/NY"],
            },
            "risk_governance": {"min_tick_density": 1},
            "max_spread_points": 1000,
        }
        strat = TrendFollowingStrategy("TrendFollowing", config)

        m15 = _make_m15_candles(n=260, trend="BULLISH")
        _inject_indicators(
            m15,
            ema50=2020.0, ema100=2015.0, ema200=2010.0,
            adx=30.0, st_val=1990.0, st_dir=1,
            rsi=80.0,  # OVERBOUGHT
        )

        md = _make_market_data(m15, session="LONDON")
        signal = strat.generate_signal(md)

        assert signal is None, "Overbought RSI (80) should block BUY"
        assert "Overbought" in strat.last_rejection_reason

    def test_rsi_oversold_blocks_sell(self):
        """
        RSI < 25 should block SELL signals even with perfect alignment.
        """
        config = {
            "TrendFollowing": {
                "enabled": True,
                "max_spread_points": 1000,
                "adx_threshold": 25,
                "min_confidence": 0.05,
                "min_bars_between_signals": 0,
                "allowed_sessions": ["LONDON", "NEW_YORK", "LONDON/NY"],
            },
            "risk_governance": {"min_tick_density": 1},
            "max_spread_points": 1000,
        }
        strat = TrendFollowingStrategy("TrendFollowing", config)

        m15 = _make_m15_candles(n=260, trend="BEARISH")
        _inject_indicators(
            m15,
            ema50=1980.0, ema100=1985.0, ema200=1990.0,
            adx=30.0, st_val=2010.0, st_dir=-1,
            rsi=20.0,  # OVERSOLD
        )

        md = _make_market_data(m15, session="LONDON", current_price=1975.0)
        signal = strat.generate_signal(md)

        assert signal is None, "Oversold RSI (20) should block SELL"
        assert "Oversold" in strat.last_rejection_reason


# ============================================================================
# 4. COOLDOWN GATING
# ============================================================================

class TestCooldownGating:
    """Verifies signal cooldown enforcement."""

    def test_cooldown_blocks_rapid_signals(self):
        """
        A signal within min_bars_between_signals should be rejected.
        """
        config = {
            "TrendFollowing": {
                "enabled": True,
                "max_spread_points": 1000,
                "adx_threshold": 25,
                "min_confidence": 0.05,
                "min_bars_between_signals": 20,
                "allowed_sessions": ["LONDON", "NEW_YORK", "LONDON/NY"],
            },
            "risk_governance": {"min_tick_density": 1},
            "max_spread_points": 1000,
        }
        strat = TrendFollowingStrategy("TrendFollowing", config)

        m15 = _make_m15_candles(n=260, trend="BULLISH")
        _inject_indicators(
            m15,
            ema50=2020.0, ema100=2015.0, ema200=2010.0,
            adx=30.0, st_val=1990.0, st_dir=1,
            rsi=55.0,
        )

        md = _make_market_data(m15, session="LONDON")

        # First signal should pass
        signal1 = strat.generate_signal(md)
        assert signal1 is not None, "First signal should pass"

        # Second signal immediately after should be blocked by cooldown
        signal2 = strat.generate_signal(md)
        assert signal2 is None, "Rapid second signal should be blocked"
        assert "cooldown" in strat.last_rejection_reason.lower()
