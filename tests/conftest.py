"""
V5-INSIGNIA Institutional Test Fixtures (conftest.py)
=====================================================
Shared fixtures for the exhaustive institutional test suite.
Provides mock configs, candle factories, and component constructors.
"""

import pytest
import numpy as np
from datetime import datetime, timezone
from unittest.mock import MagicMock
from core.common.types import CandleArray, TradeSignal, FilteredSignal
from core.base_strategy import MarketData
from core.common.exceptions import CriticalRiskViolationError


# ---------------------------------------------------------------------------
# CONFIGURATION FIXTURES
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_config():
    """Standard Institutional Bot Configuration with all required sections."""
    return {
        "symbol": "XAUUSDm",
        "paths": {
            "strategy_health_file": "config/strategy_health.json",
            "shadow_fill_audit": "logs/shadow_fill_audit.csv",
            "crash_report": "logs/crash_report.log",
        },
        "risk_governance": {
            "risk_per_trade_pct": 1.0,
            "max_daily_loss_pct": 3.0,
            "max_drawdown_halt_pct": 10.0,
            "max_parallel_strategies": 4,
            "strategy_loss_halt_pct": 3.0,
        },
        "backtest": {
            "initial_balance": 10000.0,
            "initial_balance_per_strategy": 5000.0,
            "deterministic": True,
            "random_seed": 42,
            "timeframe": "M5",
        },
        "execution": {
            "latency_ms": 150,
            "max_spread_points": 500.0,
        },
        "symbols_config": {
            "XAUUSDm": {
                "point": 0.01,
                "tick_value": 1.0,
                "lot_step": 0.01,
                "min_lot": 0.01,
                "max_lot": 50.0,
                "spread_pips": 15,
                "commission_per_lot": 7.0,
                "contract_size": 100.0,
            }
        },
        "TrendFollowing": {
            "enabled": True,
            "max_spread_points": 1000,
            "adx_threshold": 25,
            "min_confidence": 0.65,
        },
    }


@pytest.fixture
def symbol_info():
    """Standard symbol properties for risk and sizing tests."""
    return {
        "point": 0.01,
        "tick_value": 1.0,
        "min_lot": 0.01,
        "max_lot": 50.0,
        "lot_step": 0.01,
        "spread_pips": 15,
        "commission_per_lot": 7.0,
        "contract_size": 100.0,
    }


# ---------------------------------------------------------------------------
# CANDLE FACTORIES
# ---------------------------------------------------------------------------

@pytest.fixture
def candle_factory():
    """
    Factory to create synthetic CandleArrays for testing.
    Supports FLAT, BULLISH, BEARISH trends with configurable volatility.
    """
    def _create(n=100, trend="FLAT", volatility=1.0, base_price=2000.0, tf_seconds=300):
        t = (np.arange(n) * tf_seconds + 1700000000).astype(np.int64)

        if trend == "BULLISH":
            close = base_price + np.linspace(0, 10 * volatility, n)
        elif trend == "BEARISH":
            close = base_price - np.linspace(0, 10 * volatility, n)
        else:
            rng = np.random.default_rng(42)
            close = base_price + rng.normal(0, volatility, n)

        high = close + 1.5 * volatility
        low = close - 1.5 * volatility
        open_ = close - 0.5 * volatility

        return CandleArray(
            time=t,
            open=open_.astype(np.float64),
            high=high.astype(np.float64),
            low=low.astype(np.float64),
            close=close.astype(np.float64),
            tick_volume=np.full(n, 300, dtype=np.int64),
            spread=np.full(n, 15, dtype=np.int64),
        )
    return _create


@pytest.fixture
def m1_candle_factory():
    """
    Factory to create synthetic M1 CandleArrays that fit inside an M5 bar.
    Each M5 bar contains 5 M1 candles. Used for intra-bar replay tests.
    """
    def _create(m5_time, n_bars=5, base_price=2000.0, volatility=1.0):
        t = (np.arange(n_bars) * 60 + m5_time).astype(np.int64)
        rng = np.random.default_rng(99)
        close = base_price + rng.normal(0, volatility, n_bars).cumsum()
        high = close + abs(rng.normal(0, 0.5 * volatility, n_bars))
        low = close - abs(rng.normal(0, 0.5 * volatility, n_bars))
        open_ = np.roll(close, 1)
        open_[0] = base_price

        return CandleArray(
            time=t,
            open=open_.astype(np.float64),
            high=high.astype(np.float64),
            low=low.astype(np.float64),
            close=close.astype(np.float64),
            tick_volume=np.full(n_bars, 100, dtype=np.int64),
            spread=np.full(n_bars, 15, dtype=np.int64),
        )
    return _create


@pytest.fixture
def market_data_factory(candle_factory):
    """Factory to create MarketData snapshots for strategy testing."""
    def _create(trend="BULLISH", session="LONDON", n=260):
        m5 = candle_factory(n=n, trend=trend)
        h1 = candle_factory(n=80, trend=trend)
        m15 = candle_factory(n=120, trend=trend)

        current_price = float(m5.close[-1])

        return MarketData(
            symbol="XAUUSDm",
            htf_candles=h1,
            m15_candles=m15,
            m5_candles=m5,
            d1_candles=None,
            current_price=current_price,
            bid=current_price,
            ask=current_price + 1.0,
            spread=1.0,
            point=0.01,
            session=session,
            timestamp=datetime.now(timezone.utc),
        )
    return _create


# ---------------------------------------------------------------------------
# COMPONENT FIXTURES
# ---------------------------------------------------------------------------

@pytest.fixture
def risk_guardian(mock_config):
    """Pre-configured RiskGuardian instance for testing."""
    from core.risk.risk_guardian import RiskGuardian
    guardian = RiskGuardian(mock_config)
    guardian.silent = True  # Suppress log noise during tests
    return guardian


@pytest.fixture
def order_manager(mock_config, tmp_path):
    """Pre-configured OrderManager with temp audit log."""
    # Override audit log path to avoid filesystem side-effects
    mock_config["paths"]["shadow_fill_audit"] = str(tmp_path / "audit.csv")
    from core.execution.order_manager import OrderManager
    return OrderManager(mock_config)
