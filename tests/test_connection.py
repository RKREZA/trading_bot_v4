import os
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from core.connection import MT5Connection, PositionManager


def _account_info(balance: float = 1000.0):
    return SimpleNamespace(
        login=123,
        server="Demo",
        balance=balance,
        equity=balance,
        profit=0.0,
        margin=100.0,
        margin_free=900.0,
        margin_level=900.0,
    )


@pytest.fixture
def env_vars():
    with patch.dict(os.environ, {"MT5_LOGIN": "1", "MT5_PASSWORD": "x", "MT5_SERVER": "demo"}):
        yield


@pytest.fixture
def mock_mt5():
    with patch("core.connection.mt5") as mock:
        yield mock


def test_connect_success(env_vars, mock_mt5):
    conn = MT5Connection(max_retries=1, health_check_interval=0)
    mock_mt5.initialize.return_value = True
    mock_mt5.account_info.return_value = _account_info()

    with patch("core.connection.time.sleep", return_value=None):
        assert conn.connect() is True

    assert conn.connected is True


def test_connect_failure_returns_false(env_vars, mock_mt5):
    conn = MT5Connection(max_retries=2, health_check_interval=0)
    mock_mt5.initialize.return_value = False
    mock_mt5.last_error.return_value = (1, "mock")

    with patch("core.connection.time.sleep", return_value=None):
        assert conn.connect() is False

    assert conn.connected is False


def test_position_manager_filters_by_magic(mock_mt5):
    conn = MT5Connection()
    conn.connected = True
    conn.config = {"magic_number": 234000}
    conn.ensure_connected = lambda: True

    p1 = SimpleNamespace(symbol="XAUUSDm", magic=234000)
    p2 = SimpleNamespace(symbol="XAUUSDm", magic=999999)
    mock_mt5.positions_get.return_value = [p1, p2]

    pm = PositionManager(conn)
    positions = pm.get_open_positions("XAUUSDm")

    assert len(positions) == 1
    assert positions[0].magic == 234000


def test_calculate_lot_size_returns_bounded_value(mock_mt5):
    conn = MT5Connection()
    conn.connected = True
    conn.account_info = {"balance": 1000.0}
    conn.ensure_connected = lambda: True
    conn.config = {"strategy_defaults": {"min_sl_points": 150}}

    mock_mt5.symbol_info.return_value = SimpleNamespace(
        point=0.01,
        trade_tick_value=1.0,
        volume_min=0.01,
        volume_max=5.0,
        volume_step=0.01,
    )

    signal = SimpleNamespace(entry_price=100.0, stop_loss=99.0)
    pm = PositionManager(conn)
    lot = pm.calculate_lot_size("XAUUSDm", signal, risk_percent=1.0, account_balance=1000.0)

    assert 0.01 <= lot <= 5.0
