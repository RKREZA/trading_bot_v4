import pytest
from unittest.mock import patch, MagicMock
from core.connection import MT5Connection, PositionManager

@pytest.fixture
def connection():
    return MT5Connection(max_retries=2, health_check_interval=0)

@pytest.fixture
def mock_mt5():
    with patch("core.connection.mt5") as mock:
        yield mock

def test_connect_success(connection, mock_mt5):
    mock_mt5.initialize.return_value = True
    assert connection.connect() is True
    assert connection.connected is True

def test_connect_failure_retries(connection, mock_mt5):
    mock_mt5.initialize.return_value = False
    mock_mt5.last_error.return_value = (1, "Mock Error")
    
    # Needs os.environ mocked so it doesn't fail on credentials
    with patch.dict('os.environ', {'MT5_LOGIN': '1', 'MT5_PASSWORD': '2', 'MT5_SERVER': '3'}):
        assert connection.connect() is False
    assert connection.connected is False

def test_position_manager_has_open_position(mock_mt5):
    conn = MT5Connection()
    conn.connected = True
    pm = PositionManager(conn)
    
    # Setup mock returns
    mock_pos = MagicMock()
    mock_pos.symbol = "XAUUSDm"
    mock_mt5.positions_get.return_value = [mock_pos]
    
    assert pm.has_open_position("XAUUSDm") is True
    assert pm.has_open_position("EURUSD") is False

def test_place_order_retries_on_busy(mock_mt5):
    conn = MT5Connection(max_retries=2)
    conn.connected = True
    conn.config = {"magic_number": 234000}
    pm = PositionManager(conn)
    
    fail_result = MagicMock()
    fail_result.retcode = 10004 # Requote
    
    success_result = MagicMock()
    success_result.retcode = 10009 # DONE
    success_result.order = 123456
    
    mock_mt5.order_send.side_effect = [fail_result, success_result]
    mock_mt5.symbol_info_tick.return_value = MagicMock(ask=100.0, bid=99.0)
    
    ticket = pm.place_order("XAUUSDm", "BUY", 0.01, 100.0, 95.0, 105.0)
    assert ticket == 123456
