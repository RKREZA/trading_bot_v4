import pytest
import os
from unittest.mock import patch, MagicMock
from core.connection import MT5Connection

class TestMT5Connection:
    """Verifies the MT5 Connection Lifecycle and Health Monitoring."""

    @pytest.fixture
    def conn(self):
        return MT5Connection(max_retries=1, health_check_interval=0)

    def test_credential_loading_from_env(self, conn):
        """Ensure credentials are correctly pulled from environment variables."""
        with patch.dict(os.environ, {
            "MT5_LOGIN": "12345",
            "MT5_PASSWORD": "password",
            "MT5_SERVER": "MetaQuotes-Demo"
        }):
            creds = conn._get_credentials()
            assert creds["login"] == 12345
            assert creds["password"] == "password"
            assert creds["server"] == "MetaQuotes-Demo"

    @patch("core.connection.mt5")
    def test_connect_success(self, mock_mt5, conn):
        """Verify successful connection and account info initialization."""
        mock_mt5.initialize.return_value = True
        
        # Mock account info object
        mock_info = MagicMock()
        mock_info.balance = 10000.0
        mock_info.equity = 10000.0
        mock_info.login = 12345
        mock_info.server = "Demo"
        mock_info.profit = 0.0
        mock_info.margin = 0.0
        mock_info.margin_free = 10000.0
        mock_info.leverage = 100
        mock_mt5.account_info.return_value = mock_info
        
        with patch.dict(os.environ, {"MT5_LOGIN": "123", "MT5_PASSWORD": "p", "MT5_SERVER": "s"}):
            success = conn.connect()
            assert success is True
            assert conn.connected is True
            assert conn.account_info["balance"] == 10000.0

    @patch("core.connection.mt5")
    def test_health_check_dead_reconnect(self, mock_mt5, conn):
        """Ensure health check detects disconnection and triggers recovery."""
        conn.connected = True
        conn._last_health_check = 0 # Force check
        
        # 1. Health check fails (returns None)
        mock_mt5.account_info.return_value = None
        assert conn.is_alive() is False
        assert conn.connected is False

    @patch("core.connection.mt5")
    def test_ensure_connected_reconnects(self, mock_mt5, conn):
        """Verify that ensure_connected triggers a reconnect if alive check fails."""
        conn.connected = False
        
        with patch.object(conn, 'reconnect', return_value=True) as mock_reconnect:
            res = conn.ensure_connected()
            assert res is True
            mock_reconnect.assert_called_once()
