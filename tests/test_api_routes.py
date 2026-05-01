"""Tests for FastAPI routes via TestClient."""

import pytest
from unittest.mock import MagicMock, patch

try:
    from fastapi.testclient import TestClient
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False

pytestmark = pytest.mark.skipif(not HAS_FASTAPI, reason="fastapi not installed")


@pytest.fixture
def client():
    """Create a TestClient with mocked services."""
    with patch("core.common.database._get_engine"), \
         patch("core.common.database._get_session_factory"):

        from api.dependencies import services

        services.config = {
            "risk_governance": {"max_drawdown_halt_pct": 10.0},
            "symbols_config": {"XAUUSDm": {"point": 0.01}},
        }
        services.risk_engine = MagicMock()
        services.risk_engine.kill_switch_active = False
        services.order_manager = MagicMock()
        services.recon_engine = MagicMock()
        services.recon_engine.get_account_summary.return_value = {}
        services.data_manager = MagicMock()
        services.strategies = {}
        services.is_trading = False

        from api.main import app
        yield TestClient(app, raise_server_exceptions=False)


class TestHealthEndpoint:
    def test_health(self, client):
        res = client.get("/api/health")
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "ok"
        assert "timestamp" in data


class TestMetricsEndpoint:
    def test_metrics(self, client):
        res = client.get("/api/metrics")
        assert res.status_code == 200
        data = res.json()
        assert "trade_count" in data
        assert "sharpe_ratio" in data


class TestTradingControl:
    @patch("api.routers.trading.mt5_service")
    def test_start_stop(self, mock_mt5, client):
        mock_mt5.connected = True
        res = client.post("/api/control/start")
        assert res.status_code == 200
        assert res.json()["status"] == "started"

        res = client.post("/api/control/stop")
        assert res.status_code == 200
        assert res.json()["status"] == "stopped"

    def test_kill_switch(self, client):
        res = client.post("/api/control/kill")
        assert res.status_code == 200
        assert res.json()["status"] == "killed"


class TestStrategiesEndpoint:
    def test_list_strategies(self, client):
        res = client.get("/api/strategies/")
        assert res.status_code == 200
        assert isinstance(res.json(), list)


class TestConfigEndpoint:
    def test_get_config(self, client):
        res = client.get("/api/config/")
        assert res.status_code == 200
