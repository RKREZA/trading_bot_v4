"""
Test suite for dashboard panels.
"""

import pytest
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch
from dashboard import TradingDashboard


class TestDashboard:
    """Test dashboard panel creation."""

    def test_dashboard_initialization(self):
        """Test dashboard initializes correctly."""
        dashboard = TradingDashboard()
        assert dashboard is not None
        assert dashboard.metric_engine is not None
        assert dashboard.layout is not None

    def test_make_header_online(self):
        """Test header panel when connected."""
        dashboard = TradingDashboard()
        state = {
            "connection": {"connected": True},
        }
        panel = dashboard._make_header(state)
        assert panel is not None

    def test_make_header_offline(self):
        """Test header panel when disconnected."""
        dashboard = TradingDashboard()
        state = {
            "connection": {"connected": False},
        }
        panel = dashboard._make_header(state)
        assert panel is not None

    def test_make_environment(self):
        """Test environment panel."""
        dashboard = TradingDashboard()
        state = {
            "symbol": "XAUUSDm",
            "price": 2000.0,
            "ask": 2000.5,
            "bid": 1999.5,
            "spread": 50.0,
            "pips": 5.0,
            "session": "LONDON",
            "regime_type": "TRENDING",
            "volatility": "HIGH",
            "digits": 2
        }
        panel = dashboard._make_environment(state)
        assert panel is not None

    def test_make_risk(self):
        """Test risk panel."""
        dashboard = TradingDashboard()
        state = {
            "account": {"balance": 10000, "equity": 10500, "profit": 500},
            "equity_history": [10000, 10100, 10200, 10300, 10500]
        }
        panel = dashboard._make_risk(state)
        assert panel is not None

    def test_make_exposure(self):
        """Test exposure panel."""
        dashboard = TradingDashboard()
        state = {
            "positions": [
                {"symbol": "XAUUSDm", "volume": 0.5, "type_text": "BUY"}
            ]
        }
        panel = dashboard._make_exposure(state)
        assert panel is not None

    def test_make_performance(self):
        """Test performance panel."""
        dashboard = TradingDashboard()
        state = {
            "metrics": {
                "total_trades": 10,
                "win_rate": 60.0,
                "profit_factor": 1.5,
                "sharpe_ratio": 1.2
            }
        }
        panel = dashboard._make_performance(state)
        assert panel is not None

    def test_make_performance_empty(self):
        """Test performance panel with no metrics."""
        dashboard = TradingDashboard()
        state = {}
        panel = dashboard._make_performance(state)
        assert panel is not None

    def test_make_trades(self):
        """Test trades panel."""
        dashboard = TradingDashboard()
        state = {
            "trade_history": [
                {"timestamp": "2024-01-01T10:00:00Z", "direction": "BUY", "profit": 100},
                {"timestamp": "2024-01-01T11:00:00Z", "direction": "SELL", "profit": -50}
            ]
        }
        panel = dashboard._make_trades(state)
        assert panel is not None

    def test_make_trades_empty(self):
        """Test trades panel with no history."""
        dashboard = TradingDashboard()
        state = {}
        panel = dashboard._make_trades(state)
        assert panel is not None

    def test_make_setups(self):
        """Test setups panel."""
        dashboard = TradingDashboard()
        
        signal = MagicMock()
        signal.direction = "BUY"
        signal.confidence = 0.75
        
        state = {
            "setups": {
                "TestStrategy": {
                    "signal": signal,
                    "fidelity": 0.75,
                    "thresholds": {"min_confidence": 0.7},
                    "metrics": {"confidence": 0.75}
                }
            }
        }
        panel = dashboard._make_setups(state)
        assert panel is not None

    def test_make_footer(self):
        """Test footer panel."""
        dashboard = TradingDashboard()
        state = {
            "analysis_logs": ["Log 1", "Log 2"],
            "system_logs": ["System 1", "System 2"],
            "news_list": []
        }
        panels = dashboard._make_footer(state)
        assert len(panels) == 3

    def test_update_full(self):
        """Test full dashboard update."""
        dashboard = TradingDashboard()
        state = {
            "connection": {"connected": True},
            "symbol": "XAUUSDm",
            "price": 2000.0,
            "ask": 2000.5,
            "bid": 1999.5,
            "spread": 50.0,
            "pips": 5.0,
            "session": "LONDON",
            "regime_type": "TRENDING",
            "volatility": "HIGH",
            "digits": 2,
            "account": {"balance": 10000, "equity": 10500, "profit": 500},
            "equity_history": [10000, 10500],
            "positions": [],
            "metrics": {"total_trades": 10, "win_rate": 60.0, "profit_factor": 1.5, "sharpe_ratio": 1.2},
            "trade_history": [],
            "setups": {},
            "analysis_logs": [],
            "system_logs": [],
            "news_list": []
        }
        layout = dashboard.update(state)
        assert layout is not None