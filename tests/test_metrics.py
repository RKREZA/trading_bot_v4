"""Tests for the observability MetricsCollector."""

import pytest
from core.observability.metrics import MetricsCollector


class TestMetricsCollector:
    def test_empty_snapshot(self):
        m = MetricsCollector()
        snap = m.get_snapshot()
        assert snap["trade_count"] == 0
        assert snap["win_rate"] == 0.0
        assert snap["sharpe_ratio"] == 0.0

    def test_record_trades(self):
        m = MetricsCollector()
        m.record_trade(pnl=100.0, balance=10000.0, equity=10100.0, latency_ms=50.0)
        m.record_trade(pnl=-30.0, balance=10100.0, equity=10070.0, latency_ms=80.0)
        m.record_trade(pnl=50.0, balance=10070.0, equity=10120.0, latency_ms=45.0)

        snap = m.get_snapshot()
        assert snap["trade_count"] == 3
        assert snap["win_rate"] == pytest.approx(66.67, abs=0.01)
        assert snap["avg_latency_ms"] == pytest.approx(58.33, abs=0.01)
        assert snap["peak_equity"] == 10120.0
        assert snap["current_equity"] == 10120.0

    def test_drawdown_calculation(self):
        m = MetricsCollector()
        m.record_trade(pnl=500.0, balance=10000.0, equity=10500.0)
        m.record_trade(pnl=-200.0, balance=10500.0, equity=10300.0)

        snap = m.get_snapshot()
        assert snap["peak_equity"] == 10500.0
        assert snap["drawdown_pct"] == pytest.approx(1.90, abs=0.01)

    def test_signal_and_rejection_counters(self):
        m = MetricsCollector()
        m.record_signal()
        m.record_signal()
        m.record_rejection()

        snap = m.get_snapshot()
        assert snap["signal_count"] == 2
        assert snap["reject_count"] == 1

    def test_custom_counters(self):
        m = MetricsCollector()
        m.increment("api_calls", 5)
        m.increment("api_calls", 3)
        m.increment("ws_messages")

        snap = m.get_snapshot()
        assert snap["counters"]["api_calls"] == 8
        assert snap["counters"]["ws_messages"] == 1

    def test_reset(self):
        m = MetricsCollector()
        m.record_trade(pnl=100.0, balance=10000.0, equity=10100.0)
        m.record_signal()
        m.increment("test", 5)

        m.reset()
        snap = m.get_snapshot()
        assert snap["trade_count"] == 0
        assert snap["signal_count"] == 0
        assert snap["counters"] == {}
