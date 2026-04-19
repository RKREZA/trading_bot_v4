"""
Test suite for core performance metrics calculations.
"""

import pytest
import numpy as np
from core.data.metrics import MetricEngine


class TestPerformanceMetrics:
    """Test performance metric calculations."""

    def test_calculate_sharpe_ratio_basic(self):
        """Test basic Sharpe ratio calculation."""
        returns = [0.01, 0.02, -0.01, 0.03, 0.02]
        sharpe = MetricEngine.calculate_sharpe_ratio(returns)
        assert sharpe > 0

    def test_calculate_sharpe_ratio_empty(self):
        """Test Sharpe ratio with empty returns."""
        sharpe = MetricEngine.calculate_sharpe_ratio([])
        assert sharpe == 0.0

    def test_calculate_sortino_ratio(self):
        """Test Sortino ratio calculation."""
        returns = [0.01, 0.02, -0.01, 0.03, 0.02]
        sortino = MetricEngine.calculate_sortino_ratio(returns)
        assert sortino > 0

    def test_calculate_win_rate(self):
        """Test win rate calculation."""
        profits = [100, -50, 200, -30, 150, 80]
        win_rate = MetricEngine.calculate_win_rate(profits)
        assert win_rate == pytest.approx(66.67, rel=0.1)

    def test_calculate_profit_factor(self):
        """Test profit factor calculation."""
        profits = [100, -50, 200, -30, 150]
        pf = MetricEngine.calculate_profit_factor(profits)
        assert pf > 1

    def test_calculate_max_consecutive_losses(self):
        """Test consecutive losses calculation."""
        profits = [100, -50, -30, -20, 200, -10]
        max_losses = MetricEngine.calculate_max_consecutive_losses(profits)
        assert max_losses == 3

    def test_calculate_performance_metrics_empty(self):
        """Test performance metrics with empty trade history."""
        metrics = MetricEngine.calculate_performance_metrics([])
        assert metrics["total_trades"] == 0
        assert metrics["win_rate"] == 0.0

    def test_calculate_performance_metrics_full(self):
        """Test full performance metrics calculation."""
        trade_history = [
            {"profit": 100},
            {"profit": -50},
            {"profit": 200},
            {"profit": -30},
            {"profit": 150}
        ]
        metrics = MetricEngine.calculate_performance_metrics(trade_history)
        assert metrics["total_trades"] == 5
        assert metrics["win_rate"] == pytest.approx(60.0, rel=0.1)
        assert metrics["profit_factor"] > 1