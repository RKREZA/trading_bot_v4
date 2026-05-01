"""Tests for the Monte Carlo simulator with distribution data."""

import pytest
from backtesting.monte_carlo import MonteCarloSimulator


@pytest.fixture
def sample_history():
    """50 trades with realistic PnL distribution."""
    import numpy as np
    rng = np.random.default_rng(42)
    trades = []
    balance = 10000.0
    for _ in range(50):
        pnl = rng.normal(20, 80)
        trades.append({"pnl": float(pnl), "balance_at_start": balance})
        balance += pnl
    return trades


class TestMonteCarloSimulator:
    def test_insufficient_data(self):
        mc = MonteCarloSimulator(iterations=100)
        result = mc.run([], 10000.0)
        assert result["status"] == "INSUFFICIENT_DATA"
        assert result["robustness_score"] == 0

    def test_few_trades_rejected(self):
        mc = MonteCarloSimulator(iterations=100)
        result = mc.run([{"pnl": 10}] * 5, 10000.0)
        assert result["status"] == "INSUFFICIENT_DATA"

    def test_basic_run(self, sample_history):
        mc = MonteCarloSimulator(iterations=200, seed=42)
        result = mc.run(sample_history, 10000.0)

        assert result["status"] == "SUCCESS"
        assert result["iterations"] == 200
        assert result["num_trades"] == 50

    def test_distribution_data_present(self, sample_history):
        mc = MonteCarloSimulator(iterations=200, seed=42)
        result = mc.run(sample_history, 10000.0)

        assert "distributions" in result
        assert "balance" in result["distributions"]
        assert "drawdown" in result["distributions"]
        assert "sharpe" in result["distributions"]

        bal_dist = result["distributions"]["balance"]
        assert "percentiles" in bal_dist
        assert "histogram" in bal_dist
        assert "p5" in bal_dist["percentiles"]
        assert "p95" in bal_dist["percentiles"]
        assert len(bal_dist["histogram"]["counts"]) == 50

    def test_equity_paths_returned(self, sample_history):
        mc = MonteCarloSimulator(iterations=200, seed=42)
        result = mc.run(sample_history, 10000.0)

        assert "equity_paths" in result
        assert len(result["equity_paths"]) > 0
        assert len(result["equity_paths"]) <= 50

    def test_summary_fields(self, sample_history):
        mc = MonteCarloSimulator(iterations=200, seed=42)
        result = mc.run(sample_history, 10000.0)

        s = result["summary"]
        assert "median_final_balance" in s
        assert "worst_case_balance_5pct" in s
        assert "probability_of_profit" in s
        assert "probability_of_ruin" in s
        assert "robustness_score" in s
        assert 0 <= s["robustness_score"] <= 100

    def test_iteration_clamping(self):
        mc = MonteCarloSimulator(iterations=50)
        assert mc.iterations == 100

        mc = MonteCarloSimulator(iterations=9999)
        assert mc.iterations == 5000

    def test_legacy_compat_fields(self, sample_history):
        mc = MonteCarloSimulator(iterations=200, seed=42)
        result = mc.run(sample_history, 10000.0)

        assert "robustness_score" in result
        assert "median_final_balance" in result
        assert "worst_case_balance_95ci" in result
        assert "probability_of_ruin" in result
