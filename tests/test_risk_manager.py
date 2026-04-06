import pytest
from core.risk_engine import RiskEngine
from core.risk.risk_guardian import RiskGuardian

class TestRiskEngine:
    """Tests for the base RiskEngine used in backtesting."""

    def test_config_loading(self, mock_config):
        r = RiskEngine(mock_config)
        assert r.risk_per_trade_pct == 0.5
        assert r.max_daily_loss_pct == 3.0
        assert r.max_total_drawdown_pct == 10.0

    def test_lot_size_calculation(self, mock_config):
        r = RiskEngine(mock_config)
        # 10,000 balance, 0.5% risk = $50 risk
        # SL distance = 1.0 (100 points @ 0.01)
        # Risk / (Points * TickValue) = 50 / (100 * 1) = 0.5 lots
        lot = r.calculate_lot_size(
            balance=10000.0,
            stop_loss_distance=1.0,
            point=0.01,
            tick_value=1.0,
            symbol="XAUUSDm"
        )
        assert lot == 0.5

    def test_cost_ratio_rejection(self, mock_config):
        mock_config["max_cost_ratio"] = 0.1 # 10% max cost
        r = RiskEngine(mock_config)
        # High spread forcing high cost
        lot = r.calculate_lot_size(
            balance=10000.0,
            stop_loss_distance=0.5,
            point=0.01,
            tick_value=1.0,
            symbol="XAUUSDm",
            spread_points=40.0 # 40 points spread on 50 points SL is very high cost
        )
        assert lot == 0.0

    def test_drawdown_circuit_breaker(self, mock_config):
        r = RiskEngine(mock_config)
        # Initial balance 10,000. Max DD 10% (1,000).
        # Current equity 8,500 (15% DD).
        allowed, reason = r.check_circuit_breakers(current_balance=10000.0, current_equity=8500.0)
        assert allowed is False
        assert "Max Total DD" in reason

class TestRiskGuardian:
    """Tests for the Institutional RiskGuardian."""

    def test_guardian_lot_sizing_with_constraints(self, mock_config, symbol_info):
        g = RiskGuardian(mock_config)
        # 10,000 balance, 0.5% risk = $50 risk.
        # SL 2.0 = 200 points. 50 / 200 = 0.25 lots.
        lots = g.calculate_lot_size(10000.0, 2.0, symbol_info)
        assert lots == 0.25

    def test_guardian_daily_loss_limit(self, mock_config):
        g = RiskGuardian(mock_config)
        g.daily_loss = 400.0 # 4% loss, limit is 3%
        allowed, reason = g.check_governance(10000.0, 9600.0)
        assert allowed is False
        assert "KILL_SWITCH_TRIGGERED" in reason

    def test_consecutive_loss_scaling(self, mock_config, symbol_info):
        g = RiskGuardian(mock_config)
        g.consecutive_losses = 4
        # Risk should be halved twice (or reduced significantly)
        # Base risk 0.5% -> 0.25%
        lots_normal = g.calculate_lot_size(10000.0, 1.0, symbol_info)
        
        g.consecutive_losses = 0
        lots_base = g.calculate_lot_size(10000.0, 1.0, symbol_info)
        
        assert lots_normal < lots_base

    def test_max_parallel_strategies(self, mock_config):
        mock_config["risk_governance"]["max_parallel_strategies"] = 2
        g = RiskGuardian(mock_config)
        allowed, reason = g.check_governance(10000.0, 10000.0, open_positions=2)
        assert allowed is False
        assert "MAX_PARALLEL_STRATEGIES_REACHED" in reason
