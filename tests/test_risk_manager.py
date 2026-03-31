import pytest
from hypothesis import given, strategies as st, settings, HealthCheck
from unittest.mock import MagicMock
from core.types import BotConfig

from core.risk_manager import RiskManager

@pytest.fixture
def base_config():
    return {
        "risk": {
            "risk_per_trade": 1.0,
            "max_drawdown_halt_pct": 10.0,
            "drawdown_scaling": True,
            "max_consecutive_losses": 4,
            "max_hourly_trades": 3
        },
        "session_config": {
            "LONDON": {"risk_multiplier": 1.0},
            "TOKYO": {"risk_multiplier": 0.5}
        }
    }

@pytest.fixture
def risk_manager(base_config):
    rm = RiskManager(base_config)
    rm.silent = True
    return rm

class TestRiskManagerProperties:
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    @given(balance=st.floats(min_value=100, max_value=1_000_000),
           session=st.sampled_from(["LONDON", "TOKYO", "NEW_YORK"]))
    def test_risk_never_exceeds_max(self, risk_manager, balance, session):
        risk = risk_manager.calculate_scaled_risk(balance, session)
        assert 0.0 <= risk <= 2.0  # Hard ceiling in logic

    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    @given(balance=st.floats(min_value=100, max_value=1_000_000))
    def test_risk_zero_at_max_drawdown(self, risk_manager, balance):
        # Establish high watermark
        risk_manager.calculate_scaled_risk(balance * 2.0)
        # Check current risk at 50% drawdown
        risk = risk_manager.calculate_scaled_risk(balance)
        assert risk == 0.0

def test_circuit_breaker_consecutive_losses(risk_manager):
    context = {"consecutive_losses": 4}
    allowed, reason = risk_manager.circuit_breaker.check_all(context)
    assert not allowed
    assert "consecutive losses" in reason.lower()

def test_circuit_breaker_low_margin(risk_manager):
    context = {"margin_level": 150} # Below 200 threshold
    allowed, reason = risk_manager.circuit_breaker.check_all(context)
    assert not allowed
    assert "low margin" in reason.lower()

def test_kelly_zero_on_losing_streak(risk_manager):
    # Setup a streak of losses
    for i in range(20):
        risk_manager.update_history({'ticket': i, 'pnl': -10.0})
    
    # Kelly should fall back to base or quarter-kelly scaled bounds
    # Since win_rate = 0, kelly factor is 0, so risk should be min clamped to 0.5%
    risk = risk_manager.calculate_scaled_risk(1000)
    assert risk == 0.5
