from core.risk_engine import RiskEngine


def _cfg():
    return {
        "backtest": {"initial_balance": 1000.0},
        "risk_governance": {
            "risk_per_trade_pct": 0.5,
            "max_daily_loss_pct": 3.0,
            "max_drawdown_halt_pct": 10.0,
        },
    }


def test_reads_nested_risk_governance_config():
    r = RiskEngine(_cfg())
    assert r.risk_per_trade_pct == 0.5
    assert r.max_daily_loss_pct == 3.0
    assert r.max_total_drawdown_pct == 10.0


def test_circuit_breaker_triggers_on_total_drawdown():
    r = RiskEngine(_cfg())
    allowed, reason = r.check_circuit_breakers(current_balance=900.0, current_equity=800.0)
    assert allowed is False
    assert "Max Total DD" in reason


def test_update_history_tracks_losses_and_resets_on_win():
    r = RiskEngine(_cfg())
    r.update_history(-5.0, 995.0)
    r.update_history(-4.0, 991.0)
    assert r.consecutive_losses == 2
    assert r.daily_loss == 9.0

    r.update_history(3.0, 994.0)
    assert r.consecutive_losses == 0
