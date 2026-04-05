from core.risk_engine import RiskEngine


def _cfg():
    return {
        "risk_governance": {
            "risk_per_trade_pct": 1.0,
            "max_daily_loss_pct": 5.0,
            "max_drawdown_halt_pct": 20.0,
        },
        "backtest": {"initial_balance": 1000.0},
    }


def test_lot_size_calculation_positive():
    engine = RiskEngine(_cfg())
    lot = engine.calculate_lot_size(
        balance=1000.0,
        stop_loss_distance=1.5,
        point=0.01,
        tick_value=1.0,
        symbol="XAUUSDm",
        spread_points=2.0,
        commission_per_lot=0.0,
    )
    assert lot >= 0.01


def test_zero_sl_distance_returns_zero():
    engine = RiskEngine(_cfg())
    lot = engine.calculate_lot_size(
        balance=1000.0,
        stop_loss_distance=0.0,
        point=0.01,
        tick_value=1.0,
        symbol="XAUUSDm",
    )
    assert lot == 0.0


def test_high_cost_ratio_rejects_trade():
    cfg = _cfg()
    cfg["max_cost_ratio"] = 0.01
    engine = RiskEngine(cfg)
    lot = engine.calculate_lot_size(
        balance=1000.0,
        stop_loss_distance=0.10,
        point=0.01,
        tick_value=1.0,
        symbol="XAUUSDm",
        spread_points=50.0,
        commission_per_lot=20.0,
    )
    assert lot == 0.0
