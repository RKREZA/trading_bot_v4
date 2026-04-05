from core.execution_engine import ExecutionEngine
from core.types import TradeSignal


def _cfg():
    return {
        "execution": {
            "latency_ms": 0,
            "slippage_pips": 0.0,
            "max_spread_pips": 5.0,
        }
    }


def test_rejects_order_when_spread_too_high():
    engine = ExecutionEngine(_cfg())
    sig = TradeSignal(direction="BUY", price=100.0, stop_loss=99.0, take_profit=101.0)

    fill = engine.execute_order(sig, "XAUUSDm", current_price=100.0, spread=0.2, point=0.01)
    assert fill is None


def test_executes_order_with_valid_spread():
    engine = ExecutionEngine(_cfg())
    sig = TradeSignal(direction="SELL", price=100.0, stop_loss=101.0, take_profit=99.0)

    fill = engine.execute_order(sig, "XAUUSDm", current_price=100.0, spread=0.02, point=0.01, timestamp=12345)

    assert fill is not None
    assert fill["direction"] == "SELL"
    assert fill["timestamp"] == 12345
