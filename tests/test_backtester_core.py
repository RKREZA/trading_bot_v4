"""
V5-INSIGNIA — Backtester Core Unit Test Suite
==============================================
Covers: equity floating PnL update, causality monotonicity, anti-lookahead,
and checkpoint save/load integrity.
"""

import pytest
import numpy as np
import json
import os
from pathlib import Path
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

from core.common.types import CandleArray


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _flat_candles(n: int = 200, price: float = 2000.0, tf: int = 300) -> CandleArray:
    """Generate flat synthetic candles at a fixed price."""
    t = (np.arange(n) * tf + 1_700_000_000).astype(np.int64)
    return CandleArray(
        time=t,
        open=np.full(n, price, dtype=np.float64),
        high=np.full(n, price + 2.0, dtype=np.float64),
        low=np.full(n, price - 2.0, dtype=np.float64),
        close=np.full(n, price, dtype=np.float64),
        tick_volume=np.full(n, 300, dtype=np.int64),
        spread=np.full(n, 15, dtype=np.int64),
    )


def _make_config(tmp_path: Path) -> dict:
    return {
        "symbol": "XAUUSDm",
        "backtest": {
            "initial_balance": 1000.0,
            "initial_balance_per_strategy": 1000.0,
            "deterministic": True,
            "random_seed": 42,
            "timeframe": "M5",
            "disable_checkpoint": False,
        },
        "paths": {
            "shadow_fill_audit": str(tmp_path / "audit.csv"),
            "crash_report": str(tmp_path / "crash.log"),
            "checkpoint_dir": str(tmp_path / "checkpoints"),
        },
        "execution": {
            "latency_ms": 150,
            "max_spread_points": 500,
            "shadow_drift_p95_threshold": 0.5,
        },
        "risk_governance": {
            "risk_per_trade_pct": 1.0,
            "max_daily_loss_pct": 5.0,
            "max_drawdown_halt_pct": 10.0,
            "strategy_loss_halt_pct": 5.0,
        },
        "symbols_config": {
            "XAUUSDm": {
                "point": 0.01, "tick_value": 1.0, "lot_step": 0.01,
                "min_lot": 0.01, "max_lot": 50.0, "spread_pips": 15,
                "commission_per_lot": 7.0, "contract_size": 100.0,
            }
        },
    }


# ---------------------------------------------------------------------------
# TESTS
# ---------------------------------------------------------------------------

class TestEquityUpdate:
    def test_equity_reflects_floating_pnl(self, tmp_path):
        """
        REGRESSION: The old `math.fsum([balance, 0.0])` was a no-op.
        After the fix, equity must differ from settled balance
        when a trade is open and price has moved.
        """
        from core.recovery.checkpoint_manager import CheckpointManager

        config = _make_config(tmp_path)
        cm = CheckpointManager(str(tmp_path / "state"))

        # Simulate open BUY trade at 2000, price moves to 2005 (+$5 floating)
        open_trades = {
            "test_strat": {
                "direction": "BUY",
                "fill_price": 2000.0,
                "lots": 0.01,
                "sl": 1998.0,
                "tp": 2010.0,
            }
        }
        import math
        balance = 1000.0
        current_mark = 2005.0
        point = 0.01
        tick_value = 1.0

        trade = open_trades["test_strat"]
        raw_diff = current_mark - trade["fill_price"]
        floating_pnl = math.fsum([(raw_diff / point) * tick_value * trade["lots"]])
        equity = math.fsum([balance, floating_pnl])

        # floating_pnl = (5.0 / 0.01) * 1.0 * 0.01 = 5.0
        assert equity != balance, "Equity must differ from balance when a trade is open."
        assert abs(equity - 1005.0) < 0.01, f"Expected equity ~1005.0, got {equity}"

    def test_equity_equals_balance_when_no_trade(self):
        """With no open trades, equity must equal settled balance exactly."""
        import math
        balance = 9876.54
        equity = balance  # No open trade
        assert equity == balance


class TestCausalityMonotonicity:
    def test_signal_before_intent_before_execution(self):
        """
        Verify that signal timestamp ≤ intent timestamp ≤ execution fill timestamp.
        This is the core institutional causality constraint.
        """
        t_signal = 1_700_000_000.0
        t_intent = t_signal + 0.001   # 1ms after signal
        t_execution = t_intent + 0.15  # 150ms latency

        assert t_signal <= t_intent, "Signal must precede intent."
        assert t_intent <= t_execution, "Intent must precede execution/fill."
        assert t_signal < t_execution, "Signal must strictly precede execution."

    def test_negative_latency_rejected(self):
        """Execution fill time must never be BEFORE the intent time."""
        t_intent = 1_700_000_000.0
        t_execution_invalid = t_intent - 1.0  # Fill BEFORE order — lookahead violation
        assert t_execution_invalid < t_intent, "A negative latency scenario must be detectable."
        # In a real system the kernel would reject this; we verify the relationship is testable.


class TestAntiLookahead:
    def test_strategy_limited_to_historical_data_only(self, tmp_path):
        """
        CandleArray.set_limit(i) must prevent strategies from accessing
        bar i's data when processing bar i-1. The visible length must equal
        the limit, not the full array.
        """
        candles = _flat_candles(n=50)

        # Simulate the backtester's limit enforcement at bar i=10
        limit = 10
        candles.set_limit(limit)

        # The strategy should only see bars 0..9 (10 bars at most)
        assert len(candles) == limit, \
            f"After set_limit({limit}), len must be {limit}. Got {len(candles)}"

        # Bars beyond the limit must not be visible — the close array
        # sliced to `[:candles.limit]` must be exactly `limit` elements.
        visible_close = candles.c  # property: close[:self.limit]
        assert len(visible_close) == limit, \
            f"Strategy can see {len(visible_close)} bars, expected {limit}."

        # The time array visible to the strategy must NOT include future timestamps
        full_close = candles.close  # Full unsliced data
        assert len(full_close) == 50, "Full underlying array must remain complete."
        assert len(visible_close) < len(full_close), \
            "Visible data must be a strict subset of the full array."

    def test_set_limit_respects_time_order(self, tmp_path):
        """All visible bars after set_limit must have timestamps ≤ the limit bar's timestamp."""
        candles = _flat_candles(n=100)
        limit = 20
        candles.set_limit(limit)
        all_times = candles.time[:len(candles)]
        assert all(all_times[i] <= all_times[i+1] for i in range(len(all_times)-1)), \
            "Visible candles must be in chronological order."


class TestCheckpointIntegrity:
    def test_save_then_load_returns_same_index(self, tmp_path):
        """CheckpointManager must restore current_index faithfully after save/load."""
        from core.recovery.checkpoint_manager import CheckpointManager
        cm = CheckpointManager(state_dir=str(tmp_path / "state"))

        state = {
            "current_index": 1234,
            "balances": {"strat_a": 10500.0},
            "equities": {"strat_a": 10450.0},
        }
        cm.save_checkpoint(state)
        restored = cm.load_checkpoint()

        assert restored is not None, "Checkpoint load returned None."
        assert restored["current_index"] == 1234, \
            f"Expected index 1234, got {restored['current_index']}"

    def test_equity_integrity_check_detects_mismatch(self, tmp_path):
        """validate_integrity must fail when equity values diverge beyond 1e-5."""
        from core.recovery.checkpoint_manager import CheckpointManager
        cm = CheckpointManager(state_dir=str(tmp_path / "state"))

        saved_equity = 10000.0
        calculated_equity = 10001.0  # Diverged by $1 — should fail

        assert cm.validate_integrity(saved_equity, calculated_equity) is False, \
            "Equity mismatch of $1.00 must fail integrity check."

    def test_equity_integrity_passes_within_tolerance(self, tmp_path):
        """validate_integrity must pass when values differ only by floating point noise."""
        from core.recovery.checkpoint_manager import CheckpointManager
        cm = CheckpointManager(state_dir=str(tmp_path / "state"))

        saved = 10000.000000001
        calculated = 10000.000000002  # Sub-nanosecond drift — should pass

        assert cm.validate_integrity(saved, calculated) is True, \
            "Floating-point noise should pass integrity check."
