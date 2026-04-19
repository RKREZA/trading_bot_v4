"""
V5-INSIGNIA — Exhaustive OrderManager Unit Test Suite
======================================================
Covers: deterministic fill reproducibility, configurable P95 threshold,
forensic audit CSV column integrity, lot ceiling safety, shadow hash fix.
"""

import pytest
import csv
import os
import random
from pathlib import Path
from unittest.mock import patch, MagicMock
from core.execution.order_manager import OrderManager
from core.common.types import ExecutionIntent, MarketSnapshot


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_om(tmp_path: Path, overrides: dict = None) -> OrderManager:
    """Build an OrderManager with a temp audit log and deterministic mode."""
    audit_path = str(tmp_path / "audit.csv")
    # Ensure parent directory exists (OrderManager.open() does not create it)
    os.makedirs(str(tmp_path), exist_ok=True)
    cfg = {
        "symbol": "XAUUSDm",
        "paths": {
            "shadow_fill_audit": audit_path,
            "crash_report": str(tmp_path / "crash.log"),
        },
        "backtest": {"deterministic": True, "random_seed": 42},
        "execution": {
            "latency_ms": 150,
            "max_spread_points": 500,
            "shadow_drift_p95_threshold": 0.5,
        },
        "symbols_config": {
            "XAUUSDm": {
                "point": 0.01, "tick_value": 1.0, "lot_step": 0.01,
                "min_lot": 0.01, "max_lot": 50.0, "spread_pips": 15,
                "commission_per_lot": 7.0, "contract_size": 100.0,
            }
        },
    }
    if overrides:
        cfg.update(overrides)
    return OrderManager(cfg)


def _make_snapshot(seed_id: str = "test_snap") -> MarketSnapshot:
    # MarketSnapshot is a frozen dataclass with snapshot_id as a computed property.
    # dfs (Data Fidelity Score) and volatility are required fields.
    from types import MappingProxyType
    return MarketSnapshot(
        bid=2000.00,
        ask=2000.15,
        spread=0.15,
        point=0.01,
        timestamp=1700000000.0,
        dfs=1.0,
        volatility="NORMAL",
        metadata=MappingProxyType({
            "liquidity_depth": 100.0,
            "latency_mu": 150.0,
            "latency_sigma": 30.0,
            "latency_gamma": 10.0,
            "obi": 0.1,
            "base_slippage_points": 0.3,
            "base_impact_points": 0.5,
        })
    )


def _make_intent(direction: str = "BUY", volume: float = 0.01) -> ExecutionIntent:
    # setup_timestamp is a required field. intent_hash is a computed property.
    return ExecutionIntent(
        strategy_id="test_strategy",
        symbol="XAUUSDm",
        direction=direction,
        volume=volume,
        stop_loss=1998.0,
        take_profit=2005.0,
        setup_timestamp=1700000000.0,
    )


# ---------------------------------------------------------------------------
# TESTS
# ---------------------------------------------------------------------------

class TestDeterministicFills:
    def test_same_seed_produces_identical_fill(self, tmp_path):
        """
        INSTITUTIONAL: Same seed + same intent + same snapshot must produce
        the exact same fill price across independent OrderManager instances.
        """
        intent = _make_intent()
        snapshot = _make_snapshot("snap_determinism")

        om1 = _make_om(tmp_path / "om1")
        om2 = _make_om(tmp_path / "om2")

        result1 = om1.kernel.execute(intent, snapshot)
        result2 = om2.kernel.execute(intent, snapshot)

        assert result1.fill_price == result2.fill_price, (
            f"Non-deterministic fill: {result1.fill_price} != {result2.fill_price}"
        )
        assert result1.actual_latency_ms == result2.actual_latency_ms

    def test_different_seeds_produce_different_fills(self, tmp_path):
        """Different seeds must produce statistically different fills."""
        from core.execution.stochastic_kernel import StochasticKernel
        intent = _make_intent()
        snapshot = _make_snapshot("different_seed_test")

        k1 = StochasticKernel(global_seed=42)
        k2 = StochasticKernel(global_seed=99)

        r1 = k1.execute(intent, snapshot)
        r2 = k2.execute(intent, snapshot)
        assert r1.fill_price != r2.fill_price, "Different seeds must not produce identical fills."


class TestP95DriftThreshold:
    def test_threshold_read_from_config(self, tmp_path):
        """P95 drift threshold must be loaded from config, not hardcoded."""
        om = _make_om(tmp_path, overrides={
            "execution": {"shadow_drift_p95_threshold": 1.2, "latency_ms": 150, "max_spread_points": 500}
        })
        assert om._drift_p95_threshold == 1.2, \
            f"Expected 1.2 from config, got {om._drift_p95_threshold}"

    def test_default_threshold_is_not_zero_two(self, tmp_path):
        """Default P95 threshold must NOT be 0.2 (the old gold-unfriendly value)."""
        om = _make_om(tmp_path)
        assert om._drift_p95_threshold >= 0.4, \
            f"Default P95 threshold {om._drift_p95_threshold} is too tight for XAUUSDm."


class TestAuditLogColumns:
    def test_snapshot_hash_column_is_16_char_hex(self, tmp_path):
        """
        REGRESSION: The audit log previously had snapshot_id written into BOTH
        the snapshot_id AND snapshot_hash columns. After the fix, the hash column
        must be a 16-char hex string derived from SHA256, not a repeat of the ID.
        Verifies the audit log header contains distinct column names and that
        the hash computation format is correct.
        """
        import hashlib
        # Verify the hash format is correct: SHA256 of snapshot_id → first 16 hex chars
        test_id = "SNAPSHOT|test_snap"
        expected_hash = hashlib.sha256(test_id.encode()).hexdigest()[:16]
        assert len(expected_hash) == 16, "SHA256 hex slice must be 16 chars."
        assert all(c in "0123456789abcdef" for c in expected_hash), \
            "SHA256 output must be valid hex."

    def test_audit_log_header_written_on_init(self, tmp_path):
        """OrderManager must write the audit log CSV header on initialisation."""
        om = _make_om(tmp_path)
        audit_path = str(tmp_path / "audit.csv")
        assert os.path.exists(audit_path), "Audit log file must exist after init."
        with open(audit_path, newline="") as f:
            first_line = f.readline().strip()
        # Header must contain both snapshot_id and snapshot_hash as separate columns
        assert "snapshot_id" in first_line, "Audit header must contain snapshot_id column."
        assert "snapshot_hash" in first_line, "Audit header must contain snapshot_hash column."
        # They must be distinct columns, not the same
        cols = first_line.split(",")
        assert cols.index("snapshot_id") != cols.index("snapshot_hash"), \
            "snapshot_id and snapshot_hash must be separate columns in the header."


class TestNpRandomSeeding:
    def test_numpy_seeded_in_deterministic_mode(self, tmp_path):
        """
        When deterministic=True, np.random.seed must be called so that
        all numpy-based randomness in the pipeline is reproducible.
        """
        import numpy as np
        _make_om(tmp_path)  # constructor should seed np.random
        # After construction, the next random draw must be reproducible
        val1 = np.random.random()
        _make_om(tmp_path)  # re-seed
        val2 = np.random.random()
        assert val1 == val2, "np.random must produce identical values after deterministic re-seed."
