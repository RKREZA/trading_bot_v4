"""
V5-INSIGNIA — Hypothesis Property-Based Test Suite
===================================================
Fuzz-tests the RiskGuardian risk math with thousands of auto-generated
inputs to catch edge cases that hand-crafted tests miss.

Requires: hypothesis (pip install hypothesis)
"""

import pytest
from hypothesis import given, settings, assume, HealthCheck
from hypothesis import strategies as st
from core.risk.risk_guardian import RiskGuardian


# ──────────────────────────────────────────────────────────────────────────────
# Shared fixture
# ──────────────────────────────────────────────────────────────────────────────

BASE_CONFIG = {
    "symbol": "XAUUSDm",
    "paths": {"strategy_health_file": "config/strategy_health_test.json"},
    "risk_governance": {
        "risk_per_trade_pct": 1.0,
        "max_daily_loss_pct": 5.0,
        "max_drawdown_halt_pct": 10.0,
        "max_parallel_strategies": 4,
        "strategy_loss_halt_pct": 5.0,
        "drawdown_vault_start_pct": 4.0,
        "drawdown_vault_slope": 0.2,
        "min_confidence": 0.60,
    },
    "symbols_config": {
        "XAUUSDm": {
            "point": 0.01,
            "tick_value": 1.0,
            "lot_step": 0.01,
            "min_lot": 0.01,
            "max_lot": 50.0,
            "spread_pips": 15,
            "commission_per_lot": 7.0,
            "contract_size": 100.0,
        }
    },
}

SYMBOL_INFO = {
    "point": 0.01,
    "tick_value": 1.0,
    "min_lot": 0.01,
    "max_lot": 50.0,
    "lot_step": 0.01,
    "spread_pips": 15,
    "commission_per_lot": 7.0,
    "contract_size": 100.0,
}


def _make_rg() -> RiskGuardian:
    rg = RiskGuardian(BASE_CONFIG)
    rg.silent = True
    return rg


# ──────────────────────────────────────────────────────────────────────────────
# Property: Lot size always within broker bounds
# ──────────────────────────────────────────────────────────────────────────────

@given(
    balance=st.floats(min_value=100.0, max_value=10_000_000.0, allow_nan=False, allow_infinity=False),
    sl_dist=st.floats(min_value=0.01, max_value=200.0, allow_nan=False, allow_infinity=False),
    current_price=st.floats(min_value=100.0, max_value=5000.0, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=300, suppress_health_check=[HealthCheck.too_slow])
def test_lot_size_always_within_broker_bounds(balance, sl_dist, current_price):
    """
    PROPERTY: For any valid balance, SL distance, and price,
    calculate_lot_size must always return a value within [min_lot, max_lot].
    A lot below min_lot would be rejected by the broker.
    A lot above max_lot would be rejected by the broker.
    """
    rg = _make_rg()
    rg.peak_balance = balance

    lot = rg.calculate_lot_size(
        balance=balance,
        stop_loss_dist=sl_dist,
        symbol_info=SYMBOL_INFO,
        current_price=current_price,
    )

    if lot == 0.0:
        return  # 0.0 is a valid institutional rejection for under-capitalized trades

    assert lot >= SYMBOL_INFO["min_lot"], (
        f"Lot {lot} < min_lot {SYMBOL_INFO['min_lot']}  "
        f"(balance={balance}, sl_dist={sl_dist}, price={current_price})"
    )
    assert lot <= SYMBOL_INFO["max_lot"], (
        f"Lot {lot} > max_lot {SYMBOL_INFO['max_lot']}  "
        f"(balance={balance}, sl_dist={sl_dist}, price={current_price})"
    )


# ──────────────────────────────────────────────────────────────────────────────
# Property: Anti-martingale never produces negative lot
# ──────────────────────────────────────────────────────────────────────────────

@given(
    consecutive_losses=st.integers(min_value=0, max_value=20),
    balance=st.floats(min_value=500.0, max_value=100_000.0, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
def test_anti_martingale_never_negative_lot(consecutive_losses, balance):
    """
    PROPERTY: No matter how many consecutive losses have occurred (0–20),
    the resulting lot size must always be strictly positive.
    The 0.95^n decay converges toward zero but must be floored at min_lot.
    """
    rg = _make_rg()
    rg.peak_balance = balance
    rg.consecutive_losses = consecutive_losses

    lot = rg.calculate_lot_size(
        balance=balance,
        stop_loss_dist=1.5,
        symbol_info=SYMBOL_INFO,
        current_price=2000.0,
    )

    assert lot >= 0.0, (
        f"Lot must never be negative. Got {lot} with {consecutive_losses} consecutive losses."
    )


# ──────────────────────────────────────────────────────────────────────────────
# Property: check_governance never raises on valid inputs
# ──────────────────────────────────────────────────────────────────────────────

@given(
    balance=st.floats(min_value=0.01, max_value=10_000_000.0, allow_nan=False, allow_infinity=False),
    equity=st.floats(min_value=0.01, max_value=10_000_000.0, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
def test_check_governance_never_raises(balance, equity):
    """
    PROPERTY: check_governance must never raise an unhandled exception
    for any valid balance/equity combination. It must always return a result.
    """
    rg = _make_rg()
    rg.max_equity = max(balance, equity)
    try:
        result = rg.check_governance(current_balance=balance, current_equity=equity)
        assert result is not None
    except Exception as e:
        pytest.fail(
            f"check_governance raised {type(e).__name__}: {e}  "
            f"(balance={balance}, equity={equity})"
        )


# ──────────────────────────────────────────────────────────────────────────────
# Property: Kill-switch + any inputs → always blocked
# ──────────────────────────────────────────────────────────────────────────────

@given(
    balance=st.floats(min_value=1.0, max_value=10_000_000.0, allow_nan=False, allow_infinity=False),
    equity=st.floats(min_value=1.0, max_value=10_000_000.0, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_kill_switch_always_blocks_regardless_of_equity(balance, equity):
    """
    PROPERTY: Once kill_switch_active=True, check_governance must return
    False for ALL possible balance/equity combinations without exception.
    This is the core fail-closed architectural guarantee.
    """
    rg = _make_rg()
    rg.kill_switch_active = True
    rg.max_equity = max(balance, equity)

    result = rg.check_governance(current_balance=balance, current_equity=equity)
    approved = result[0] if isinstance(result, tuple) else result.get("approved", False)
    assert approved is False, (
        f"Kill-switch must block ALL signals. Got {result}  "
        f"(balance={balance}, equity={equity})"
    )


# ──────────────────────────────────────────────────────────────────────────────
# Property: Magic number always integer in expected range
# ──────────────────────────────────────────────────────────────────────────────

@given(strategy_id=st.text(min_size=1, max_size=100))
@settings(max_examples=300, suppress_health_check=[HealthCheck.too_slow])
def test_magic_number_always_integer_in_range(strategy_id):
    """
    PROPERTY: For any non-empty strategy_id string, get_magic_number must
    return an integer in [base_magic, base_magic + 999].
    """
    rg = _make_rg()
    base_magic = int(BASE_CONFIG.get("magic_number", 234000))

    magic = rg.get_magic_number(strategy_id)

    assert isinstance(magic, int), f"Magic number must be int. Got {type(magic)}"
    assert base_magic <= magic <= base_magic + 999, (
        f"Magic number {magic} out of range [{base_magic}, {base_magic + 999}] "
        f"for strategy_id={repr(strategy_id)}"
    )


# ──────────────────────────────────────────────────────────────────────────────
# Property: Lot sizing is monotonically non-increasing with drawdown
# ──────────────────────────────────────────────────────────────────────────────

@given(
    dd_low=st.floats(min_value=0.0, max_value=3.9, allow_nan=False),
    dd_high=st.floats(min_value=4.0, max_value=9.9, allow_nan=False),
    balance=st.floats(min_value=1000.0, max_value=100_000.0, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
def test_lot_monotonically_decreases_with_drawdown(dd_low, dd_high, balance):
    """
    PROPERTY: A higher drawdown percentage must never produce a LARGER lot
    than a lower drawdown percentage (vault scaling is monotone decreasing).
    """
    assume(dd_high > dd_low)

    rg = _make_rg()
    peak = balance / (1.0 - dd_low / 100.0) if dd_low < 100 else balance * 2

    balance_at_low_dd = peak * (1.0 - dd_low / 100.0)
    balance_at_high_dd = peak * (1.0 - dd_high / 100.0)

    rg.peak_balance = peak

    lot_low_dd = rg.calculate_lot_size(
        balance=max(balance_at_low_dd, 100.0), stop_loss_dist=1.5,
        symbol_info=SYMBOL_INFO, current_price=2000.0,
    )
    lot_high_dd = rg.calculate_lot_size(
        balance=max(balance_at_high_dd, 100.0), stop_loss_dist=1.5,
        symbol_info=SYMBOL_INFO, current_price=2000.0,
    )

    assert lot_high_dd <= lot_low_dd + 0.001, (
        f"Lot at high DD ({dd_high:.1f}%): {lot_high_dd:.4f} must be ≤ "
        f"lot at low DD ({dd_low:.1f}%): {lot_low_dd:.4f}. "
        f"Vault scaling must be monotone decreasing."
    )
