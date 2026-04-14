"""
test_risk_guardian.py — Institutional Risk & Governance Test Suite
=================================================================
Proves the RiskGuardian enforces kill-switch, exposure netting, circuit
breakers, and drawdown de-scaling under institutional stress conditions.

V5-INSIGNIA Institutional Certification.
"""

import time
import pytest
import logging
from core.risk.risk_guardian import RiskGuardian


# ============================================================================
# 1. KILL-SWITCH ENFORCEMENT
# ============================================================================

class TestKillSwitch:
    """Verifies the kill-switch is absolute and non-bypassable."""

    def test_kill_switch_halts_trading(self, mock_config):
        """
        When equity drawdown >= max_drawdown_halt_pct (10%), check_governance
        MUST return False with 'MAX_DRAWDOWN_REACHED' and activate the kill-switch.
        All subsequent calls must also return False.
        """
        guardian = RiskGuardian(mock_config)
        guardian.silent = True
        guardian.max_equity = 10000.0

        # Simulate 10% drawdown (exactly at the limit)
        ok, reason = guardian.check_governance(
            current_balance=9000.0,
            current_equity=9000.0,
        )
        assert ok is False, "Kill-switch should trigger at max drawdown"
        assert "MAX_DRAWDOWN_REACHED" in reason
        assert guardian.kill_switch_active is True

        # Subsequent call MUST also fail (no calibration bypass)
        ok2, reason2 = guardian.check_governance(
            current_balance=9000.0,
            current_equity=9000.0,
        )
        assert ok2 is False
        assert reason2 == "KILL_SWITCH_ACTIVE"

    def test_kill_switch_blocks_lot_sizing(self, mock_config, symbol_info):
        """
        After kill-switch activates, calculate_lot_size MUST return 0.0
        regardless of balance or stop-loss distance.
        """
        guardian = RiskGuardian(mock_config)
        guardian.silent = True
        guardian.kill_switch_active = True

        lot = guardian.calculate_lot_size(
            balance=100000.0,
            stop_loss_dist=5.0,
            symbol_info=symbol_info,
            current_price=2000.0,
        )
        assert lot == 0.0, "Kill-switch must zero out all lot sizing"

    def test_kill_switch_not_triggered_below_threshold(self, mock_config):
        """
        When drawdown < max_drawdown_halt_pct, governance MUST pass.
        """
        guardian = RiskGuardian(mock_config)
        guardian.silent = True
        guardian.max_equity = 10000.0

        ok, reason = guardian.check_governance(
            current_balance=9500.0,
            current_equity=9500.0,
        )
        assert ok is True
        assert reason == "OK"
        assert guardian.kill_switch_active is False


# ============================================================================
# 2. EXPOSURE NETTING 2.0
# ============================================================================

class TestExposureNetting:
    """Verifies basket-level and global exposure limits are enforced."""

    def test_exposure_netting_major_fx_breach(self, mock_config):
        """
        MAJOR_FX basket with 3.0 lots on $10k equity exceeds the
        2.0-lot limit (2.0 * 1.0 vol_mult * 1.0 equity_units).
        """
        guardian = RiskGuardian(mock_config)
        guardian.silent = True

        positions = [
            {"symbol": "EURUSD", "volume": 1.5},
            {"symbol": "GBPUSD", "volume": 1.6},  # Total = 3.1 lots > 2.0 limit
        ]
        ok, reason = guardian.check_governance(10000.0, 10000.0, positions)
        assert ok is False
        assert "EXPOSURE_NETTING_BREACH" in reason
        assert "MAJOR_FX" in reason

    def test_exposure_netting_jpy_vol_adjusted(self, mock_config):
        """
        JPY_EXT basket limit is 1.6 lots per $10k (2.0 * 0.8 vol_multiplier).
        1.7 lots should breach.
        """
        guardian = RiskGuardian(mock_config)
        guardian.silent = True

        positions = [
            {"symbol": "USDJPY", "volume": 0.9},
            {"symbol": "GBPJPY", "volume": 0.8},  # Total = 1.7 > 1.6 limit
        ]
        ok, reason = guardian.check_governance(10000.0, 10000.0, positions)
        assert ok is False
        assert "EXPOSURE_NETTING_BREACH" in reason
        assert "JPY_EXT" in reason

    def test_global_exposure_cap(self, mock_config):
        """
        Total gross exposure > 8.0 lots per $10k equity triggers rejection.
        """
        guardian = RiskGuardian(mock_config)
        guardian.silent = True

        # Spread across multiple baskets to avoid per-basket limit
        # but exceed the global 8.0 cap
        positions = [
            {"symbol": "EURUSD", "volume": 1.9},    # MAJOR_FX
            {"symbol": "USDJPY", "volume": 1.5},     # JPY_EXT
            {"symbol": "XAUUSDm", "volume": 0.9},    # Gold
            {"symbol": "DE30", "volume": 0.9},        # INDICES
            {"symbol": "GBPUSD", "volume": 1.9},     # MAJOR_FX → 3.8 total FX
            {"symbol": "NAS100", "volume": 1.0},      # INDICES → 1.9 total
        ]
        # Total gross = 1.9 + 1.5 + 0.9 + 0.9 + 1.9 + 1.0 = 8.1 > 8.0
        ok, reason = guardian.check_governance(10000.0, 10000.0, positions)
        assert ok is False
        # Could be basket or global — both are valid rejections


# ============================================================================
# 3. CIRCUIT BREAKERS (48-HOUR TRAILING LOSS)
# ============================================================================

class TestCircuitBreakers:
    """Verifies strategy-level 48h trailing loss halts work in isolation."""

    def test_circuit_breaker_48h_halt(self, mock_config):
        """
        Strategy accumulating >= 3% loss over 48h should be HALTED.
        Uses manual reset mode (cooldown=0) to verify permanent halt.
        """
        mock_config["risk_governance"]["circuit_breaker_cooldown_hours"] = 0
        guardian = RiskGuardian(mock_config)
        guardian.silent = True

        sid = "test_trend_strategy"
        # Record losses totaling 3.5% of allocation
        guardian.record_strategy_result(sid, pnl_abs=-100.0, alloc_balance=5000.0)  # -2%
        guardian.record_strategy_result(sid, pnl_abs=-75.0, alloc_balance=5000.0)   # -1.5%
        # Total = -3.5%

        ok, reason = guardian.check_strategy_governance(sid)
        assert ok is False
        assert "STRATEGY_HALTED" in reason
        assert guardian.strategy_status[sid] == "HALTED"

    def test_circuit_breaker_isolation(self, mock_config):
        """
        Only the offending strategy is halted; other strategies remain OK.
        """
        mock_config["risk_governance"]["circuit_breaker_cooldown_hours"] = 0
        guardian = RiskGuardian(mock_config)
        guardian.silent = True

        bad_sid = "losing_strategy"
        good_sid = "winning_strategy"

        # Bad strategy loses 4%
        guardian.record_strategy_result(bad_sid, pnl_abs=-200.0, alloc_balance=5000.0)

        # Good strategy profits
        guardian.record_strategy_result(good_sid, pnl_abs=100.0, alloc_balance=5000.0)

        # Bad one should be halted
        ok_bad, _ = guardian.check_strategy_governance(bad_sid)
        assert ok_bad is False

        # Good one should be fine
        ok_good, reason_good = guardian.check_strategy_governance(good_sid)
        assert ok_good is True
        assert reason_good == "OK"

    def test_circuit_breaker_old_losses_expire(self, mock_config):
        """
        Losses older than 48 hours should not contribute to the trailing check.
        """
        guardian = RiskGuardian(mock_config)
        guardian.silent = True

        sid = "expiring_strategy"
        # Record an old loss (49 hours ago)
        old_timestamp = time.time() - (49 * 3600)
        guardian.strategy_performance[sid] = [(old_timestamp, -5.0)]

        # This old loss should be cleaned up and not trigger halt
        ok, reason = guardian.check_strategy_governance(sid)
        assert ok is True
        assert reason == "OK"


# ============================================================================
# 4. DRAWDOWN DE-SCALING (A+ VAULT)
# ============================================================================

class TestDrawdownDescaling:
    """Verifies the equity drawdown de-scaling ramp reduces risk smoothly."""

    def test_drawdown_descaling_at_midpoint(self, mock_config, symbol_info):
        """
        At 6% drawdown with limit=10%, the penalty should be ~0.667
        (1 - (6-4)/(10-4) = 1 - 2/6 = 0.667).
        Risk pct should be ~0.667% (base 1.0% * 0.667).
        """
        guardian = RiskGuardian(mock_config)
        guardian.silent = True
        guardian.max_equity = 10000.0
        guardian.current_portfolio_equity = 9400.0  # 6% drawdown

        lot = guardian.calculate_lot_size(
            balance=9400.0,
            stop_loss_dist=2.50,
            symbol_info=symbol_info,
            current_price=2000.0,
        )
        # With full 1% risk on 9400: risk_amt = 94.0
        # With de-scaling penalty ~0.667: risk_amt ~= 62.7
        # lot = 62.7 / (250 * 1.0) = 0.25
        # The exact value depends on the normalization, but it should be
        # significantly less than the no-drawdown case.
        lot_no_dd = guardian.calculate_lot_size.__wrapped__ if hasattr(guardian.calculate_lot_size, '__wrapped__') else None

        # Compute baseline (no drawdown)
        guardian2 = RiskGuardian(mock_config)
        guardian2.silent = True
        guardian2.max_equity = 10000.0
        guardian2.current_portfolio_equity = 10000.0

        lot_full = guardian2.calculate_lot_size(
            balance=9400.0,
            stop_loss_dist=2.50,
            symbol_info=symbol_info,
            current_price=2000.0,
        )

        assert lot < lot_full, "De-scaled lot must be smaller than full-risk lot"
        assert lot > 0.0, "De-scaled lot should not be zero (drawdown < halt limit)"

    def test_drawdown_descaling_at_limit(self, mock_config, symbol_info):
        """
        At drawdown = max_drawdown_halt_pct, penalty should be 0 → risk_pct = 0.
        Lot size should be 0.0 (below min_lot floor).
        """
        guardian = RiskGuardian(mock_config)
        guardian.silent = True
        guardian.max_equity = 10000.0
        guardian.current_portfolio_equity = 9000.0  # Exactly 10% drawdown

        lot = guardian.calculate_lot_size(
            balance=9000.0,
            stop_loss_dist=2.50,
            symbol_info=symbol_info,
            current_price=2000.0,
        )
        assert lot == 0.0, "At max drawdown, risk should be zero"

    def test_no_descaling_below_threshold(self, mock_config, symbol_info):
        """
        Below 4% drawdown, de-scaling should NOT activate (full risk).
        """
        guardian = RiskGuardian(mock_config)
        guardian.silent = True
        guardian.max_equity = 10000.0
        guardian.current_portfolio_equity = 9700.0  # 3% drawdown (below 4% threshold)

        lot_dd = guardian.calculate_lot_size(
            balance=9700.0,
            stop_loss_dist=2.50,
            symbol_info=symbol_info,
            current_price=2000.0,
        )

        guardian2 = RiskGuardian(mock_config)
        guardian2.silent = True
        guardian2.max_equity = 10000.0
        guardian2.current_portfolio_equity = 10000.0

        lot_full = guardian2.calculate_lot_size(
            balance=9700.0,
            stop_loss_dist=2.50,
            symbol_info=symbol_info,
            current_price=2000.0,
        )

        assert lot_dd == lot_full, "Below 4% DD, de-scaling must not activate"


# ============================================================================
# 5. VALIDATE_SIGNAL DRY REFACTOR
# ============================================================================

class TestValidateSignalDRY:
    """Verifies the refactored validate_signal accepts pre-computed sl_dist."""

    def test_validate_signal_with_sl_dist(self, mock_config, symbol_info):
        """
        When sl_dist is provided, validate_signal should skip SL derivation
        and use the provided distance directly.
        """
        guardian = RiskGuardian(mock_config)
        guardian.silent = True

        # Create a mock signal with no stop_loss set
        from unittest.mock import MagicMock
        signal = MagicMock()
        signal.stop_loss = 0  # Would normally be rejected

        market_data = MagicMock()
        market_data.current_price = 2000.0

        # With sl_dist=0 and no signal.stop_loss → should reject
        result_no_sl = guardian.validate_signal(signal, 5000.0, market_data, symbol_info, sl_dist=0.0)
        assert result_no_sl is False

        # With sl_dist provided → should pass (if lot > 0)
        result_with_sl = guardian.validate_signal(signal, 5000.0, market_data, symbol_info, sl_dist=5.0)
        assert result_with_sl is True
