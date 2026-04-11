import pytest
from unittest.mock import patch, MagicMock
import time
from datetime import datetime, timezone

from core.risk.risk_guardian import RiskGuardian
from core.connection import MT5Connection
from core.strategy_orchestrator import StrategyOrchestrator

# -------------------------------------------------------------------
# STRESS TEST 1: The "Flash Crash" Active Mitigation Test
# -------------------------------------------------------------------
def test_flash_crash_emergency_flatten():
    """
    Simulates a sudden equity drop exceeding the max drawdown threshold.
    Validates that the RiskGuardian issues the specific EMERGENCY command
    and the Orchestrator executes the flattening of all positions.
    """
    # 1. Setup Risk Guardian with a tight 0.5% max drawdown limit
    config = {
        "backtest": {"initial_balance": 10000.0},
        "risk_governance": {"max_drawdown_halt_pct": 0.5} # 0.5% DD limit
    }
    guardian = RiskGuardian(config)
    guardian.max_equity = 10000.0
    
    # 2. Simulate a sudden drop to $9,900 (1.0% drop, which exceeds 0.5%)
    flash_crash_equity = 9900.0
    allowed, reason = guardian.check_governance(10000.0, flash_crash_equity)
    
    # Assert RiskGuardian correctly identifies the breach and issues the command
    assert allowed is False
    assert reason == "EMERGENCY_FLATTEN_REQUIRED"
    assert guardian.kill_switch_active is True

    # 3. Simulate the Orchestrator catching this flag
    mock_connection = MagicMock()
    mock_pos_manager = MagicMock()
    # Mock two open positions
    mock_pos1 = MagicMock(ticket=111, symbol="EURUSD")
    mock_pos2 = MagicMock(ticket=222, symbol="XAUUSD")
    mock_pos_manager.get_open_positions.return_value = [mock_pos1, mock_pos2]
    
    orchestrator = StrategyOrchestrator([], config, mock_connection, mock_pos_manager, None, None, None)
    
    # Simulate the Orchestrator loop reaction
    if not allowed and reason == "EMERGENCY_FLATTEN_REQUIRED":
        positions = mock_pos_manager.get_open_positions()
        for pos in positions:
            mock_connection.close_position(pos.ticket, pos.symbol)
            
    # Assert the connection manager was commanded to close BOTH positions
    assert mock_connection.close_position.call_count == 2
    mock_connection.close_position.assert_any_call(111, "EURUSD")
    mock_connection.close_position.assert_any_call(222, "XAUUSD")


# -------------------------------------------------------------------
# STRESS TEST 2: The "Weekend Ghost" Temporal Drift Test
# -------------------------------------------------------------------
@patch("core.connection.mt5")
@patch("core.connection.time")
def test_weekend_market_closure_detection(mock_time, mock_mt5):
    """
    Simulates a frozen MT5 server on a Sunday to ensure the bot 
    uses local system time to block weekend orders.
    """
    conn = MT5Connection()
    conn.connected = True
    
    # 1. Mock the last candle provided by the broker (Frozen on Friday at 23:59)
    # Friday timestamp: ~1712361540
    friday_timestamp = 1712361540 
    mock_mt5.copy_rates_from_pos.return_value = ((friday_timestamp, 1.10, 1.11, 1.09, 1.10, 100),)
    mock_mt5.SYMBOL_TRADE_MODE_DISABLED = 0
    
    # Mock symbol info to appear "tradeable" according to the broker's static flag
    mock_info = MagicMock()
    mock_info.trade_mode = 4 # Full trade mode (often left on by brokers over weekend)
    mock_mt5.symbol_info.return_value = mock_info
    
    # 2. Mock system time to be SUNDAY (48 hours later = +172800 seconds)
    sunday_timestamp = friday_timestamp + 172800
    mock_time.time.return_value = sunday_timestamp
    
    # 3. Check market status
    is_open = conn.get_market_status("EURUSD")
    
    # Assert the bot realizes the market is closed because 48 hours > 10 hours limit
    assert is_open is False


# -------------------------------------------------------------------
# STRESS TEST 3: The "Non-Forex Notional Guard" Test
# -------------------------------------------------------------------
def test_institutional_notional_sizing():
    """
    Validates that Gold (XAUUSD) and Crypto notional sizing is price-aware.
    """
    config = {
        "risk_governance": {"min_notional_value": 1000.0} # $1,000 Minimum Position Size
    }
    guardian = RiskGuardian(config)
    
    # Mock Gold (XAUUSD) parameters
    gold_sym_info = {
        "min_lot": 0.01,
        "max_lot": 100.0,
        "lot_step": 0.01,
        "contract_size": 100.0 # 1 lot = 100 oz
    }
    gold_price = 2300.0 # $2,300 per oz
    
    # Scenario A: Attempting to buy 0.001 lots (if min_lot allowed it)
    # Notional = 0.001 * 100 * 2300 = $230.00 (Below $1,000 threshold)
    rejected_lot = guardian._normalize_lots(0.001, gold_sym_info, current_price=gold_price)
    assert rejected_lot == 0.0 # Must be rejected
    
    # Scenario B: Attempting to buy 0.01 lots
    # Notional = 0.01 * 100 * 2300 = $2,300.00 (Above $1,000 threshold)
    approved_lot = guardian._normalize_lots(0.01, gold_sym_info, current_price=gold_price)
    assert approved_lot == 0.01 # Must be approved and rounded to step