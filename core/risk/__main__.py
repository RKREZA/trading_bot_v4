import logging
from core.risk.risk_guardian import RiskGuardian

def run_diagnostic():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    logger = logging.getLogger("risk_diag")
    
    logger.info("Starting Institutional Risk Diagnostic...")
    
    config = {
        "risk_governance": {
            "risk_per_trade_pct": 0.5,
            "max_daily_loss_pct": 2.0,
            "max_drawdown_halt_pct": 10.0,
            "max_consecutive_losses": 4
        },
        "backtest": {
            "initial_balance": 10000.0
        }
    }
    
    guardian = RiskGuardian(config)
    
    # --- Test Case 1: Standard Sizing ---
    balance = 10000
    sl_dist = 0.00500 # 50 pips on EURUSD
    sym_info = {
        "point": 0.00001,
        "tick_value": 1.0,
        "spread_pips": 10,
        "commission_per_lot": 7,
        "min_lot": 0.01,
        "lot_step": 0.01
    }
    
    lots = guardian.calculate_lot_size(balance, sl_dist, sym_info)
    expected_risk = balance * 0.005 # $50
    logger.info(f"Test 1 (Standard): Lots={lots} (Expected Risk: ${expected_risk})")
    
    # --- Test Case 2: Circuit Breaker Breach ---
    guardian.record_trade_result(-250) # Simulate a loss
    is_ok, reason = guardian.check_governance(10000, 9750)
    logger.info(f"Test 2 (DD Check): OK={is_ok}, Reason={reason}")
    
    # --- Test Case 3: Scaling Reduction ---
    guardian.consecutive_losses = 3
    lots_scaled = guardian.calculate_lot_size(balance, sl_dist, sym_info)
    logger.info(f"Test 3 (Scaling): Lots={lots_scaled} (Should be half of {lots})")
    
    if lots_scaled < lots:
        logger.info("RISK_GUARDIAN: Standalone Diagnostic PASSED.")
    else:
        logger.error("RISK_GUARDIAN: Standalone Diagnostic FAILED.")

if __name__ == "__main__":
    run_diagnostic()
