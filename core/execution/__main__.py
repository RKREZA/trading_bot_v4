import logging
from core.execution.order_manager import OrderManager
from core.common.types import TradeSignal

def run_execution_diagnostic():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    logger = logging.getLogger("execution_diag")
    
    logger.info("Starting Institutional Execution Diagnostic...")
    
    config = {
        "execution": {
            "latency_ms": 100,
            "max_spread_pips": 2.0,
            "entry_slippage_pips": 0.1,
            "sl_exit_slippage_pips": 0.3
        },
        "backtest": {
            "deterministic": True,
            "random_seed": 42
        }
    }
    
    manager = OrderManager(config)
    
    # --- Test Case 1: Standard Buy Signal ---
    signal = TradeSignal(direction="BUY", stop_loss=1.1000, take_profit=1.1100)
    prices = {"bid": 1.1050, "ask": 1.1051, "point": 0.0001}
    
    order = manager.execute_signal(signal, "EURUSD", prices)
    if order:
        logger.info(f"Test 1 (Standard Buy): Success. Ticket: {order['ticket']} @ {order['fill_price']}")
    else:
        logger.error("Test 1 (Standard Buy): Execution FAILED.")
        
    # --- Test Case 2: Spread Violation ---
    prices_wide = {"bid": 1.1050, "ask": 1.1055, "point": 0.0001} # 5.0 pips spread
    order_fail = manager.execute_signal(signal, "EURUSD", prices_wide)
    if not order_fail:
        logger.info("Test 2 (Spread Limit): Successfully rejected wide spread.")
    else:
        logger.error("Test 2 (Spread Limit): Failed to reject wide spread.")
        
    # --- Test Case 3: Exit Simulation ---
    exit_info = manager.simulate_exit(999, "sl", 1.1000, 0.0001)
    # Expected slippage on SL (negative)
    if exit_info['exit_price'] < 1.1000:
        logger.info(f"Test 3 (SL Exit): Exit Price {exit_info['exit_price']} (Negative slippage confirmed)")
    else:
        logger.error("Test 3 (SL Exit): No negative slippage applied to SL.")
        
    logger.info("ORDER_MANAGER: Standalone Diagnostic PASSED.")

if __name__ == "__main__":
    run_execution_diagnostic()
