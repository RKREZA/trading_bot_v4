import time
import logging
from unittest.mock import MagicMock
from core.strategy_orchestrator import StrategyOrchestrator
from core.common.types import MarketRegime, VolatilityStatus

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("stress_test")

class MockPosition:
    def __init__(self, ticket, symbol, price_open, sl, tp, type):
        self.ticket = ticket
        self.symbol = symbol
        self.price_open = price_open
        self.sl = sl
        self.tp = tp
        self.type = type # 0 for BUY, 1 for SELL

def run_stress_test():
    logger.info("Starting Institutional Stress Test: Connection Recovery Logic")
    
    # 1. Setup Mock environment
    config = {
        "magic_number": 234000,
        "trailing_stop": {
            "enabled": True,
            "phase1_rr_threshold": 1.5,
            "phase2_be_offset_pct": 0.1,
            "phase3_trail_mult": 1.5
        },
        "symbols_config": {
            "XAUUSDm": {"min_lot": 0.01, "max_lot": 20.0, "lot_step": 0.01, "point": 0.01}
        }
    }
    
    connection_mock = MagicMock()
    position_manager_mock = MagicMock()
    
    # Simulate an open position
    # Entry: 2000.00, SL: 1990.00 (Risk: 10.0), TP: 2050.0
    mock_pos = MockPosition(ticket=12345, symbol="XAUUSDm", price_open=2000.00, sl=1990.00, tp=2050.0, type=0)
    position_manager_mock.get_open_positions.return_value = [mock_pos]
    
    orchestrator = StrategyOrchestrator(
        runtimes=[], 
        config=config, 
        connection=connection_mock,
        position_manager=position_manager_mock,
        notification_manager=None,
        broker_clock=MagicMock()
    )
    
    # 2. Simulate Connection Drop during trailing stop update
    logger.info("Simulating price move to 2017.0 (1.7R profit)...")
    # Current RR = 17 / 10 = 1.7 > 1.5 (Threshold)
    # Expected behavior: Move SL to BE + offset (2000.0 + 1.0 = 2001.0)
    
    # Mock connection failure on first attempt
    connection_mock.modify_sl_tp.side_effect = [False, True] # First fails, second succeeds (after recovery)
    
    # Call trailing stop logic
    # manage_trailing_stops(self, symbol, bid, ask, atr, last_candle, session)
    orchestrator.manage_trailing_stops("XAUUSDm", 2017.0, 2017.1, 5.0, None, "GLOBAL")
    
    # 3. Verify interaction
    logger.info("Verifying modify_sl_tp calls...")
    assert connection_mock.modify_sl_tp.call_count >= 1
    
    # 4. Simulate Background Thread Recovery
    logger.info("Verifying recovery in next cycle...")
    # The background thread should keep trying until it succeeds
    # (In this mock, we just call it again)
    orchestrator.manage_trailing_stops("XAUUSDm", 2017.0, 2017.1, 5.0, None, "GLOBAL")
    
    # Check if the second call (which returns True) was made
    if connection_mock.modify_sl_tp.call_count == 2:
        logger.info("SUCCESS: System attempted recovery after initial failure.")
    else:
        logger.error(f"FAILURE: Unexpected call count: {connection_mock.modify_sl_tp.call_count}")

if __name__ == "__main__":
    run_stress_test()
