import sys
import os
sys.path.append(os.getcwd())

import unittest
from unittest.mock import MagicMock
from core.execution.order_manager import OrderManager

class TestLiveExecutionPath(unittest.TestCase):
    def test_connection_link(self):
        # 1. Mock Connection
        mock_conn = MagicMock()
        mock_conn.place_order.return_value = {"ticket": 1234567, "is_error": False}
        
        # 2. Instantiate with connection
        config = {"backtest": {"enabled": False}}
        om = OrderManager(config, mock_conn)
        
        # 3. Dummy Signal
        from core.common.types import TradeSignal
        sig = TradeSignal(direction="BUY", price=2300.0, volume=0.1, stop_loss=2290.0, take_profit=2320.0)
        
        # 4. Execute
        result = om.execute_signal(sig, "XAUUSDm", {"bid": 2300.0, "ask": 2300.5, "point": 0.01})
        
        # 5. Verify it CALLED the connection (Not simulated)
        self.assertEqual(result["ticket"], 1234567)
        mock_conn.place_order.assert_called_once()
        print("Success: OrderManager correctly identified live connection.")

if __name__ == "__main__":
    unittest.main()
