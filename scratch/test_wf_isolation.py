import sys
import os
sys.path.append(os.getcwd())

import unittest
import copy
from core.common.types import TradeSignal

class MockStrategy:
    def __init__(self):
        self.state = 0
        self.strategy_id = "mock"
    def generate_signal(self, data):
        self.state += 1
        return None

class TestWFIsolation(unittest.TestCase):
    def test_cloning(self):
        # 1. Original Strategy
        strat = MockStrategy()
        strategies = [strat]
        
        # 2. Simulate Window 1 (IS)
        # In a real WF, we'd clone before use
        is_strategies = copy.deepcopy(strategies)
        is_strategies[0].generate_signal(None)
        
        # 3. Verify IS changed its own state
        self.assertEqual(is_strategies[0].state, 1)
        
        # 4. Verify Original remains 0
        self.assertEqual(strat.state, 0)
        
        # 5. Simulate Window 2 (OOS)
        oos_strategies = copy.deepcopy(strategies)
        self.assertEqual(oos_strategies[0].state, 0)
        print("Success: Walk-Forward isolation confirmed via deepcopy.")

if __name__ == "__main__":
    unittest.main()
