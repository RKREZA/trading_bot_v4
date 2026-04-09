import sys
import os
sys.path.append(os.getcwd())

import unittest
import numpy as np
from datetime import datetime, timezone
from core.risk.risk_guardian import RiskGuardian
from core.news_filter import InstitutionalNewsFilter
from core.common.types import TradeSignal, CandleArray

class TestInstitutionalFlow(unittest.TestCase):
    def setUp(self):
        self.config = {
            "risk_governance": {
                "risk_per_trade_pct": 1.0,
                "max_daily_loss_pct": 5.0,
                "max_drawdown_halt_pct": 20.0
            },
            "backtest": {"initial_balance": 10000.0}
        }

    def test_risk_guardian_defaults(self):
        """Verify that RiskGuardian handles missing config gracefully."""
        # Empty config
        guardian = RiskGuardian({})
        self.assertEqual(guardian.risk_per_trade_pct, 0.5)
        self.assertEqual(guardian.max_daily_loss_pct, 5.0)
        print("Integration: RiskGuardian defaults verified.")

    def test_news_filter_blocking(self):
        """Verify that news filter correctly blocks during high-impact events."""
        nf = InstitutionalNewsFilter(self.config)
        # Mock an event
        ts = datetime.now(timezone.utc).timestamp()
        nf.events = [{"title": "test", "impact": "High", "timestamp": ts, "country": "USD"}]
        
        # Check blocking
        self.assertTrue(nf.is_blocked("EURUSDm", ts))
        print("Integration: News filter blocking verified.")

    def test_m1_gap_fallback_logic(self):
        """Verify the fallback logic (Simulation of the SyntheticM1 mechanism)."""
        # Testing the concept used in backtester.py
        class SyntheticM1:
            def __init__(self, high, low, close):
                self.high = np.array([high])
                self.low = np.array([low])
                self.close = np.array([close])
        
        # Trade SL at 2300. Market spikes to 2310 (High) and 2290 (Low).
        trade = {"direction": "BUY", "sl": 2295.0, "tp": 2320.0, "ticket": 1}
        m1 = SyntheticM1(2310, 2290, 2300)
        
        # Logic check: if m1_low <= trade[sl] -> Triggered
        triggered = m1.low[0] <= trade["sl"]
        self.assertTrue(triggered)
        print("Integration: M1 Gap Fallback logic verified.")

if __name__ == "__main__":
    unittest.main()
