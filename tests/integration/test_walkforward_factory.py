import sys
import os
sys.path.append(os.getcwd())

import unittest
from unittest.mock import MagicMock, patch
from datetime import datetime, timezone
from strategies.trend_following import TrendFollowingStrategy
from strategies.liquidity_session import LiquiditySessionStrategy

class TestWalkForwardFactoryPattern(unittest.TestCase):
    """Tests for walk-forward optimization factory pattern."""

    def test_factory_creates_fresh_instance(self):
        """Verify factory pattern creates strategy with clean state."""
        config = {
            "TrendFollowing": {
                "enabled": True,
                "adx_threshold": 20,
                "sl_atr": 2.0,
                "rr_target": 2.0
            }
        }
        
        original = TrendFollowingStrategy("trendfollowing_v4", config)
        original.ema_fast = 100  # Modify internal state
        
        fresh = original.__class__(original.strategy_id, original.config)
        
        self.assertNotEqual(fresh.ema_fast, 100)
        self.assertEqual(fresh.ema_fast, 50)  # Default value
        print("Test: Factory creates fresh instance - PASSED")

    def test_factory_preserves_config(self):
        """Verify factory preserves configuration parameters."""
        config = {
            "LiquiditySession": {
                "enabled": True,
                "range_maturity_limit": 5.0,
                "vol_trigger_mult": 0.8,
                "sl_atr": 2.0,
                "rr_target": 6.0
            }
        }
        
        original = LiquiditySessionStrategy("liquiditysession_v4", config)
        original.asian_high = 2000.0  # Simulate running state
        original.range_set = True
        
        fresh = original.__class__(original.strategy_id, original.config)
        
        self.assertEqual(fresh.range_maturity_limit, 5.0)
        self.assertEqual(fresh.vol_trigger_mult, 0.8)
        self.assertEqual(fresh.asian_high, 0.0)  # Reset to default
        self.assertEqual(fresh.range_set, False)  # Reset to default
        print("Test: Factory preserves config, resets state - PASSED")

    def test_factory_no_cross_contamination(self):
        """Verify multiple fresh instances don't share state."""
        config = {
            "TrendFollowing": {"enabled": True}
        }
        
        strat1 = TrendFollowingStrategy("trendfollowing_v4", config)
        strat2 = TrendFollowingStrategy("trendfollowing_v4", config)
        
        strat1.ema_fast = 999
        
        self.assertNotEqual(strat1.ema_fast, strat2.ema_fast)
        self.assertEqual(strat2.ema_fast, 50)  # Original default
        print("Test: No cross-contamination between instances - PASSED")

    def test_factory_applies_config_overrides(self):
        """Verify config overrides are applied correctly."""
        config = {
            "TrendFollowing": {
                "enabled": True,
                "ema_fast": 20,
                "ema_slow": 100
            }
        }
        
        strat = TrendFollowingStrategy("trendfollowing_v4", config)
        
        self.assertEqual(strat.ema_fast, 20)
        self.assertEqual(strat.ema_slow, 100)
        print("Test: Config overrides applied - PASSED")


class TestStrategyStateReset(unittest.TestCase):
    """Tests to verify strategies reset properly between backtest windows."""

    def test_liquidity_session_resets_daily(self):
        """Verify LiquiditySessionStrategy.reset_daily_stats() works."""
        config = {"LiquiditySession": {"enabled": True}}
        strat = LiquiditySessionStrategy("liquiditysession_v4", config)
        
        strat.asian_high = 1950.0
        strat.asian_low = 1900.0
        strat.range_set = True
        strat.london_trade_taken = True
        strat.ny_trade_taken = True
        
        strat.reset_daily_stats()
        
        self.assertEqual(strat.asian_high, 0.0)
        self.assertEqual(strat.asian_low, 0.0)
        self.assertEqual(strat.range_set, False)
        self.assertEqual(strat.london_trade_taken, False)
        self.assertEqual(strat.ny_trade_taken, False)
        print("Test: LiquiditySession daily reset - PASSED")


if __name__ == "__main__":
    unittest.main(verbosity=2)
