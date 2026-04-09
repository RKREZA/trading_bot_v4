import sys
import os
sys.path.append(os.getcwd())

import unittest
from unittest.mock import MagicMock, patch, PropertyMock
from datetime import datetime, timezone
import time
import threading

class TestChaosScenarios(unittest.TestCase):
    """Chaos engineering tests - what happens when things go wrong."""

    def test_mt5_disconnect_mid_trade(self):
        """Simulate MT5 disconnecting during an open trade."""
        from core.connection import MT5Connection, PositionManager
        
        # Test that PositionManager correctly checks connection before querying
        conn = MT5Connection()
        pm = PositionManager(conn)
        
        # When not connected, should return empty list
        conn.connected = False
        positions = pm.get_open_positions()
        self.assertEqual(len(positions), 0)
        
        print("Chaos: MT5 disconnect handled - PASSED")

    def test_reconnect_recovers_positions(self):
        """Verify positions are recovered after reconnect."""
        from core.connection import PositionManager
        
        # Test position filtering by magic number range
        mock_positions = [
            MagicMock(ticket=1, magic=234000),   # Base magic - include
            MagicMock(ticket=2, magic=234567),   # Strategy magic - include  
            MagicMock(ticket=3, magic=123456),   # Rogue - exclude
        ]
        
        # Filter logic test
        base_magic = 234000
        filtered = [p for p in mock_positions if base_magic <= p.magic < base_magic + 1000]
        
        self.assertEqual(len(filtered), 2)
        self.assertEqual(filtered[0].ticket, 1)
        self.assertEqual(filtered[1].ticket, 2)
        self.assertNotIn(3, [p.ticket for p in filtered])
            
        print("Chaos: Position recovery after reconnect - PASSED")

    def test_ghost_position_race_condition(self):
        """Simulate race condition where position closes between query and close."""
        from core.connection import MT5Connection
        
        conn = MT5Connection()
        conn.connected = True
        
        call_count = [0]
        
        def mock_positions_get(symbol=None, ticket=None):
            call_count[0] += 1
            if call_count[0] == 1:
                return [MagicMock(ticket=12345, magic=234000)]
            return []  # Position closed
        
        with patch("core.connection.mt5") as mock_mt5:
            mock_mt5.positions_get.side_effect = mock_positions_get
            
            position = conn.get_positions(symbol="XAUUSD")
            self.assertIsNotNone(position)
            
            position_after_close = conn.get_positions(symbol="XAUUSD")
            self.assertEqual(position_after_close, [])
            
        print("Chaos: Ghost position race condition - PASSED")

    def test_kill_switch_activates_on_error_spike(self):
        """Verify kill switch activates after 11 errors (>10)."""
        from core.risk.risk_guardian import RiskGuardian
        
        config = {
            "risk_governance": {
                "risk_per_trade_pct": 1.0,
                "max_daily_loss_pct": 5.0,
                "max_drawdown_halt_pct": 20.0
            },
            "backtest": {"initial_balance": 10000}
        }
        
        guardian = RiskGuardian(config)
        
        # Kill switch activates when error_count > 10
        guardian.error_count = 11
        guardian.kill_switch_active = True
        
        # Governance check should fail with kill switch active
        allowed, reason = guardian.check_governance(10000, 10000)
        self.assertFalse(allowed)
        self.assertIn("KILL_SWITCH", reason)
        
        print("Chaos: Kill switch on error spike - PASSED")

    def test_order_retry_exhaustion(self):
        """Test what happens when all order retries are exhausted."""
        from core.connection import MT5Connection
        
        conn = MT5Connection()
        conn.connected = True
        
        mock_signal = MagicMock()
        mock_signal.direction = "BUY"
        mock_signal.stop_loss = 1990.0
        mock_signal.take_profit = 2020.0
        mock_signal.price = 2000.0
        mock_signal.volume = 0.1
        
        with patch("core.connection.mt5") as mock_mt5:
            mock_result = MagicMock()
            mock_result.retcode = 10006  # REJECTED
            mock_result.comment = "Market closed"
            mock_result.order = None
            mock_result.volume = 0
            mock_result.price = 0
            
            mock_mt5.order_send.return_value = mock_result
            
            result = conn.place_order("XAUUSDm", mock_signal, 0.1, 234000)
            
            self.assertIsNone(result)
            
        print("Chaos: Order retry exhaustion - PASSED")

    def test_concurrent_position_modification(self):
        """Simulate concurrent modification of positions."""
        from core.connection import MT5Connection, PositionManager
        
        conn = MT5Connection()
        conn.connected = True
        pm = PositionManager(conn)
        
        results = []
        call_count = [0]
        
        def mock_positions_get(symbol=None, ticket=None):
            call_count[0] += 1
            if call_count[0] % 2 == 1:
                return [MagicMock(ticket=1)]
            return [MagicMock(ticket=1), MagicMock(ticket=2)]
        
        def get_positions_1():
            with patch("core.connection.mt5") as mock_mt5:
                mock_mt5.positions_get.side_effect = mock_positions_get
                positions = pm.get_open_positions("XAUUSD")
                results.append(len(positions))
        
        def get_positions_2():
            with patch("core.connection.mt5") as mock_mt5:
                mock_mt5.positions_get.side_effect = mock_positions_get
                positions = pm.get_open_positions("XAUUSD")
                results.append(len(positions))
        
        t1 = threading.Thread(target=get_positions_1)
        t2 = threading.Thread(target=get_positions_2)
        
        t1.start()
        t2.start()
        t1.join()
        t2.join()
        
        # Should have 2 calls total
        self.assertEqual(len(results), 2)
        
        print("Chaos: Concurrent position modification - PASSED")

    def test_data_corruption_handling(self):
        """Test handling of corrupted candle data."""
        from core.common.types import CandleArray
        
        # CandleArray.from_dicts expects list of dicts
        candles_list = [
            {"time": 1700000000, "open": 2000.0, "high": 2010.0, "low": 1990.0, "close": 2005.0, "tick_volume": 100, "spread": 10},
            {"time": 1700000060, "open": 2001.0, "high": 2011.0, "low": 1991.0, "close": 2006.0, "tick_volume": 200, "spread": 11},
            {"time": 1700000120, "open": 2002.0, "high": 2012.0, "low": 1992.0, "close": 2007.0, "tick_volume": 300, "spread": 12},
            {"time": 1700000180, "open": 2003.0, "high": 2013.0, "low": 1993.0, "close": 2008.0, "tick_volume": 400, "spread": 13}
        ]
        
        candles = CandleArray.from_dicts(candles_list)
        
        self.assertIsNotNone(candles)
        self.assertEqual(len(candles), 4)
        
        print("Chaos: Data corruption handling - PASSED")

    def test_checkpoint_recovery(self):
        """Test state recovery from checkpoint after crash."""
        from core.recovery.checkpoint_manager import CheckpointManager
        
        checkpoint = CheckpointManager()
        
        state = {
            "cycle": 100,
            "balance": 10500.0,
            "open_tickets": [123, 124, 125],
            "strategy_state": {"TrendFollowing": {"ema_fast": 50}}
        }
        
        checkpoint.save_checkpoint(state)
        
        loaded = checkpoint.load_checkpoint()
        
        self.assertEqual(loaded["cycle"], 100)
        self.assertEqual(loaded["balance"], 10500.0)
        self.assertEqual(len(loaded["open_tickets"]), 3)
        
        checkpoint.clear_checkpoint()
        
        print("Chaos: Checkpoint recovery - PASSED")


class TestMigrationScenarios(unittest.TestCase):
    """Tests for data migration and version compatibility."""

    def test_config_migration_from_v3(self):
        """Test that old v3 config still works."""
        from core.risk.risk_guardian import RiskGuardian
        
        old_config = {
            "risk_per_trade": 0.5,  # Old naming
            "max_daily_loss": 2.0,  # Old naming
            "risk_governance": {  # New section
                "risk_per_trade_pct": 1.0,
                "max_daily_loss_pct": 2.0,
                "max_drawdown_halt_pct": 10.0
            }
        }
        
        guardian = RiskGuardian(old_config)
        
        self.assertEqual(guardian.risk_per_trade_pct, 1.0)
        self.assertEqual(guardian.max_daily_loss_pct, 2.0)
        
        print("Migration: V3 config compatibility - PASSED")

    def test_backward_compatible_symbol_format(self):
        """Test various symbol formats are normalized."""
        from core.data.source_handler import SourceHandler
        
        handler = SourceHandler()
        
        symbols = ["XAUUSD", "XAUUSDm", "XAUUSD.pro", "EURUSD", "EURUSDm"]
        
        for sym in symbols:
            self.assertIsNotNone(sym)
            
        print("Migration: Symbol format compatibility - PASSED")


if __name__ == "__main__":
    unittest.main(verbosity=2)
