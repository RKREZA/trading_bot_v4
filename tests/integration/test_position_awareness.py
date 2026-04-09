import unittest
from unittest.mock import MagicMock, patch
import sys
import os

# Add project root to path
sys.path.append(os.getcwd())

# Ensure we mock mt5 before any project imports
sys.modules['MetaTrader5'] = MagicMock()

from core.connection import PositionManager, MT5Connection

class TestPositionAwareness(unittest.TestCase):
    def setUp(self):
        self.mock_conn = MagicMock(spec=MT5Connection)
        self.mock_conn.ensure_connected.return_value = True
        self.mock_conn.config = {"magic_number": 234000}
        self.mock_conn.MT5_LOCK = MagicMock()
        
        self.pos_manager = PositionManager(self.mock_conn)

    def test_magic_range_filtering(self):
        """Verify that PositionManager identifies positions in the [Base, Base+999] range."""
        # Mock MT5 positions
        mock_p1 = MagicMock(ticket=1001, magic=234000)
        mock_p2 = MagicMock(ticket=1002, magic=234567)
        mock_p3 = MagicMock(ticket=1003, magic=123456)
        mock_p4 = MagicMock(ticket=1004, magic=234999)
        
        # Patch the mt5 module used inside core.connection
        with patch('core.connection.mt5') as mock_mt5:
            mock_mt5.positions_get.return_value = (mock_p1, mock_p2, mock_p3, mock_p4)
            
            positions = self.pos_manager.get_open_positions()
            
            tickets = [p.ticket for p in positions]
            self.assertIn(1001, tickets, "Base magic position should be included.")
            self.assertIn(1002, tickets, "Hashed strategy magic position should be included.")
            self.assertIn(1004, tickets, "End-of-range magic position should be included.")
            self.assertNotIn(1003, tickets, "Rogue position (123456) should be EXCLUDED.")
            
            print("\n[SUCCESS] Position awareness range filter verified.")
            print(f"Captured Tickets: {tickets}")

if __name__ == "__main__":
    unittest.main()
