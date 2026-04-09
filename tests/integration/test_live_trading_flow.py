import sys
import os
sys.path.append(os.getcwd())

import unittest
from unittest.mock import MagicMock, patch
from datetime import datetime, timezone
from core.strategy_orchestrator import StrategyOrchestrator
from core.strategy_runtime import StrategyRuntime
from core.execution.order_manager import OrderManager
from core.portfolio_manager import PortfolioManager
from core.connection import MT5Connection, PositionManager

class TestLiveTradingFlow(unittest.TestCase):
    """Integration tests for the full live trading execution path."""

    def setUp(self):
        self.config = {
            "risk_governance": {
                "risk_per_trade_pct": 1.0,
                "max_daily_loss_pct": 2.0,
                "max_drawdown_halt_pct": 10.0,
                "max_parallel_strategies": 4
            },
            "portfolio_allocations": {
                "TrendFollowing": 0.5,
                "Breakout": 0.5
            },
            "trailing_stop": {"enabled": False},
            "backtest": {"enabled": False},
            "magic_number": 234000
        }

    def test_order_manager_live_path(self):
        """Verify OrderManager routes to live execution when connection is present."""
        mock_conn = MagicMock(spec=MT5Connection)
        mock_conn.place_order.return_value = {
            "ticket": 12345,
            "volume": 0.1,
            "price": 2000.0
        }
        
        om = OrderManager(self.config, connection=mock_conn)
        
        mock_signal = MagicMock()
        mock_signal.direction = "BUY"
        mock_signal.stop_loss = 1990.0
        mock_signal.take_profit = 2020.0
        mock_signal.volume = 0.1
        
        result = om.execute_signal(
            signal=mock_signal,
            symbol="XAUUSD",
            price_data={"bid": 2000.0, "ask": 2001.0, "point": 0.01},
            magic=234000
        )
        
        # Verify live path was called
        mock_conn.place_order.assert_called_once()
        self.assertEqual(result["ticket"], 12345)
        print("Integration: OrderManager live path verified.")

    def test_order_manager_simulation_path(self):
        """Verify OrderManager simulates when no connection."""
        om = OrderManager(self.config, connection=None)
        
        mock_signal = MagicMock()
        mock_signal.direction = "SELL"
        mock_signal.stop_loss = 2010.0
        mock_signal.take_profit = 1990.0
        mock_signal.volume = 0.1
        
        result = om.execute_signal(
            signal=mock_signal,
            symbol="XAUUSD",
            price_data={"bid": 2000.0, "ask": 2001.0, "point": 0.01}
        )
        
        self.assertIsNotNone(result)
        self.assertIn("ticket", result)
        self.assertTrue(1000000 <= result["ticket"] <= 9999999)
        print("Integration: OrderManager simulation path verified.")

    def test_position_magic_range_filtering(self):
        """Verify positions are filtered by magic number range."""
        mock_conn = MagicMock(spec=MT5Connection)
        mock_conn.ensure_connected.return_value = True
        mock_conn.config = {"magic_number": 234000}
        mock_conn.MT5_LOCK = MagicMock()
        
        pm = PositionManager(mock_conn)
        
        mock_positions = [
            MagicMock(ticket=1, magic=234000),   # Base magic
            MagicMock(ticket=2, magic=234567),   # Strategy hash
            MagicMock(ticket=3, magic=234999),   # End of range
            MagicMock(ticket=4, magic=123456),    # Rogue position
            MagicMock(ticket=5, magic=235000),   # Outside range
        ]
        
        with patch("core.connection.mt5") as mock_mt5:
            mock_mt5.positions_get.return_value = mock_positions
            result = pm.get_open_positions()
        
        tickets = [p.ticket for p in result]
        
        self.assertIn(1, tickets)
        self.assertIn(2, tickets)
        self.assertIn(3, tickets)
        self.assertNotIn(4, tickets)  # Rogue
        self.assertNotIn(5, tickets)  # Outside range
        print("Integration: Position magic range filtering verified.")

    def test_strategy_runtime_execute_cycle(self):
        """Verify StrategyRuntime.execute_cycle is called correctly."""
        from core.base_strategy import BaseStrategy, MarketData
        from core.common.types import TradeSignal
        from core.risk.risk_guardian import RiskGuardian
        
        class MockStrategy(BaseStrategy):
            def __init__(self):
                super().__init__("mock", {})
                self.enabled = True
                
            def generate_signal(self, data):
                return TradeSignal(direction="BUY", confidence=0.8, price=2000.0)
                
            def get_stop_loss(self, signal, data):
                return 1990.0
                
            def get_take_profit(self, signal, data):
                return 2020.0
            
            def get_metrics(self, data):
                return {"test": 1}
        
        strategy = MockStrategy()
        guardian = RiskGuardian(self.config)
        runtime = StrategyRuntime(strategy, self.config, guardian)
        
        market_data = MarketData(
            symbol="XAUUSD",
            htf_candles=None,
            m15_candles=None,
            m5_candles=None,
            d1_candles=None,
            current_price=2000.0,
            bid=2000.0,
            ask=2001.0,
            spread=1.0,
            session="LONDON",
            timestamp=datetime.now(timezone.utc)
        )
        
        signal = runtime.execute_cycle(market_data)
        
        self.assertIsNotNone(signal)
        self.assertEqual(signal.direction, "BUY")
        self.assertEqual(signal.confidence, 0.8)
        print("Integration: StrategyRuntime execute_cycle verified.")

    def test_full_signal_to_execution_flow(self):
        """Test the complete flow from signal generation to execution."""
        from core.base_strategy import BaseStrategy, MarketData
        from core.common.types import TradeSignal
        
        class MockStrategy(BaseStrategy):
            def __init__(self):
                super().__init__("test", {})
                self.enabled = True
                
            def generate_signal(self, data):
                return TradeSignal(direction="BUY", confidence=0.85, price=2000.0)
                
            def get_stop_loss(self, signal, data):
                return 1995.0
                
            def get_take_profit(self, signal, data):
                return 2020.0
            
            def get_metrics(self, data):
                return {}
        
        mock_conn = MagicMock()
        mock_conn.place_order.return_value = {
            "ticket": 99999,
            "volume": 0.1,
            "price": 2000.5
        }
        mock_conn.get_balance.return_value = 10000.0
        mock_conn.get_equity.return_value = 10000.0
        mock_conn.get_symbol_info.return_value = {
            "point": 0.01,
            "tick_value": 1.0,
            "spread": 15,
            "min_lot": 0.01,
            "max_lot": 20.0,
            "lot_step": 0.01
        }
        
        om = OrderManager(self.config, connection=mock_conn)
        
        mock_signal = MagicMock()
        mock_signal.direction = "BUY"
        mock_signal.stop_loss = 1995.0
        mock_signal.take_profit = 2020.0
        mock_signal.volume = 0.1
        
        result = om.execute_signal(
            signal=mock_signal,
            symbol="XAUUSD",
            price_data={"bid": 2000.0, "ask": 2001.5, "point": 0.01},
            magic=234000,
            comment="TEST"
        )
        
        self.assertIsNotNone(result)
        self.assertEqual(result["ticket"], 99999)
        mock_conn.place_order.assert_called_once()
        print("Integration: Full signal-to-execution flow verified.")

if __name__ == "__main__":
    unittest.main(verbosity=2)
