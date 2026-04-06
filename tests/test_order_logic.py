import pytest
from unittest.mock import MagicMock, patch
from core.execution_engine import ExecutionEngine
from core import TradeSignal

class TestOrderLogic:
    """Verifies the integrity of the Execution Layer."""

    @pytest.fixture
    def execution_engine(self, mock_config):
        return ExecutionEngine(mock_config)

    def test_execute_order_fill(self, execution_engine):
        """Verify basic order fill with deterministic slippage."""
        sig = TradeSignal(direction="BUY", price=2000.0, stop_loss=1990.0, take_profit=2020.0)
        
        # Mock slippage to be 0 for deterministic test
        execution_engine.entry_slippage_points = 0.0
        
        fill = execution_engine.execute_order(
            signal=sig,
            symbol="XAUUSDm",
            current_price=2000.0,
            spread=0.10,
            point=0.01,
            timestamp=1700000000
        )
        
        assert fill is not None
        assert fill["direction"] == "BUY"
        assert fill["fill_price"] == 2000.0
        assert fill["sl"] == 1990.0

    def test_spread_rejection(self, execution_engine):
        """Ensure orders are rejected if spread exceeds maximum points."""
        sig = TradeSignal(direction="BUY", price=2000.0)
        execution_engine.max_spread_points = 10.0 # 10 points max
        
        # 1.0 spread / 0.01 point = 100 points (Too High)
        fill = execution_engine.execute_order(
            signal=sig,
            symbol="XAUUSDm",
            current_price=2000.0,
            spread=1.0, 
            point=0.01
        )
        
        assert fill is None

    def test_news_filter_blockage(self, execution_engine):
        """Verify order rejection when news filter is active."""
        sig = TradeSignal(direction="SELL", price=2000.0)
        
        # Patch the news filter's is_blocked method
        with patch.object(execution_engine.news_filter, 'is_blocked', return_value=True):
            fill = execution_engine.execute_order(
                signal=sig,
                symbol="XAUUSDm",
                current_price=2000.0,
                spread=0.05,
                point=0.01
            )
            assert fill is None

    def test_slippage_sampling_ranges(self, execution_engine):
        """Ensure sampled slippage stays within configured ranges."""
        execution_engine.entry_slippage_points = 5.0
        point = 0.01
        
        for _ in range(100):
            slip = execution_engine.sample_slippage_points(point, event="entry")
            # Slip is uniform(0, 5) * 0.01 -> [0, 0.05]
            assert 0.0 <= slip <= 0.051 # Small float delta
