"""
Test suite for strategy extensibility.
"""

import pytest
from unittest.mock import MagicMock, patch
from strategies import create_strategy, STRATEGY_REGISTRY


class TestStrategyExtensibility:
    """Test strategy creation and extensibility."""

    def test_create_sma_strategy(self):
        """Test SMA sample strategy creation."""
        config = {"SMASampleStrategy": {"fast_sma_period": 20, "slow_sma_period": 50}}
        strategy = create_strategy("SMA_v1", "SMASAMPLE", config)
        assert strategy is not None
        assert strategy.strategy_id == "SMA_v1"

    def test_strategy_registry_contains_sample(self):
        """Test strategy is registered."""
        assert "SMASAMPLE" in STRATEGY_REGISTRY

    def test_strategy_auto_resolution(self):
        """Test auto-resolution of strategy type."""
        strategy = create_strategy("SMA_test_strategy", "SMASAMPLE", {})
        assert strategy is not None

    def test_strategy_with_config(self):
        """Test strategy uses config values."""
        config = {
            "SMASampleStrategy": {
                "fast_sma_period": 10,
                "slow_sma_period": 30,
                "min_confidence": 0.8
            }
        }
        strategy = create_strategy("SMA_v1", "SMASAMPLE", config)
        assert strategy is not None

    def test_nonexistent_strategy_raises(self):
        """Test nonexistent strategy raises error."""
        with pytest.raises(ValueError):
            create_strategy("Test", "NONEXISTENT", {})

    def test_strategy_methods_exist(self):
        """Test strategy has required methods."""
        strategy = create_strategy("SMA_v1", "SMASAMPLE", {})
        assert hasattr(strategy, "generate_signal")
        assert hasattr(strategy, "get_stop_loss")
        assert hasattr(strategy, "get_take_profit")
        assert hasattr(strategy, "get_metrics")
        assert hasattr(strategy, "get_thresholds")

    def test_multiple_strategy_instances(self):
        """Test multiple strategy instances are independent."""
        config = {"SMASampleStrategy": {"fast_sma_period": 20}}
        s1 = create_strategy("SMA_v1", "SMASAMPLE", config)
        s2 = create_strategy("SMA_v2", "SMASAMPLE", config)
        assert s1 is not s2