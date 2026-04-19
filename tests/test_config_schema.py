"""
Test suite for configuration schema validation.
"""

import pytest
from core.config.schema import V5ConfigSchema, ConfigValidationError


class TestConfigSchema:
    """Test configuration validation."""

    def test_validate_basic(self):
        """Test basic config validation."""
        config = {
            "risk_governance": {
                "risk_per_trade_pct": 2.0,
                "max_daily_loss_pct": 10.0
            },
            "execution": {
                "latency_ms": 150
            }
        }
        validated = V5ConfigSchema.validate(config)
        assert validated is not None
        assert "risk_governance" in validated

    def test_validate_with_defaults(self):
        """Test config with missing values uses defaults."""
        config = {}
        validated = V5ConfigSchema.validate(config)
        assert validated["risk_governance"]["risk_per_trade_pct"] == 1.0
        assert validated["risk_governance"]["max_daily_loss_pct"] == 5.0

    def test_validate_type_coercion(self):
        """Test type coercion from string to float."""
        config = {
            "risk_governance": {
                "risk_per_trade_pct": "2.0",
                "max_daily_loss_pct": "10.0"
            }
        }
        validated = V5ConfigSchema.validate(config)
        assert validated["risk_governance"]["risk_per_trade_pct"] == 2.0

    def test_validate_invalid_type_raises(self):
        """Test invalid type raises error."""
        config = {
            "risk_governance": {
                "risk_per_trade_pct": "not_a_number"
            }
        }
        with pytest.raises(ConfigValidationError):
            V5ConfigSchema.validate(config)

    def test_execution_config(self):
        """Test execution config validation."""
        config = {
            "execution": {
                "latency_ms": 200,
                "max_spread_points": 500.0
            }
        }
        validated = V5ConfigSchema.validate(config)
        assert validated["execution"]["latency_ms"] == 200
        assert validated["execution"]["max_spread_points"] == 500.0

    def test_partial_fill_config(self):
        """Test partial fill config validation."""
        config = {
            "execution": {
                "partial_fill": {
                    "enabled": True,
                    "threshold_lots": 5.0
                }
            }
        }
        validated = V5ConfigSchema.validate(config)
        assert validated["execution"]["partial_fill"]["enabled"] is True
        assert validated["execution"]["partial_fill"]["threshold_lots"] == 5.0

    def test_coerce_float_with_int(self):
        """Test coercion from int to float."""
        result = V5ConfigSchema._coerce_float(42)
        assert result == 42.0

    def test_coerce_float_with_float(self):
        """Test float remains float."""
        result = V5ConfigSchema._coerce_float(3.14)
        assert result == pytest.approx(3.14)

    def test_coerce_float_invalid(self):
        """Test invalid float raises."""
        with pytest.raises(ConfigValidationError):
            V5ConfigSchema._coerce_float("invalid")