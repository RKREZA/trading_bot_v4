"""
Test suite for Pydantic configuration schema validation.
"""

import pytest
from pydantic import ValidationError
from core.config.schema import (
    GlobalConfig,
    RiskGovernanceConfig,
    ExecutionConfig,
    PartialFillConfig,
)


class TestRiskGovernanceConfig:
    def test_defaults(self):
        cfg = RiskGovernanceConfig()
        assert cfg.risk_per_trade_pct == 1.0
        assert cfg.max_daily_loss_pct == 5.0

    def test_explicit_values(self):
        cfg = RiskGovernanceConfig(risk_per_trade_pct=2.0, max_daily_loss_pct=10.0)
        assert cfg.risk_per_trade_pct == 2.0
        assert cfg.max_daily_loss_pct == 10.0

    def test_string_coercion(self):
        cfg = RiskGovernanceConfig(risk_per_trade_pct="2.0", max_daily_loss_pct="10.0")
        assert cfg.risk_per_trade_pct == 2.0

    def test_invalid_type_raises(self):
        with pytest.raises(ValidationError):
            RiskGovernanceConfig(risk_per_trade_pct="not_a_number")


class TestExecutionConfig:
    def test_defaults(self):
        cfg = ExecutionConfig()
        assert cfg.latency_ms == 150
        assert cfg.max_spread_points == 500.0

    def test_explicit_values(self):
        cfg = ExecutionConfig(latency_ms=200, max_spread_points=800.0)
        assert cfg.latency_ms == 200
        assert cfg.max_spread_points == 800.0

    def test_partial_fill_nested(self):
        cfg = ExecutionConfig(
            partial_fill=PartialFillConfig(enabled=True, threshold_lots=5.0)
        )
        assert cfg.partial_fill.enabled is True
        assert cfg.partial_fill.threshold_lots == 5.0


class TestGlobalConfig:
    def test_empty_gives_defaults(self):
        cfg = GlobalConfig()
        assert cfg.risk_governance.risk_per_trade_pct == 1.0
        assert cfg.risk_governance.max_daily_loss_pct == 5.0
        assert cfg.execution.latency_ms == 150

    def test_from_dict(self):
        data = {
            "risk_governance": {
                "risk_per_trade_pct": 2.0,
                "max_daily_loss_pct": 10.0,
            },
            "execution": {
                "latency_ms": 200,
                "max_spread_points": 500.0,
            },
        }
        cfg = GlobalConfig(**data)
        assert cfg.risk_governance.risk_per_trade_pct == 2.0
        assert cfg.execution.latency_ms == 200

    def test_extra_fields_allowed(self):
        cfg = GlobalConfig(custom_field="test")
        assert cfg.custom_field == "test"

    def test_version_default(self):
        cfg = GlobalConfig()
        assert cfg.version == "6.0"
