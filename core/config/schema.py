import logging

logger = logging.getLogger("trading_bot.config_schema")

class ConfigValidationError(Exception):
    pass

class V5ConfigSchema:
    """
    Robust pure-Python schema validator for hot-reloading.
    Prevents the live execution engine from crashing if a user introduces 
    a bad type (e.g. string instead of float) in config.json.
    """
    
    @classmethod
    def validate(cls, config_dict: dict) -> dict:
        """Validates and coerces the incoming configuration dictionary."""
        validated = {}
        
        # Risk Governance
        risk = config_dict.get("risk_governance", {})
        validated["risk_governance"] = {
            "risk_per_trade_pct": cls._coerce_float(risk.get("risk_per_trade_pct", 1.0)),
            "max_daily_loss_pct": cls._coerce_float(risk.get("max_daily_loss_pct", 5.0)),
            "max_drawdown_halt_pct": cls._coerce_float(risk.get("max_drawdown_halt_pct", 15.0)),
            "max_consecutive_losses": int(risk.get("max_consecutive_losses", 5)),
            "max_parallel_strategies": int(risk.get("max_parallel_strategies", 4)),
            "max_net_exposure_long": int(risk.get("max_net_exposure_long", 3)),
            "max_net_exposure_short": int(risk.get("max_net_exposure_short", 3)),
            "min_notional_value": cls._coerce_float(risk.get("min_notional_value", 0.0))
        }
        
        # Execution
        exe = config_dict.get("execution", {})
        pf = exe.get("partial_fill", {})
        validated["execution"] = {
            "latency_ms": int(exe.get("latency_ms", 150)),
            "max_spread_points": cls._coerce_float(exe.get("max_spread_points", 500.0)),
            "entry_slippage_points": cls._coerce_float(exe.get("entry_slippage_points", 1.0)),
            "sl_exit_slippage_points": cls._coerce_float(exe.get("sl_exit_slippage_points", 2.0)),
            "partial_fill": {
                "enabled": bool(pf.get("enabled", True)),
                "threshold_lots": cls._coerce_float(pf.get("threshold_lots", 5.0)),
                "part1_ratio": cls._coerce_float(pf.get("part1_ratio", 0.4)),
                "part2_ratio": cls._coerce_float(pf.get("part2_ratio", 0.6)),
                "worse_slippage_multiplier": cls._coerce_float(pf.get("worse_slippage_multiplier", 2.0))
            }
        }
        
        # Include base/unvalidated objects untouched to prevent overwriting complex nested structures
        # if they aren't explicitly coerced above.
        merged = config_dict.copy()
        merged.update(validated)
        return merged

    @staticmethod
    def _coerce_float(val) -> float:
        try:
            return float(val)
        except (ValueError, TypeError):
            raise ConfigValidationError(f"Expected numeric float, got {type(val)}: {val}")
