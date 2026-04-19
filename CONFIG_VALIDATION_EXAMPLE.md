# Configuration Validation Examples
=====================================

This document provides examples of how to validate and extend the configuration for the V5-INSIGNIA trading system.

## Basic Configuration Structure

The system uses a hierarchical configuration approach:
1. Global configuration in `config.json`
2. Symbol-specific overrides in `configs/symbols/{SYMBOL}.json`

## Example: Adding a New Strategy Parameter

To add a custom parameter to your strategy configuration:

### 1. Update Global Configuration (`config.json`)
```json
{
  "risk_governance": {
    "risk_per_trade_pct": 2.0,
    "max_daily_loss_pct": 10.0,
    "custom_risk_multiplier": 1.5
  },
  "execution": {
    "latency_ms": 150,
    "custom_slippage_model": "volume_weighted"
  },
  "MyCustomStrategy": {
    "enabled": true,
    "lookback_period": 50,
    "entry_threshold": 0.75,
    "exit_threshold": 0.25,
    "use_volatility_filter": true
  }
}
```

### 2. Access Parameters in Your Strategy
```python
class MyCustomStrategy(BaseStrategy):
    def __init__(self, strategy_id: str, config: dict):
        super().__init__(strategy_id, config)
        
        # Access global risk parameters
        self.custom_risk_multiplier = config.get("risk_governance", {}).get("custom_risk_multiplier", 1.0)
        
        # Access execution parameters
        self.custom_slippage_model = config.get("execution", {}).get("custom_slippage_model", "fixed")
        
        # Access strategy-specific parameters
        self.lookback_period = config.get("MyCustomStrategy", {}).get("lookback_period", 20)
        self.entry_threshold = config.get("MyCustomStrategy", {}).get("entry_threshold", 0.7)
        self.exit_threshold = config.get("MyCustomStrategy", {}).get("exit_threshold", 0.3)
        self.use_volatility_filter = config.get("MyCustomStrategy", {}).get("use_volatility_filter", False)
```

### 3. Access Parameters via Strategy Config Helper
```python
class MyCustomStrategy(BaseStrategy):
    def get_strat_config(self) -> dict:
        """Get strategy-specific configuration block."""
        base_name = self.strategy_id.rsplit('_v', 1)[0] if '_v' in self.strategy_id else self.strategy_id
        
        # Check 'strategies' key first
        strategies_block = self.config.get("strategies", {})
        if self.strategy_id in strategies_block:
            return strategies_block[self.strategy_id]
        if base_name in strategies_block:
            return strategies_block[base_name]
            
        # Fallback to direct key
        if self.strategy_id in self.config:
            return self.config[self.strategy_id]
        if base_name in self.config:
            return self.config[base_name]
            
        return {}
    
    def __init__(self, strategy_id: str, config: dict):
        super().__init__(strategy_id, config)
        strat_config = self.get_strat_config()
        
        self.lookback_period = strat_config.get("lookback_period", 50)
        self.entry_threshold = strat_config.get("entry_threshold", 0.75)
        # ... etc
```

## Configuration Validation

The system includes automatic validation in `core/config/schema.py`:

### Validation Features:
1. **Type Coercion**: Strings are converted to appropriate types (float, int, bool)
2. **Default Values**: Missing parameters use sensible defaults
3. **Error Handling**: Invalid types raise `ConfigValidationError` with descriptive messages

### Example Validation Rules:
- `risk_per_trade_pct`: Must be convertible to float (default: 1.0)
- `max_consecutive_losses`: Must be convertible to int (default: 5)
- `latency_ms`: Must be convertible to int (default: 150)

## Best Practices

1. **Use Descriptive Names**: Choose clear, self-documenting parameter names
2. **Group Related Parameters**: Keep strategy-specific parameters together
3. **Provide Sensible Defaults**: Ensure the system works even with partial config
4. **Document Changes**: Update this file when adding new configuration parameters
5. **Test Config Changes**: Validate that new parameters work in both backtest and live modes

## Extending the Schema Validator

To add validation for new global parameters, edit `core/config/schema.py`:

```python
@classmethod
def validate(cls, config_dict: dict) -> dict:
    validated = {}
    
    # ... existing validation ...
    
    # Add your custom validation here
    validated["custom_section"] = {
        "custom_param": cls._coerce_float(config_dict.get("custom_section", {}).get("custom_param", 1.0)),
        "custom_flag": bool(config_dict.get("custom_section", {}).get("custom_flag", False))
    }
    
    # ... rest of method ...
```

## Testing Configuration Changes

Validate your configuration changes by running:
```bash
python -c "from core.config.loader import ConfigLoader; loader = ConfigLoader(); config = loader.get_symbol_config('XAUUSDm'); print('Config loaded successfully')"
```

This will load and validate the configuration using the same process as the live system.