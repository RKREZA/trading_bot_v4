from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional
import json
import yaml
import os
import logging

logger = logging.getLogger("trading_bot.config")

class RiskConfig(BaseModel):
    max_drawdown: float = 0.10
    daily_loss_limit: float = 0.03
    max_concurrent_trades: int = 3
    risk_per_trade: float = 0.01

class StrategyConfig(BaseModel):
    name: str
    symbol: str
    timeframe: int = 5
    parameters: Dict[str, Any] = {}
    enabled: bool = False

class GlobalConfig(BaseModel):
    version: str = "5.0"
    mt5_login: int = 0
    mt5_password: str = ""
    mt5_server: str = ""
    broker_utc_offset: int = 0
    risk: RiskConfig = RiskConfig()
    strategies: List[StrategyConfig] = []

class ConfigManager:
    """
    Handles versioned configuration loading and runtime reloading.
    Supports JSON and YAML.
    """
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.current_config: GlobalConfig = GlobalConfig()
        self.load()

    def load(self):
        if not os.path.exists(self.config_path):
            logger.warning(f"Config file not found at {self.config_path}. Using defaults.")
            return

        try:
            with open(self.config_path, 'r') as f:
                if self.config_path.endswith('.yaml') or self.config_path.endswith('.yml'):
                    data = yaml.safe_load(f)
                else:
                    data = json.load(f)
            
            self.current_config = GlobalConfig(**data)
            logger.info(f"Configuration loaded from {self.config_path}")
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")

    def save(self):
        try:
            with open(self.config_path, 'w') as f:
                if self.config_path.endswith('.yaml') or self.config_path.endswith('.yml'):
                    yaml.dump(self.current_config.model_dump(), f)
                else:
                    json.dump(self.current_config.model_dump(), f, indent=4)
            logger.info(f"Configuration saved to {self.config_path}")
        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")

    def update_strategy(self, strategy_name: str, enabled: bool, params: Dict[str, Any] = None):
        for s in self.current_config.strategies:
            if s.name == strategy_name:
                s.enabled = enabled
                if params:
                    s.parameters.update(params)
                break
        self.save()
