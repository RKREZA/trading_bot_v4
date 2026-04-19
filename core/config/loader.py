import json
import os
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger("trading_bot.config_loader")

class ConfigLoader:
    """
    V5-INSIGNIA Hierarchical Configuration Loader.
    Implements a Global -> Symbol inheritance model.
    """
    
    def __init__(self, global_path: str = "config/config.json", symbols_dir: str = "config/symbols"):
        self.global_path = global_path
        self.symbols_dir = symbols_dir
        self.global_config = self._load_json(global_path)

    def _load_json(self, path: str) -> Dict[str, Any]:
        if not os.path.exists(path):
            logger.warning(f"Config file not found: {path}")
            return {}
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load config {path}: {e}")
            return {}

    def get_symbol_config(self, symbol: str) -> Dict[str, Any]:
        """
        Loads and merges the global config with symbol-specific overrides.
        """
        symbol_path = os.path.join(self.symbols_dir, f"{symbol}.json")
        if not os.path.exists(symbol_path):
            logger.warning(f"Symbol config not found for {symbol}. Using global defaults only.")
            return self.global_config.copy()

        symbol_spec = self._load_json(symbol_path)
        
        # Deep Merge Logic
        merged = self.global_config.copy()
        
        # Symbol JSON has: symbol_info, strategies, backtest
        # We want to merge these into the root or their respective blocks
        
        # 1. Symbol Info -> symbols_config[symbol]
        if "symbol_info" in symbol_spec:
            if "symbols_config" not in merged:
                merged["symbols_config"] = {}
            merged["symbols_config"][symbol] = symbol_spec["symbol_info"]
            
        # 2. Strategies -> Root level (to maintain compatibility with existing strategy logic)
        if "strategies" in symbol_spec:
            for strat_name, strat_cfg in symbol_spec["strategies"].items():
                merged[strat_name] = strat_cfg
                
        # 3. Backtest -> backtest block
        if "backtest" in symbol_spec:
            if "backtest" not in merged:
                merged["backtest"] = {}
            merged["backtest"].update(symbol_spec["backtest"])
            
        # 4. Sessions -> sessions block (if exists)
        if "sessions" in symbol_spec:
            merged["sessions"] = symbol_spec["sessions"]
            
        return merged

    def list_symbols(self) -> list:
        """Discovers all symbols with valid configuration files."""
        if not os.path.exists(self.symbols_dir):
            return []
        
        symbols = []
        for filename in os.listdir(self.symbols_dir):
            if filename.endswith(".json"):
                symbols.append(filename.replace(".json", ""))
        return sorted(symbols)

    @classmethod
    def load_active_config(cls, symbol: str) -> Dict[str, Any]:
        """Static helper for quick loading."""
        loader = cls(global_path="config/config.json")
        return loader.get_symbol_config(symbol)
