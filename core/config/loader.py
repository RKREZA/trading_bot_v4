import json
import os
import logging
from typing import Dict, Any

from dotenv import load_dotenv

from core.config.schema import GlobalConfig

logger = logging.getLogger("trading_bot.config_loader")

load_dotenv()


class ConfigLoader:
    def __init__(
        self,
        global_path: str = "config/config.json",
        symbols_dir: str = "config/symbols",
        environment: str = "live",
    ):
        self.global_path = global_path
        self.symbols_dir = symbols_dir
        self.environment = environment

        raw = self._load_json(global_path)

        if environment == "backtest":
            backtest_cfg = self._load_json("config/backtest.json")
            self._deep_update(raw, backtest_cfg)

        self._apply_env_overrides(raw)
        self._raw_global = raw
        self.config = GlobalConfig.model_validate(raw)

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

    def _deep_update(self, d: Dict, u: Dict) -> Dict:
        for k, v in u.items():
            if isinstance(v, dict) and isinstance(d.get(k), dict):
                d[k] = self._deep_update(d[k], v)
            else:
                d[k] = v
        return d

    def _apply_env_overrides(self, raw: Dict[str, Any]) -> None:
        if "mt5" not in raw:
            raw["mt5"] = {}
        mt5 = raw["mt5"]

        env_login = os.getenv("MT5_LOGIN")
        if env_login:
            mt5["login"] = int(env_login)

        env_password = os.getenv("MT5_PASSWORD")
        if env_password:
            mt5["password"] = env_password

        env_server = os.getenv("MT5_SERVER")
        if env_server:
            mt5["server"] = env_server

        env_offset = os.getenv("BROKER_UTC_OFFSET")
        if env_offset:
            mt5["broker_utc_offset"] = int(env_offset)

    def get_symbol_config(self, symbol: str) -> Dict[str, Any]:
        symbol_path = os.path.join(self.symbols_dir, f"{symbol}.json")
        merged = self._raw_global.copy()

        if not os.path.exists(symbol_path):
            logger.warning(f"Symbol config not found for {symbol}. Using global defaults.")
            return merged

        symbol_spec = self._load_json(symbol_path)

        if "symbol_info" in symbol_spec:
            if "symbols_config" not in merged:
                merged["symbols_config"] = {}
            merged["symbols_config"][symbol] = symbol_spec["symbol_info"]

        for key, val in symbol_spec.items():
            if key in ("symbol_info", "backtest"):
                continue
            if key == "strategies":
                for strat_name, strat_cfg in val.items():
                    if strat_name not in merged:
                        merged[strat_name] = {}
                    merged[strat_name].update(strat_cfg)
                    if "strategies" not in merged:
                        merged["strategies"] = {}
                    if isinstance(merged["strategies"], dict):
                        if strat_name not in merged["strategies"]:
                            merged["strategies"][strat_name] = {}
                        merged["strategies"][strat_name].update(strat_cfg)
            else:
                if key not in merged:
                    merged[key] = val
                elif isinstance(val, dict) and isinstance(merged[key], dict):
                    merged[key].update(val)
                else:
                    merged[key] = val

        if "backtest" in symbol_spec:
            if "backtest" not in merged:
                merged["backtest"] = {}
            merged["backtest"].update(symbol_spec["backtest"])

        return merged

    def get_validated_symbol_config(self, symbol: str) -> GlobalConfig:
        raw = self.get_symbol_config(symbol)
        return GlobalConfig.model_validate(raw)

    def list_symbols(self) -> list:
        if not os.path.exists(self.symbols_dir):
            return []
        symbols = []
        for filename in os.listdir(self.symbols_dir):
            if filename.endswith(".json"):
                symbols.append(filename.replace(".json", ""))
        return sorted(symbols)

    def reload(self) -> None:
        raw = self._load_json(self.global_path)
        if self.environment == "backtest":
            backtest_cfg = self._load_json("config/backtest.json")
            self._deep_update(raw, backtest_cfg)
        self._apply_env_overrides(raw)
        self._raw_global = raw
        self.config = GlobalConfig.model_validate(raw)
        logger.info("Configuration reloaded")

    @classmethod
    def load_active_config(cls, symbol: str) -> Dict[str, Any]:
        loader = cls(global_path="config/config.json")
        return loader.get_symbol_config(symbol)


def load_config(global_path: str = "config/config.json") -> Dict[str, Any]:
    loader = ConfigLoader(global_path=global_path)
    return loader._raw_global
