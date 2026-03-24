"""
TRADING BOT V3 - Data Fetcher
Fetches candle data from MT5 with time-based caching.
"""

import logging
import time
from typing import Dict, List, Optional

try:
    import MetaTrader5 as mt5
except ImportError:
    mt5 = None

logger = logging.getLogger("trading_bot.data")

# Cache duration per timeframe (seconds)
CACHE_TTL = {
    "M1": 60,
    "M5": 300,
    "M15": 900,
    "M30": 1800,
    "H1": 3600,
    "H4": 14400,
    "D1": 86400,
}

TIMEFRAME_MAP = {
    "M1": mt5.TIMEFRAME_M1 if mt5 else None,
    "M5": mt5.TIMEFRAME_M5 if mt5 else None,
    "M15": mt5.TIMEFRAME_M15 if mt5 else None,
    "M30": mt5.TIMEFRAME_M30 if mt5 else None,
    "H1": mt5.TIMEFRAME_H1 if mt5 else None,
    "H4": mt5.TIMEFRAME_H4 if mt5 else None,
    "D1": mt5.TIMEFRAME_D1 if mt5 else None,
}


class DataFetcher:
    """Fetches and caches candle data from MT5."""

    def __init__(self):
        self._cache: Dict[str, dict] = {}  # key -> {data, timestamp}

    def _cache_key(self, symbol: str, timeframe: str, count: int) -> str:
        return f"{symbol}_{timeframe}_{count}"

    def fetch_candles(self, symbol: str, timeframe: str, count: int = 500) -> List[dict]:
        """
        Fetch candle data, returning cached data if still fresh.
        """
        if timeframe not in TIMEFRAME_MAP or TIMEFRAME_MAP[timeframe] is None:
            logger.warning("Invalid timeframe: %s", timeframe)
            return []

        key = self._cache_key(symbol, timeframe, count)
        ttl = CACHE_TTL.get(timeframe, 60)
        now = time.time()

        # Return cached data if fresh
        cached = self._cache.get(key)
        if cached and (now - cached["timestamp"]) < ttl:
            return cached["data"]

        # Fetch from MT5
        try:
            rates = mt5.copy_rates_from_pos(symbol, TIMEFRAME_MAP[timeframe], 0, count)
            if rates is None or len(rates) == 0:
                logger.warning("No data returned for %s %s", symbol, timeframe)
                return cached["data"] if cached else []

            candles = []
            dtype_names = rates.dtype.names if hasattr(rates, "dtype") else None
            for r in rates:
                if dtype_names:
                    candles.append({
                        "time": int(r["time"]),
                        "open": float(r["open"]),
                        "high": float(r["high"]),
                        "low": float(r["low"]),
                        "close": float(r["close"]),
                        "tick_volume": int(r["tick_volume"]),
                    })
                else:
                    candles.append(dict(r))

            # Update cache
            self._cache[key] = {"data": candles, "timestamp": now}
            logger.debug("Fetched %d %s candles for %s", len(candles), timeframe, symbol)
            return candles

        except Exception as e:
            logger.exception("Error fetching candles for %s %s: %s", symbol, timeframe, e)
            return cached["data"] if cached else []

    def get_symbol_info(self, symbol: str) -> Optional[dict]:
        """Get current symbol information (tick data)."""
        try:
            info = mt5.symbol_info(symbol)
            if info is None:
                return None
            return {
                "point": info.point,
                "digits": info.digits,
                "contract_size": info.trade_contract_size,
                "min_lot": info.volume_min,
                "max_lot": info.volume_max,
                "spread": info.spread,
                "bid": info.bid,
                "ask": info.ask,
            }
        except Exception as e:
            logger.exception("Error getting symbol info for %s: %s", symbol, e)
            return None

    def clear_cache(self):
        """Clear all cached data."""
        self._cache.clear()
