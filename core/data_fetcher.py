"""
TRADING BOT V3 - Data Fetcher
Fetches candle data from MT5 with time-based caching.
"""

import logging
import time
import datetime
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
            # Explicitly select symbol to ensure it's in MarketWatch
            if not mt5.symbol_select(symbol, True):
                error = mt5.last_error()
                print(f"FATAL ERROR: symbol_select failed for {symbol}: {error}")
                return []
            
            # Resilient fetching: if the full count fails, try to get as much as possible
            print(f"DEBUG: MT5 Fetch: {symbol} {timeframe} ({count} candles)...")
            rates = mt5.copy_rates_from_pos(symbol, TIMEFRAME_MAP[timeframe], 0, count)
            
            if rates is None or len(rates) == 0:
                # Try to find exactly how much history is available
                # Binary search or just try 50%
                logger.warning("%s %s: Full history (%d) not available. Retrying with half...", symbol, timeframe, count)
                temp_count = count // 2
                while temp_count >= 100:
                    rates = mt5.copy_rates_from_pos(symbol, TIMEFRAME_MAP[timeframe], 0, temp_count)
                    if rates is not None and len(rates) > 0:
                        logger.info("%s %s: Recovered %d candles (requested %d)", symbol, timeframe, len(rates), count)
                        break
                    temp_count //= 2
            
            if rates is None or len(rates) == 0:
                logger.error("No data returned for %s %s after retries", symbol, timeframe)
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
                    # Some versions return tuples or named arrays without dtype.names
                    # Fallback to index-based mapping if needed, but dict(r) usually works for named rows
                    try:
                        candles.append(dict(r))
                    except:
                        # Fallback for pure tuples
                        candles.append({
                            "time": int(r[0]),
                            "open": float(r[1]),
                            "high": float(r[2]),
                            "low": float(r[3]),
                            "close": float(r[4]),
                            "tick_volume": int(r[5]),
                        })

            # Update cache
            self._cache[key] = {"data": candles, "timestamp": now}
            logger.debug("Fetched %d %s candles for %s", len(candles), timeframe, symbol)
            return candles

        except Exception as e:
            logger.exception("Error fetching candles for %s %s: %s", symbol, timeframe, e)
            return cached["data"] if cached else []

    def fetch_candles_range(self, symbol: str, timeframe: str, date_from: datetime.datetime, date_to: datetime.datetime) -> List[dict]:
        """
        Fetch OHLC data for a specific date range.
        """
        if timeframe not in TIMEFRAME_MAP or TIMEFRAME_MAP[timeframe] is None:
            return []

        try:
            if not mt5.symbol_select(symbol, True):
                return []

            print(f"DEBUG: MT5 Range Fetch: {symbol} {timeframe} ({date_from} to {date_to})...")
            rates = mt5.copy_rates_range(symbol, TIMEFRAME_MAP[timeframe], date_from, date_to)
            
            if rates is None or len(rates) == 0:
                logger.warning("No range data for %s %s", symbol, timeframe)
                return []

            candles = []
            for r in rates:
                candles.append({
                    "time": int(r["time"]),
                    "open": float(r["open"]),
                    "high": float(r["high"]),
                    "low": float(r["low"]),
                    "close": float(r["close"]),
                    "tick_volume": int(r["tick_volume"]),
                })
            return candles
        except Exception as e:
            logger.error("Error in fetch_candles_range: %s", e)
            return []

    def fetch_ticks_range(self, symbol: str, date_from: datetime.datetime, date_to: datetime.datetime) -> List[dict]:
        """
        Fetch real tick data for a specific date range.
        """
        try:
            if not mt5.symbol_select(symbol, True):
                return []

            print(f"DEBUG: MT5 Tick Fetch: {symbol} ({date_from} to {date_to})...")
            # Fetch all ticks (bid, ask, last)
            ticks = mt5.copy_ticks_range(symbol, date_from, date_to, mt5.COPY_TICKS_ALL)
            
            if ticks is None or len(ticks) == 0:
                logger.warning("No tick data for %s", symbol)
                return []

            # Convert to list of dicts for consistency
            tick_list = []
            for t in ticks:
                tick_list.append({
                    "time": int(t[0]), # time
                    "bid": float(t[1]), # bid
                    "ask": float(t[2]), # ask
                    "last": float(t[3]), # last
                    "flags": int(t[4]), # flags
                })
            return tick_list
        except Exception as e:
            logger.error("Error in fetch_ticks_range: %s", e)
            return []

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
