"""
TRADING BOT V3 - Data Fetcher
Fetches candle data from MT5 with time-based caching.
"""

import logging
import time
import datetime
import numpy as np
from typing import Dict, List, Optional
from core.common.types import CandleArray

try:
    import MetaTrader5 as mt5
except ImportError:
    mt5 = None


from .connection import MT5Connection
logger = logging.getLogger("trading_bot.data")

# Cache duration per timeframe (seconds)
CACHE_TTL = {
    "M1": 1,
    "M5": 2,
    "M15": 10,
    "M30": 30,
    "H1": 60,
    "H4": 300,
    "D1": 3600,
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
    """
    Main component for retrieving market data from MT5.
    Implements an incremental caching strategy to minimize MT5 terminal overhead:
    - TIMEFRAME_MAP: Maps string identifiers to MT5 constants.
    - CACHE_TTL: Defines the validity period for cached data per timeframe.
    - Incremental Update: Only fetches the latest few candles if the history is already cached.
    """

    def __init__(self):
        """Initializes the DataFetcher with an empty internal cache."""
        self._cache: Dict[str, dict] = {}  # key -> {data, timestamp}

    def _cache_key(self, symbol: str, timeframe: str) -> str:
        """Generate a unique cache key."""
        return f"{symbol}_{timeframe}"

    def _merge_candles(self, existing: List[dict], new: List[dict], max_len: int) -> List[dict]:
        """Merge new candles into existing cache, deduplicating by timestamp."""
        if not existing: return new
        if not new: return existing
        
        # Combine existing and new, keyed by time to automatically deduplicate
        # We prefer 'new' data for the same timestamp
        combined = {c['time']: c for c in existing}
        for c in new:
            combined[c['time']] = c
            
        # Sort by time and trim to max_len
        sorted_times = sorted(combined.keys())
        return [combined[t] for t in sorted_times[-max_len:]]

    def fetch_candles(self, symbol: str, timeframe: str, count: int = 500, force_refresh: bool = False) -> CandleArray:
        """
        Retrieves candle data for a symbol and timeframe.
        Attempts to update the cache incrementally if possible.
        
        Args:
            symbol (str): Symbol name.
            timeframe (str): Timeframe (e.g., 'M5', 'H1').
            count (int): Requested number of historical candles.
            force_refresh (bool): If True, bypasses TTL and forces a refresh.
            
        Returns:
            CandleArray: The requested data wrapped in a CandleArray object.
        """
        if timeframe not in TIMEFRAME_MAP or TIMEFRAME_MAP[timeframe] is None:
            logger.warning("Invalid timeframe: %s", timeframe)
            return CandleArray.from_dicts([])

        key = self._cache_key(symbol, timeframe)
        ttl = CACHE_TTL.get(timeframe, 60)
        now = time.time()

        cached = self._cache.get(key)
        
        # 1. Light Cache Check (Return if extremely fresh)
        if not force_refresh and cached and (now - cached["timestamp"]) < (ttl / 2):
            return cached["array"]

        # 2. Incremental Fetch Decision
        fetch_count = count
        is_incremental = False
        
        # [THE FIX]: If we already have the full history cached, we ONLY need 
        # to fetch the latest few candles to update the tip, regardless of force_refresh!
        if cached and len(cached["data"]) >= count:
            fetch_count = 10 # Just get the latest 10 candles
            is_incremental = True

        # Fetch from MT5
        try:
            import pandas as pd
            with MT5Connection.MT5_LOCK:
                select_res = mt5.symbol_select(symbol, True)
            if not select_res:
                error = mt5.last_error()
                logger.error("symbol_select failed for %s: %s", symbol, error)
                return CandleArray.from_dicts([])
            
            logger.debug("MT5 Fetch (%s): %s %s (%d candles)...", 
                         "INC" if is_incremental else "FULL", symbol, timeframe, fetch_count)
            
            with MT5Connection.MT5_LOCK:
                rates = mt5.copy_rates_from_pos(symbol, TIMEFRAME_MAP[timeframe], 0, fetch_count)
            
            if rates is None or len(rates) == 0:
                if not is_incremental:
                    # Fallback for full fetch failure
                    logger.warning("%s %s: Full history (%d) not available. Retrying with half...", symbol, timeframe, count)
                    temp_count = count // 2
                    while temp_count >= 100:
                        with MT5Connection.MT5_LOCK:
                            rates = mt5.copy_rates_from_pos(symbol, TIMEFRAME_MAP[timeframe], 0, temp_count)
                        if rates is not None and len(rates) > 0:
                            logger.info("%s %s: Recovered %d candles (requested %d)", symbol, timeframe, len(rates), count)
                            break
                        temp_count //= 2
            
            if rates is None or len(rates) == 0:
                return cached["array"] if cached else CandleArray.from_dicts([])

            # Convert to dicts
            df = pd.DataFrame(rates)
            new_candles = df.to_dict('records')

            # 3. Merge or Replace
            if is_incremental:
                candles = self._merge_candles(cached["data"], new_candles, count)
            else:
                candles = new_candles

            # Update cache
            array = CandleArray.from_dicts(candles)
            self._cache[key] = {"data": candles, "array": array, "timestamp": now}
            return array

        except Exception as e:
            logger.exception("Error fetching candles for %s %s: %s", symbol, timeframe, e)
            return cached["array"] if cached else CandleArray.from_dicts([])

    def fetch_candles_range(self, symbol: str, timeframe: str, date_from: datetime.datetime, date_to: datetime.datetime) -> CandleArray:
        """
        Fetches historical OHLC data for a specific date range (broker-limited).
        
        Args:
            symbol (str): Symbol name.
            timeframe (str): Timeframe constant.
            date_from (datetime): Start of range (UTC).
            date_to (datetime): End of range (UTC).
            
        Returns:
            CandleArray: The requested historical data.
        """
        if timeframe not in TIMEFRAME_MAP or TIMEFRAME_MAP[timeframe] is None:
            return CandleArray.from_dicts([])

        try:
            with MT5Connection.MT5_LOCK:
                if not mt5.symbol_select(symbol, True):
                    return CandleArray.from_dicts([])

                logger.debug("MT5 Range Fetch: %s %s (%s to %s)...", symbol, timeframe, date_from, date_to)
                rates = mt5.copy_rates_range(symbol, TIMEFRAME_MAP[timeframe], date_from, date_to)
            
            if rates is None or len(rates) == 0:
                logger.warning("No range data for %s %s", symbol, timeframe)
                return CandleArray.from_dicts([])

            candles = []
            for r in rates:
                candles.append({
                    "time": int(r["time"]),
                    "open": float(r["open"]),
                    "high": float(r["high"]),
                    "low": float(r["low"]),
                    "close": float(r["close"]),
                    "tick_volume": int(r["tick_volume"]),
                    "spread": int(r.get("spread", 0)),
                })
            return CandleArray.from_dicts(candles)
        except Exception as e:
            logger.error("Error in fetch_candles_range: %s", e)
            return CandleArray.from_dicts([])

    def fetch_ticks_range(self, symbol: str, date_from: datetime.datetime, date_to: datetime.datetime) -> List[dict]:
        """
        Fetches raw tick data (including bid/ask/flags) for backtesting or auditing.
        
        Args:
            symbol (str): Symbol name.
            date_from (datetime): Start time.
            date_to (datetime): End time.
            
        Returns:
            List[dict]: List of raw ticks as dictionaries.
        """
        try:
            with MT5Connection.MT5_LOCK:
                if not mt5.symbol_select(symbol, True):
                    return []

                logger.debug("MT5 Tick Fetch: %s (%s to %s)...", symbol, date_from, date_to)
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
            with MT5Connection.MT5_LOCK:
                info = mt5.symbol_info(symbol)
            if info is None:
                return None
            return {
                "point": info.point,
                "digits": info.digits,
                "contract_size": info.trade_contract_size,
                "min_lot": info.volume_min,
                "max_lot": info.volume_max,
                "lot_step": info.volume_step,
                "spread": info.spread,
                "bid": info.bid,
                "ask": info.ask,
            }
        except Exception as e:
            logger.exception("Error getting symbol info for %s: %s", symbol, e)
            return None

    @staticmethod
    def validate_data_integrity(candles: CandleArray, timeframe: str) -> dict:
        """
        Institutional Grade: detects gaps in historical data.
        Returns a report with missing candle counts and gap locations.
        """
        if len(candles) < 2:
            return {"status": "OK", "missing_count": 0}
        
        # Determine expected interval
        intervals = {"M1": 60, "M5": 300, "M15": 900, "M30": 1800, "H1": 3600, "H4": 14400, "D1": 86400}
        expected_diff = intervals.get(timeframe, 300)
        
        diffs = np.diff(candles.time)
        # We allow for weekend gaps (anything > 48 hours is considered a weekend/market close)
        gap_indices = np.where((diffs > expected_diff) & (diffs < 172800))[0]
        
        missing_total = 0
        gaps = []
        for idx in gap_indices:
            actual_diff = diffs[idx]
            missing = int((actual_diff / expected_diff) - 1)
            missing_total += missing
            gaps.append({
                "from": datetime.datetime.fromtimestamp(candles.time[idx], tz=datetime.timezone.utc),
                "to": datetime.datetime.fromtimestamp(candles.time[idx+1], tz=datetime.timezone.utc),
                "missing": missing
            })
            
        return {
            "status": "CRITICAL" if missing_total > (len(candles) * 0.05) else "WARNING" if missing_total > 0 else "OK",
            "missing_total": missing_total,
            "gap_count": len(gaps),
            "gaps": gaps[:5] # Show first 5 gaps
        }

    def clear_cache(self):
        """Clear all cached data."""
        self._cache.clear()
