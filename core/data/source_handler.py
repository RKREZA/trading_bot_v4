import logging
import time
import datetime
import numpy as np
import threading
from typing import Dict, List, Optional, Any
from core.common.types import CandleArray

try:
    import MetaTrader5 as mt5
except ImportError:
    mt5 = None

class SourceHandler:
    """
    Institutional Data Source Handler.
    Abstracts MT5 communication and provides high-performance candle caching with rate limiting.
    Independently runnable and testable.
    """

    CACHE_TTL = {
        "M1": 1,
        "M5": 5,
        "M15": 15,
        "H1": 60,
        "D1": 3600
    }
    
    # Rate limiting: Max API calls per second
    MAX_CALLS_PER_SECOND = 10
    MIN_CALL_INTERVAL = 1.0 / MAX_CALLS_PER_SECOND

    def __init__(self, connection_lock=None):
        self._cache: Dict[str, Any] = {}
        self.lock = connection_lock
        self.logger = logging.getLogger("trading_bot.data")
        self._last_call_time: Dict[str, float] = {}
        self._rate_limit_lock = threading.Lock()
        
        if mt5 is None:
            self.logger.warning("MetaTrader5 package not found. Running in simulation mode.")

    def _rate_limit(self, cache_key: str) -> None:
        """Enforces rate limiting to prevent MT5 API overload."""
        with self._rate_limit_lock:
            now = time.time()
            last_time = self._last_call_time.get(cache_key, 0)
            elapsed = now - last_time
            
            if elapsed < self.MIN_CALL_INTERVAL:
                sleep_time = self.MIN_CALL_INTERVAL - elapsed
                time.sleep(sleep_time)
            
            self._last_call_time[cache_key] = time.time()

    def fetch_candles(self, 
                      symbol: str, 
                      timeframe: str, 
                      count: int = 500, 
                      force_refresh: bool = False) -> CandleArray:
        """
        Retrieves candle data using an incremental update strategy with STRICT fidelity.
        """
        now = time.time()
        cache_key = f"{symbol}_{timeframe}"
        ttl = self.CACHE_TTL.get(timeframe, 60)
        
        cached = self._cache.get(cache_key)
        
        # 1. Immediate Cache Return (Freshness Check)
        if not force_refresh and cached and (now - cached['timestamp']) < (ttl / 2):
            return cached['array']

        if mt5 is None:
            return CandleArray.from_dicts([])

        # 2. Logic: Fetch full history
        try:
            mt5_tf = self._get_mt5_timeframe(timeframe)
            if mt5_tf is None:
                return CandleArray.from_dicts([])

            # MT5 Lock management (Institutional standard)
            if self.lock: self.lock.acquire()
            try:
                if not mt5.symbol_select(symbol, True):
                    self.logger.error(f"Symbol {symbol} not found in MT5.")
                    return CandleArray.from_dicts([])
                
                rates = mt5.copy_rates_from_pos(symbol, mt5_tf, 0, count)
            finally:
                if self.lock: self.lock.release()

            if rates is None or len(rates) == 0:
                return cached['array'] if cached else CandleArray.from_dicts([])

            # 3. STRICT Fidelity Check (Step 7)
            spreads = rates['spread']
            if np.all(spreads == 0) or np.max(spreads) == 0:
                self.logger.critical(f"DATA REJECTED: OHLC-only data detected for {symbol} ({timeframe}). Spread required.")
                raise ValueError("STRICT_DATA_VIOLATION: Bid/Ask spread data is mandatory for V4 execution.")

            # 4. Conversion to CandleArray
            array = CandleArray(
                time=rates['time'].astype(np.int64),
                open=rates['open'].astype(float),
                high=rates['high'].astype(float),
                low=rates['low'].astype(float),
                close=rates['close'].astype(float),
                tick_volume=rates['tick_volume'].astype(np.int64),
                spread=rates['spread'].astype(np.int64)
            )

            # 5. Cache update
            self._cache[cache_key] = {
                "data": rates, # raw rates for merge if needed
                "array": array,
                "timestamp": now
            }
            return array

        except Exception as e:
            self.logger.exception(f"Critical Data Fetch Error ({symbol} {timeframe}): {e}")
            return cached['array'] if cached else CandleArray.from_dicts([])

    def _merge_candles(self, existing: List[Dict], new: List[Dict], max_len: int) -> List[Dict]:
        """Deduplicates and merges new candles into existing history."""
        combined = {c['time']: c for c in existing}
        for c in new:
            combined[c['time']] = c
        
        sorted_times = sorted(combined.keys())
        return [combined[t] for t in sorted_times[-max_len:]]

    def _get_mt5_timeframe(self, tf: str):
        if not mt5: return None
        mapping = {
            "M1": mt5.TIMEFRAME_M1,
            "M5": mt5.TIMEFRAME_M5,
            "M15": mt5.TIMEFRAME_M15,
            "H1": mt5.TIMEFRAME_H1,
            "D1": mt5.TIMEFRAME_D1
        }
        return mapping.get(tf)

    def get_symbol_info(self, symbol: str) -> Optional[dict]:
        """Institutional Symbol Data Retrieval."""
        if mt5 is None: return None
        try:
            if self.lock: self.lock.acquire()
            try:
                info = mt5.symbol_info(symbol)
            finally:
                if self.lock: self.lock.release()
                
            if info is None: return None
            return {
                "point": info.point,
                "digits": info.digits,
                "tick_value": info.trade_tick_value,
                "contract_size": info.trade_contract_size,
                "min_lot": info.volume_min,
                "max_lot": info.volume_max,
                "lot_step": info.volume_step,
                "spread": info.spread,
                "bid": info.bid,
                "ask": info.ask,
            }
        except Exception:
            return None

    def fetch_candles_range(self, symbol: str, timeframe: str, date_from: datetime.datetime, date_to: datetime.datetime) -> CandleArray:
        """Fetches institutional data for a specific date range."""
        if mt5 is None: return CandleArray.from_dicts([])
        try:
            mt5_tf = self._get_mt5_timeframe(timeframe)
            if self.lock: self.lock.acquire()
            try:
                rates = mt5.copy_rates_range(symbol, mt5_tf, date_from, date_to)
            finally:
                if self.lock: self.lock.release()
                
            if rates is None or len(rates) == 0:
                return CandleArray.from_dicts([])

            return CandleArray(
                time=rates['time'].astype(np.int64),
                open=rates['open'].astype(float),
                high=rates['high'].astype(float),
                low=rates['low'].astype(float),
                close=rates['close'].astype(float),
                tick_volume=rates['tick_volume'].astype(np.int64),
                spread=rates['spread'].astype(np.int64)
            )
        except Exception:
            return CandleArray.from_dicts([])

    def fetch_ticks_range(self, symbol: str, date_from: datetime.datetime, date_to: datetime.datetime) -> List[dict]:
        """Fetches raw institutional ticks (Bid/Ask/Flags)."""
        if mt5 is None: return []
        try:
            if self.lock: self.lock.acquire()
            try:
                ticks = mt5.copy_ticks_range(symbol, date_from, date_to, mt5.COPY_TICKS_ALL)
            finally:
                if self.lock: self.lock.release()
                
            if ticks is None: return []
            return [
                {"time": int(t[0]), "bid": float(t[1]), "ask": float(t[2]), "last": float(t[3]), "flags": int(t[4])}
                for t in ticks
            ]
        except Exception:
            return []

    def validate_integrity(self, array: CandleArray, timeframe: str) -> bool:
        """Institutional Check for data gaps."""
        if len(array) < 2: return True
        
        tf_seconds = {"M1": 60, "M5": 300, "M15": 900, "H1": 3600, "D1": 86400}.get(timeframe, 300)
        diffs = np.diff(array.time)
        # Allow gaps up to 3 intervals or weekend gaps (48h)
        gaps = np.where((diffs > tf_seconds * 3) & (diffs < 172800))[0]
        
        if len(gaps) > 0:
            self.logger.warning(f"Data Gaps detected in {timeframe}: {len(gaps)} gaps found.")
            return False
        return True

if __name__ == "__main__":
    # Standalone Simulation Mode
    logging.basicConfig(level=logging.INFO)
    handler = SourceHandler()
    
    # Mock data injection for test
    mock_data = [
        {"time": i*60, "open": 1.10, "high": 1.11, "low": 1.09, "close": 1.105, "tick_volume": 100}
        for i in range(10)
    ]
    array = CandleArray.from_dicts(mock_data)
    
    print("\n--- SourceHandler Standalone Test ---")
    print(f"Candle Count: {len(array)}")
    print(f"Integrity Check (M1): {handler.validate_integrity(array, 'M1')}")
    
    # Test Gap Detection
    mock_data[5]['time'] += 500 # Inject gap
    array_gap = CandleArray.from_dicts(mock_data)
    print(f"Integrity Check (With Gap): {handler.validate_integrity(array_gap, 'M1')}")
