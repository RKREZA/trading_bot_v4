import os
import time
import pandas as pd
import fastparquet
import logging
from typing import Optional, List
from core.common.types import CandleArray

logger = logging.getLogger("trading_bot.data.parquet")

class ParquetStore:
    """
    High-Performance Parquet-based Candle Storage.
    Ensures zero data loss and atomic writes for Windows VPS stability.
    """
    
    def __init__(self, base_path: str = "data_cache"):
        self.base_path = base_path
        os.makedirs(self.base_path, exist_ok=True)

    def get_path(self, symbol: str, timeframe: str) -> str:
        symbol_dir = os.path.join(self.base_path, symbol.replace("/", "_"))
        os.makedirs(symbol_dir, exist_ok=True)
        return os.path.join(symbol_dir, f"{timeframe}.parquet")

    def save(self, symbol: str, timeframe: str, array: CandleArray):
        """
        Atomic Save of CandleArray to Parquet.
        Uses a temporary file to prevent corruption during Windows VPS crashes.
        """
        path = self.get_path(symbol, timeframe)
        temp_path = path + ".tmp"
        
        try:
            df = pd.DataFrame({
                "time": array.time,
                "open": array.open,
                "high": array.high,
                "low": array.low,
                "close": array.close,
                "tick_volume": array.tick_volume,
                "spread": array.spread
            })
            
            # Sort by time ensuring strict chronology
            df = df.sort_values("time").drop_duplicates("time")
            
            df.to_parquet(temp_path, engine="fastparquet", index=False)
            
            # Atomic swap on Windows with retry logic
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    if os.path.exists(path):
                        os.replace(temp_path, path)
                    else:
                        os.rename(temp_path, path)
                    break
                except PermissionError:
                    if attempt == max_retries - 1:
                        raise
                    logger.warning(f"File {path} in use, retrying ({attempt+1}/{max_retries})...")
                    time.sleep(0.5)
            
            logger.debug(f"Saved {len(df)} candles for {symbol} {timeframe} to {path}")
            
        except Exception as e:
            logger.critical(f"FATAL: Failed to save Parquet cache for {symbol}: {e}")
            if os.path.exists(temp_path):
                os.remove(temp_path)
            raise

    def load(self, symbol: str, timeframe: str) -> Optional[CandleArray]:
        """Loads candles from the local Parquet store."""
        path = self.get_path(symbol, timeframe)
        if not os.path.exists(path):
            return None
            
        try:
            df = pd.read_parquet(path, engine="fastparquet")
            if df.empty:
                return None
                
            return CandleArray(
                time=df['time'].values,
                open=df['open'].values,
                high=df['high'].values,
                low=df['low'].values,
                close=df['close'].values,
                tick_volume=df['tick_volume'].values,
                spread=df['spread'].values
            )
        except Exception as e:
            logger.error(f"Error loading Parquet cache for {symbol}: {e}")
            return None

    def get_last_timestamp(self, symbol: str, timeframe: str) -> int:
        """Efficiently retrieves the latest timestamp in the cache."""
        path = self.get_path(symbol, timeframe)
        if not os.path.exists(path):
            return 0
        try:
            # We use FastParquet to read only the metadata if possible, 
            # but reading the last few rows is also fast.
            df = pd.read_parquet(path, columns=["time"], engine="fastparquet")
            return int(df["time"].max()) if not df.empty else 0
        except Exception:
            return 0
