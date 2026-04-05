import logging
import datetime
from core.data.source_handler import SourceHandler
from core.data.parquet_store import ParquetStore
from core.data.sync_engine import SyncEngine
from core.common.types import CandleArray

logger = logging.getLogger("trading_bot.data_manager")

class DataManager:
    """
    V4-ULTRA Production Data Orchestrator.
    Manages MT5 <-> Local Parquet Cache lifecycle.
    Ensures 'Zero Data Inconsistency' before any backtest or strategy cycle.
    """
    
    def __init__(self, config: dict):
        self.config = config
        self.source = SourceHandler()
        self.store = ParquetStore(base_path=config.get("data_cache_path", "data_cache"))
        self.sync = SyncEngine(self.source, self.store)

    def prepare_data(self, symbol: str, timeframe: str, start_date: datetime.datetime) -> CandleArray:
        """
        Highest Priority: Flawless Data Prep.
        1. Sync latest MT5 data to local cache.
        2. Detect and fill gaps.
        3. Load FULL dataset for backtesting simulation.
        """
        logger.info(f"Preparing institutional-grade data for {symbol} ({timeframe})...")
        
        # 1. Sync & Fill Gaps (Step 2.5)
        self.sync.update_incremental(symbol, timeframe)
        
        # 2. Check if we need a deep sync (no cache or cache starts too late)
        last_cached = self.store.get_last_timestamp(symbol, timeframe)
        # If cache is missing or last timestamp is before requested start, 
        # we might need to backfill history.
        # But SyncEngine.update_incremental already handles deep sync if last_cached == 0.
        
        # 3. Load & Validate
        array = self.store.load(symbol, timeframe)
        if array is None or len(array) == 0:
            logger.critical(f"FATAL: Source data empty for {symbol} after sync attempt.")
            raise ValueError("SYSTEM_HALT: Sync Failure.")
            
        # 4. Mandatory Pre-Flight Verification (Step 2.5)
        # Verify that the array reaches the requested range
        first_time = array.time[0]
        if first_time > start_date.timestamp():
            logger.warning(f"Cache starts later than requested ({datetime.datetime.fromtimestamp(first_time)})")
            # Deep backfill required
            self.sync.sync_full_history(symbol, timeframe, start_date)
            array = self.store.load(symbol, timeframe)
            
        # Sort and return
        return array

    def get_latest_m1(self, symbol: str) -> CandleArray:
        """Helper for M1-Event Replay in backtesting engine."""
        return self.store.load(symbol, "M1")
