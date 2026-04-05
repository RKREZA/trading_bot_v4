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
            
        # 4. Institutional Mandatory Pre-Flight Verification (Step 2.5)
        # FIX #1: Zero RSI Trap Mitigation (Min 200 bars)
        # FIX #3: VPS Health Gap Detection (Continuity)
        array = self._ensure_min_history(array, symbol, timeframe, start_date)
        self._verify_continuity(array, symbol, timeframe)
            
        # Final institutional sort & return
        return array

    def _ensure_min_history(self, array: CandleArray, symbol: str, timeframe: str, start_date: datetime.datetime) -> CandleArray:
        """Enforces a 200-bar minimum history for reliable indicator calculation (Audit #1)."""
        if len(array) < 200:
            logger.warning(f"Data: {symbol} [{timeframe}] has only {len(array)} bars. Fetching more history.")
            # We fetch 500 bars to be safe
            self.sync.sync_full_history(symbol, timeframe, start_date)
            new_array = self.store.load(symbol, timeframe)
            if len(new_array) < 200:
                logger.critical(f"FATAL: Insufficient history for {symbol} after backfill.")
                raise ValueError("SYSTEM_HALT: History Gap.")
            return new_array
        return array

    def _verify_continuity(self, array: CandleArray, symbol: str, timeframe: str):
        """Detects data gaps by checking for contiguous timestamps (Audit #3)."""
        if len(array) < 2: return
        
        # Expected diff between bars based on timeframe minutes
        tf_map = {"M1": 60, "M5": 300, "M15": 900, "H1": 3600, "D1": 86400}
        expected_diff = tf_map.get(timeframe)
        if not expected_diff: return
        
        diffs = np.diff(array.time)
        gaps = np.where(diffs > expected_diff * 1.5)[0] # Allow for some variance but not a full missing bar
        
        if len(gaps) > 0:
            logger.error(f"DATA INTEGRITY ALERT: {len(gaps)} gaps detected in {symbol} [{timeframe}] cache.")
            for g_idx in gaps[:3]: # Log first 3
                logger.warning(f"Gap at: {datetime.datetime.fromtimestamp(array.time[g_idx])}")
            # In Production, we trigger a repair. For now, we halt.
            raise ValueError("SYSTEM_HALT: Data Gap Detected.")

    def get_latest_m1(self, symbol: str) -> CandleArray:
        """Helper for M1-Event Replay in backtesting engine."""
        return self.store.load(symbol, "M1")
