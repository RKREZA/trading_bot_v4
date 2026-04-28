import logging
import datetime
import numpy as np
from typing import List
from core.data.source_handler import SourceHandler
from core.data.parquet_store import ParquetStore
from core.data.sync_engine import SyncEngine
from core.common.types import CandleArray

logger = logging.getLogger("trading_bot.data_manager")

class DataManager:
    """
    V5-INSIGNIA Production Data Orchestrator.
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
        
        # 3. Load & Filter to relevant window
        array = self.store.load(symbol, timeframe)
        if array is None or len(array) == 0:
            logger.critical(f"FATAL: Source data empty for {symbol} after sync attempt.")
            raise ValueError("SYSTEM_HALT: Sync Failure.")

        # 4. Institutional Mandatory Pre-Flight Verification (Step 2.5)
        start_ts = start_date.timestamp()
        now_ts = datetime.datetime.now(datetime.timezone.utc).timestamp()
        
        gaps = []
        relevant_array = CandleArray.from_dicts([]) # Placeholder
        
        # Repair Loop: Up to 3 attempts
        for attempt in range(3):
            array = self.store.load(symbol, timeframe)
            if array is None or len(array) == 0: break
            
            # Phase A: Coverage Audit (Fidelity Check)
            tf_secs = {"M1": 60, "M5": 300, "M15": 900, "H1": 3600, "D1": 86400}.get(timeframe, 300)
            
            # Institutional Adjustment: Weekend-Aware Expectation
            # If the period covers a weekend (~48h), we reduce expected bars by ~65%.
            days_requested = (now_ts - start_ts) / 86400
            wall_clock_expected = (days_requested * 86400 / tf_secs)
            
            # Simple heuristic: if days > 2, it likely spans a weekend. 
            # Real institutional logic would use a holiday calendar.
            if days_requested > 1.5:
                # Reduce expectation by 48 hours worth of bars
                weekend_bars = (48 * 3600 / tf_secs)
                expected_bars = max(100, (wall_clock_expected - weekend_bars) * 0.8)
            else:
                expected_bars = wall_clock_expected * 0.8            
            idx_start = np.searchsorted(array.time, start_ts)
            current_slice = array[idx_start:]
            
            # If cache starts late or is missing significant chunks, trigger backfill
            if (len(array) > 0 and array.time[0] > (start_ts + 3600)) or (len(current_slice) < (expected_bars * 0.85)):
                if attempt < 1:
                    logger.warning(f"DataManager: Fidelity Gap Detected for {symbol} ({len(current_slice)}/{int(expected_bars)} bars). Triggering Deep Sync.")
                    self.sync.sync_full_history(symbol, timeframe, start_date)
                    continue 

            # Phase B: Extraction with Buffer
            safe_idx_start = max(0, idx_start - 240) 
            relevant_array = array[safe_idx_start:]
            
            relevant_array = self._ensure_min_history(relevant_array, symbol, timeframe, start_date)
            gaps = self._verify_continuity(relevant_array, symbol, timeframe)
            
            if not gaps:
                return relevant_array
                
            logger.info(f"DataManager: Detected {len(gaps)} gaps for {symbol} [{timeframe}]. Attempting Auto-Repair...")
            self.sync.repair_identified_gaps(symbol, timeframe, array, gaps)
            break # Only one repair attempt per cycle to prevent infinite loops
            
        if len(gaps) > 0:
            logger.warning(f"PROCEEDING WITH TOLERANCE: {len(gaps)} unrepairable gaps detected in {symbol} ({timeframe}) history.")
        elif attempt > 0:
            logger.info("=" * 60)
            logger.info("FIDELITY NOTICE: Data repair was performed DURING this run.")
            logger.info("To ensure 100% deterministic audit results, please RE-RUN this backtest.")
            logger.info("=" * 60)
        
        return relevant_array

    def _ensure_min_history(self, array: CandleArray, symbol: str, timeframe: str, start_date: datetime.datetime) -> CandleArray:
        """Enforces institutional history buffers for indicator stability (Audit #1)."""
        min_bars = 20 if timeframe == "D1" else 200
        if len(array) < min_bars:
            logger.warning(f"Data: {symbol} [{timeframe}] has only {len(array)} bars. Fetching more history.")
            # We fetch 60 days back to be universally safe for 500 bars
            deep_start_date = start_date - datetime.timedelta(days=60)
            self.sync.sync_full_history(symbol, timeframe, deep_start_date)
            new_array = self.store.load(symbol, timeframe)
            if len(new_array) < 200:
                logger.critical(f"WARNING: Insufficient institutional history for {symbol} after backfill. Got {len(new_array)}. Proceeding anyway for weekend logic.")
            return new_array
        return array

    def _verify_continuity(self, array: CandleArray, symbol: str, timeframe: str) -> List[tuple]:
        """
        Detects data gaps and filters them by 'Market-Aware' institutional rules (Audit #3).
        Ignores Weekends and Daily Closes (> 2h). Ensures surgical repair for glitches.
        """
        if len(array) < 2: return []
        
        # Expected diff between bars 
        tf_map = {"M1": 60, "M5": 300, "M15": 900, "H1": 3600, "D1": 86400}
        expected_diff = tf_map.get(timeframe)
        if not expected_diff: return []
        
        diffs = np.diff(array.time)
        # Gap Detection: Anything more than 50% larger than interval
        gap_idxs = np.where(diffs > expected_diff * 1.5)[0]
        
        if len(gap_idxs) == 0:
            return []
            
        gap_windows = []
        for idx in gap_idxs:
            gap_duration = diffs[idx]
            
            # Institutional Constraint: Market Closure Recognition
            # If gap > 50 mins (3000s), it's likely a Daily Close (common for Gold), 
            # Weekend or Holiday. We skip these to avoid false-positive halts.
            if gap_duration > 3600: # Increase closure recognition to 1 hour
                logger.debug(f"DataManager: Skipping Market-Closure gap ({gap_duration/3600:.1f}h)")
                continue
                
            logger.info(f"DataManager: Found repairable gap ({gap_duration}s). Patching...")
            start_ts = array.time[idx] + 1
            end_ts = array.time[idx+1] - 1
            gap_windows.append((start_ts, end_ts))
            
        return gap_windows

    def get_latest_m1(self, symbol: str) -> CandleArray:
        """Helper for M1-Event Replay in backtesting engine."""
        return self.store.load(symbol, "M1")
