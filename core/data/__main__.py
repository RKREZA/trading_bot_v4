import logging
from core.data.source_handler import SourceHandler
from core.common.types import CandleArray

def run_data_diagnostic():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    logger = logging.getLogger("data_diag")
    
    logger.info("Starting Institutional Data Diagnostic...")
    
    handler = SourceHandler()
    
    # 1. Simulation of Mock Data (Simulated Mode)
    mock_candles = [
        {"time": 1000 + (i * 60), "open": 1.1000, "high": 1.1050, "low": 1.0950, "close": 1.1025, "tick_volume": 120}
        for i in range(100)
    ]
    
    # Inject Gap
    mock_candles[50]['time'] += 1000 
    
    array = CandleArray.from_dicts(mock_candles)
    is_ok = handler.validate_integrity(array, "M1")
    
    logger.info(f"Integrity Check Result: {'PASSED' if is_ok else 'FAILED (Gaps detected - Expected for Mock)'}")
    
    # 2. Incrementality Test (Logic Check)
    cached_data = mock_candles[:50]
    new_data = mock_candles[45:60] # Overlaying
    
    merged = handler._merge_candles(cached_data, new_data, max_len=60)
    logger.info(f"Merge Integrity: {len(merged)} candles (Should be 60)")
    
    if len(merged) == 60:
        logger.info("SOURCE_HANDLER: Standalone Diagnostic PASSED.")
    else:
        logger.error("SOURCE_HANDLER: Standalone Diagnostic FAILED.")

if __name__ == "__main__":
    run_data_diagnostic()
