import sys
import os
sys.path.append(os.getcwd())

import numpy as np
import pandas as pd
import logging
from datetime import datetime, timezone
from strategies.smart_mean_reversion import SmartMeanReversionStrategy
from core.common.types import CandleArray
from core.base_strategy import MarketData
from core.config.loader import ConfigLoader

# 🔍 Forensic Audit Logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger("MR_Value_Verification")

def run_mr_test():
    """Runs high-fidelity verification of the new Value Area Mean Reversion engine."""
    loader = ConfigLoader()
    config = loader.get_symbol_config("XAUUSDm")
    
    # 🏗️ Force institutional settings for verification
    # ConfigLoader flattens the 'strategies' block into the root of the merged config
    strat_cfg = config.get("SmartMeanReversion", {})
    if not strat_cfg:
        logger.warning("SmartMeanReversion config not found in merged config. Checking ['strategies'] block...")
        strat_cfg = config.get("strategies", {}).get("SmartMeanReversion", {})
    
    strat_cfg["enabled"] = True
    
    strat = SmartMeanReversionStrategy("mr_value_v1", config)
    
    logger.info("=" * 60)
    logger.info(" INSTITUTIONAL VALUE AREA MR VERIFICATION - XAUUSDm ")
    logger.info("=" * 60)
    
    # 📂 Load 1-year dataset
    try:
        m5_df = pd.read_parquet("data_cache/XAUUSDm/M5.parquet")
        m15_df = pd.read_parquet("data_cache/XAUUSDm/M15.parquet")
        h1_df = pd.read_parquet("data_cache/XAUUSDm/H1.parquet")
        d1_df = pd.read_parquet("data_cache/XAUUSDm/D1.parquet")
    except Exception as e:
        logger.error(f"Data loading failed: {e}")
        return

    # Prepare CandleArrays
    m5 = CandleArray(m5_df.time.values, m5_df.open.values, m5_df.high.values, m5_df.low.values, m5_df.close.values, m5_df.tick_volume.values, m5_df.spread.values)
    m15 = CandleArray(m15_df.time.values, m15_df.open.values, m15_df.high.values, m15_df.low.values, m15_df.close.values, m15_df.tick_volume.values, m15_df.spread.values)
    h1 = CandleArray(h1_df.time.values, h1_df.open.values, h1_df.high.values, h1_df.low.values, h1_df.close.values, h1_df.tick_volume.values, h1_df.spread.values)
    d1 = CandleArray(d1_df.time.values, d1_df.open.values, d1_df.high.values, d1_df.low.values, d1_df.close.values, d1_df.tick_volume.values, d1_df.spread.values)

    signals = []
    rejections = {}
    
    # Simulation loop
    start_idx = 1000 # Warm up
    # To speed up, we step every 4 bars (20 min) but check the signal logic
    for i in range(start_idx, len(m5), 2):
        t = m5.time[i]
        dt = datetime.fromtimestamp(t, tz=timezone.utc)
        
        # Sliced data for causal isolation
        m5_slice = m5[:i+1]
        m15_slice = m15[m15.time <= t]
        h1_slice = h1[h1.time <= t]
        d1_slice = d1[d1.time <= t]
        
        md = MarketData(
            symbol="XAUUSDm",
            htf_candles=h1_slice,
            m15_candles=m15_slice,
            m5_candles=m5_slice,
            d1_candles=d1_slice,
            current_price=m5.close[i],
            bid=m5.close[i],
            ask=m5.close[i] + 0.02,
            spread=0.02,
            point=0.01,
            session="GLOBAL",
            timestamp=dt
        )
        
        sig = strat.generate_signal(md)
        if sig:
            signals.append({
                "time": dt,
                "dir": sig.direction,
                "price": sig.price,
                "reasons": sig.reasons
            })
            logger.info(f"🚀 SIGNAL [{dt}] {sig.direction} @ {sig.price:.2f} | Reasons: {sig.reasons} | TP1: {sig.tp1_price:.2f} | TP2: {sig.tp2_price:.2f}")
        else:
            reason = getattr(strat, "last_rejection_reason", "Unknown")
            rejections[reason] = rejections.get(reason, 0) + 1

    logger.info("-" * 60)
    logger.info(f"Total Signals: {len(signals)}")
    logger.info(f"Rejection Summary: {dict(sorted(rejections.items(), key=lambda x: x[1], reverse=True)[:5])}")
    logger.info("-" * 60)

if __name__ == "__main__":
    run_mr_test()
