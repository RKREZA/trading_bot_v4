from dotenv import load_dotenv
load_dotenv()
import os
import sys
import logging
import datetime
import pandas as pd
import numpy as np
from datetime import timezone

# Add system path
sys.path.append(os.getcwd())

from core.logger import setup_logging
from core.data.source_handler import SourceHandler
from core.indicator_engine import IndicatorEngine
from core.base_strategy import MarketData
from strategies.trend_following import TrendFollowingStrategy
from core.common.types import CandleArray

def run_forensic_audit():
    setup_logging(console=True)
    logger = logging.getLogger("forensic_audit")
    
    symbol = "XAUUSDm"
    tf = "M15"
    
    # Load Data
    handler = SourceHandler()
    m15_data = handler.fetch_candles(symbol, tf, count=5000) # Last 5000 bars
    h1_data = handler.fetch_candles(symbol, "H1", count=2000)
    
    if m15_data is None or h1_data is None:
        logger.error("Failed to load data for audit.")
        return

    # Pre-calculate Indicators (Sync with Backtester)
    m15_data._indicators = IndicatorEngine.precalculate_all(symbol, tf, m15_data)
    h1_data._indicators = IndicatorEngine.precalculate_all(symbol, "H1", h1_data)
    
    from core import MT5Connection
    conn = MT5Connection()
    if not conn.connect():
        logger.error("Failed to connect to MT5 for audit.")
        return
        
    # Mock Config (Reflecting the BUGGY state in current system)
    config = {
        "strategies": {
            "TrendFollowing": {
                "enabled": True,
                "adx_threshold": 26,
                "min_confidence": 0.3, # Relaxed
                "allowed_sessions": ["GLOBAL", "LONDON", "NEW_YORK", "TOKYO"]
            }
        }
    }
    
    strat = TrendFollowingStrategy("TrendFollowing", config)
    
    # Rejection Stats
    stats = {}
    total_signals = 0
    
    logger.info(f"Starting Forensic Audit on {len(m15_data)} bars...")
    
    for i in range(100, len(m15_data)):
        m15_view = m15_data[0:i]
        h1_view = h1_data[0:i//4 + 1] # Simple approximation for audit
        
        # Sync limits
        m15_view.set_limit(i)
        h1_view.set_limit(len(h1_view))
        
        market_data = MarketData(
            symbol=symbol,
            htf_candles=h1_view,
            m15_candles=m15_view,
            m5_candles=m15_view, # Mock
            d1_candles=None,
            current_price=float(m15_view.close[-1]),
            bid=float(m15_view.close[-1]),
            ask=float(m15_view.close[-1] + m15_view.spread[-1]),
            spread=float(m15_view.spread[-1]),
            point=0.01,
            session="GLOBAL",
            timestamp=datetime.datetime.fromtimestamp(m15_view.time[-1], tz=timezone.utc)
        )
        
        signal = strat.generate_signal(market_data)
        
        if signal:
            total_signals += 1
            logger.info(f"SIGNAL FOUND at {market_data.timestamp}: {signal}")
        else:
            reason = getattr(strat, "last_rejection_reason", "No reason")
            stats[reason] = stats.get(reason, 0) + 1

    logger.info("==================================================")
    logger.info("FORENSIC REJECTION SUMMARY")
    logger.info("==================================================")
    for reason, count in sorted(stats.items(), key=lambda x: x[1], reverse=True):
        logger.info(f"{reason}: {count}")
    logger.info(f"TOTAL SIGNALS: {total_signals}")

if __name__ == "__main__":
    run_forensic_audit()
