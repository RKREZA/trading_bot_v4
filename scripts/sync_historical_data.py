import os
import sys
import logging
import time
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv

load_dotenv()

# Ensure we can import from the core directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.connection import MT5Connection
import MetaTrader5 as mt5
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("data_sync")

def sync_data(symbol: str, days: int = 90):
    """Downloads historical data from MT5 and saves it to the Parquet cache."""
    conn = MT5Connection()
    if not conn.connect():
        logger.error("Failed to connect to MT5. Ensure the terminal is open and credentials are correct in .env.")
        return

    utc_to = datetime.now(timezone.utc)
    utc_from = utc_to - timedelta(days=days)

    logger.info(f"Starting sync for {symbol} from {utc_from.strftime('%Y-%m-%d')} to {utc_to.strftime('%Y-%m-%d')} ({days} days)")

    timeframes = {
        "M1": mt5.TIMEFRAME_M1,
        "M5": mt5.TIMEFRAME_M5,
        "M15": mt5.TIMEFRAME_M15,
        "H1": mt5.TIMEFRAME_H1,
        "D1": mt5.TIMEFRAME_D1
    }

    cache_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data_cache", symbol)
    os.makedirs(cache_dir, exist_ok=True)

    for tf_name, tf_const in timeframes.items():
        logger.info(f"Fetching {tf_name} data...")
        
        with conn.MT5_LOCK:
            rates = mt5.copy_rates_range(symbol, tf_const, utc_from, utc_to)

        if rates is None or len(rates) == 0:
            logger.warning(f"No data returned for {tf_name}. Broker limits history this far back.")
            continue

        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df['tick_volume'] = df['tick_volume'].astype(float)
        df['spread'] = df['spread'].astype(float)
        
        file_path = os.path.join(cache_dir, f"{tf_name}.parquet")
        df.to_parquet(file_path, engine='pyarrow', index=False)
        
        logger.info(f"Successfully saved {len(df)} rows to {file_path}")
        time.sleep(1) # Give MT5 terminal a quick breather

    conn.disconnect()
    logger.info(f"Data sync for {symbol} complete. Ready for Backtesting.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="MT5 Data Sync Tool")
    parser.add_argument("--symbol", type=str, default="XAUUSDm", help="Target Symbol")
    parser.add_argument("--days", type=int, default=90, help="Days of history to download")
    
    args = parser.parse_args()
    sync_data(args.symbol, args.days)