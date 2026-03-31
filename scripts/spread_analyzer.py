
import os
import sys
import logging
from datetime import datetime, timedelta, timezone
import numpy as np
import pandas as pd
from tabulate import tabulate

# Add the project root to sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from core.connection import MT5Connection
from core.data_fetcher import DataFetcher
import MetaTrader5 as mt5
from dotenv import load_dotenv

# Load credentials from .env
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("spread_analyzer")

def analyze_spreads(symbol: str = "XAUUSDm", days: int = 7):
    connection = MT5Connection()
    if not connection.connect():
        logger.error("Failed to connect to MT5")
        return

    fetcher = DataFetcher()
    
    end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(days=days)
    
    logger.info(f"Fetching ticks for {symbol} from {start_date} to {end_date}...")
    ticks = fetcher.fetch_ticks_range(symbol, start_date, end_date)
    
    if not ticks:
        logger.error(f"No ticks found for {symbol}")
        connection.disconnect()
        return

    df = pd.DataFrame(ticks)
    df['datetime'] = pd.to_datetime(df['time'], unit='s')
    df['spread'] = (df['ask'] - df['bid'])
    
    # Get symbol point for conversion (e.g. 0.01 for gold)
    with MT5Connection.MT5_LOCK:
        info = mt5.symbol_info(symbol)
    point = info.point if info else 0.01
    
    df['spread_points'] = df['spread'] / point
    
    # Session classification (UTC)
    # Tokyo: 0:00 - 8:00 UTC
    # London: 8:00 - 14:00 UTC
    # London/NY: 14:00 - 17:00 UTC
    # New York: 17:00 - 22:00 UTC
    
    def get_session(dt):
        h = dt.hour
        if 0 <= h < 8: return "TOKYO"
        if 8 <= h < 14: return "LONDON"
        if 14 <= h < 17: return "LONDON/NY"
        if 17 <= h < 22: return "NEW_YORK"
        return "OFF_HOURS"

    df['session'] = df['datetime'].apply(get_session)
    
    stats = []
    for session in ["LONDON", "LONDON/NY", "NEW_YORK", "TOKYO", "OFF_HOURS"]:
        sess_df = df[df['session'] == session]
        if sess_df.empty: continue
        
        mean_s = sess_df['spread_points'].mean()
        median_s = sess_df['spread_points'].median()
        p95_s = sess_df['spread_points'].quantile(0.95)
        
        stats.append([
            session,
            f"{mean_s:.2f}",
            f"{median_s:.2f}",
            f"{p95_s:.2f}",
            len(sess_df)
        ])
    
    print(f"\n--- Spread Analysis for {symbol} (Last {days} Days) ---")
    print(tabulate(stats, headers=["Session", "Mean (pts)", "Median (pts)", "95th % (pts)", "Ticks"], tablefmt="fancy_grid"))
    
    # Calculate recommended multipliers relative to London
    london_mean = df[df['session'] == "LONDON"]['spread_points'].mean()
    if london_mean > 0:
        print("\n--- Recommended Session Multipliers (vs London) ---")
        mults = []
        for s in ["LONDON", "LONDON/NY", "NEW_YORK", "TOKYO"]:
            s_mean = df[df['session'] == s]['spread_points'].mean()
            mult = s_mean / london_mean
            mults.append([s, f"{mult:.2f}x", f"({s_mean:.2f} pts)"])
        print(tabulate(mults, headers=["Session", "Multiplier", "Mean Spread"], tablefmt="simple"))

    connection.disconnect()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", type=str, default="XAUUSDm")
    parser.add_argument("--days", type=int, default=7)
    args = parser.parse_args()
    
    analyze_spreads(args.symbol, args.days)
