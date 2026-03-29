import sys
import argparse
import json
import logging
import os
import traceback
from dotenv import load_dotenv
load_dotenv()
from datetime import datetime
import pandas as pd

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from core.logger import setup_logging
from core.connection import MT5Connection
from core.data_fetcher import DataFetcher
from core.strategy_engine import StrategyEngine
from core.backtester import BacktestEngine
from core.validation import ValidationSuite
from core.walk_forward import WalkForwardValidation
print("DEBUG: All imports completed.")

def load_config(path):
    print(f"DEBUG: Opening config: {path}")
    with open(path, 'r') as f:
        data = json.load(f)
        return data

def main():
    print(f"DEBUG: main() started. sys.argv={sys.argv}")
    try:
        setup_logging()
        logger = logging.getLogger("trading_bot.research")
        
        parser = argparse.ArgumentParser(description="Research-Grade Backtesting System")
        print("DEBUG: Using hardcoded symbol XAUUSDm...")
        # args = parser.parse_args()
        class Args:
            symbol = "XAUUSDm"
            config = "config/backtest_config.json"
            strategy_config = "config.json"
            full = False
            walk_forward = True
        args = Args()
        print(f"DEBUG: Args set to: {args.symbol}")
        
        bt_config = load_config(args.config)
        strat_config = load_config(args.strategy_config)
        print(f"DEBUG: Configs loaded. Symbol={args.symbol}")
        
        strategy = StrategyEngine(strat_config, None) 
        print("DEBUG: StrategyEngine initialized.")

        conn = MT5Connection()
        if not conn.connect():
            print("ERROR: MT5 Connect failed")
            return
        
        fetcher = DataFetcher()
        print(f"DEBUG: Fetching {args.symbol}...")
        h4 = fetcher.fetch_candles(args.symbol, "H4", 1000)
        h1 = fetcher.fetch_candles(args.symbol, "H1", 2000)
        m30 = fetcher.fetch_candles(args.symbol, "M30", 5000)
        m5 = fetcher.fetch_candles(args.symbol, "M5", 15000)
        d1 = fetcher.fetch_candles(args.symbol, "D1", 500)
        print("DEBUG: Data fetching complete. Explicitly disconnecting MT5 before research...")
        conn.disconnect()
        print("DEBUG: MT5 disconnected. Proceeding with research suite.")
        
        if not all([h4, h1, m30, m5, d1]):
            print("ERROR: Insufficient data")
            return
            
        print("DEBUG: Running Walk-Forward...")
        wf = WalkForwardValidation(strat_config, strategy)
        wf_results = wf.run_validation(args.symbol, h4, h1, m30, m5, d1)
        
        print("\nRES: Successful WFV run.")
        for res in wf_results:
            print(f"Window: {res['window']} | OOS Profit: ${res['oos_metrics'].get('net_profit', 0)}")

    except Exception as e:
        print(f"FATAL: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()
