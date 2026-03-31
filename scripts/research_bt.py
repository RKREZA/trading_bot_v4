print("CORE DEBUG: research_bt.py START")
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
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from core.logger import setup_logging
from core.connection import MT5Connection
from core.data_fetcher import DataFetcher
from core.strategy_engine import StrategyEngine
from core.backtester import BacktestEngine
from core.validation import ValidationSuite
from core.walk_forward import WalkForwardValidation

def load_config(path):
    print(f"DEBUG: Opening config: {path}")
    with open(path, 'r') as f:
        data = json.load(f)
        print(f"DEBUG: Config {path} loaded successfully.")
        return data

def main():
    print(f"DEBUG: research_bt.py main() started. sys.argv: {sys.argv}")
    try:
        setup_logging()
        logger = logging.getLogger("trading_bot.research")
        
        print("DEBUG: Setting up ArgumentParser...")
        parser = argparse.ArgumentParser(description="Research-Grade Backtesting System")
        parser.add_argument("--symbol", type=str, default="XAUUSD", help="Symbol to test")
        parser.add_argument("--config", type=str, default="config/backtest_config.json", help="Backtest config path")
        parser.add_argument("--strategy_config", type=str, default="config.json", help="Strategy config path")
        parser.add_argument("--full", action="store_true", help="Run full validation")
        parser.add_argument("--walk-forward", action="store_true", help="Run rolling walk-forward validation")
        
        print("DEBUG: Calling parse_args()...")
        args = parser.parse_args()
        print(f"DEBUG: Arguments successfully parsed: {args}")
        
        print(f"DEBUG: Loading bt_config from {args.config}...")
        bt_config = load_config(args.config)
        print(f"DEBUG: Loading strat_config from {args.strategy_config}...")
        strat_config = load_config(args.strategy_config)
        print(f"DEBUG: Loaded configs. Symbol: {args.symbol}")
        
        # Initialize Strategy
        try:
            print("DEBUG: Pre-Initializing StrategyEngine...")
            # Check if strat_config is valid
            if not isinstance(strat_config, dict):
                print(f"ERROR: strat_config is not a dict, it's a {type(strat_config)}")
            
            strategy = StrategyEngine(strat_config, None) 
            print("DEBUG: StrategyEngine initialized successfully.")
        except Exception as e:
            print(f"FATAL ERROR during StrategyEngine init: {e}")
            traceback.print_exc()
            return

        # 1. FETCH DATA
        print("DEBUG: Creating MT5Connection...")
        conn = MT5Connection()
        print("DEBUG: Attempting MT5 connect...")
        if not conn.connect():
            print("ERROR: Failed to connect to MT5")
            return
        
        try:
            print("DEBUG: Creating DataFetcher...")
            fetcher = DataFetcher()
            print(f"DEBUG: Fetching data for {args.symbol}...")
        except Exception as e:
            print(f"ERROR initializing DataFetcher: {e}")
            traceback.print_exc()
            return
            
        h4 = fetcher.fetch_candles(args.symbol, "H4", 1000)
        h1 = fetcher.fetch_candles(args.symbol, "H1", 2000)
        m30 = fetcher.fetch_candles(args.symbol, "M30", 5000)
        m5 = fetcher.fetch_candles(args.symbol, "M5", 15000)
        d1 = fetcher.fetch_candles(args.symbol, "D1", 500)
        conn.disconnect()
        
        logger.info(f"Data status: H4: {len(h4) if h4 else 0}, H1: {len(h1) if h1 else 0}, M30: {len(m30) if m30 else 0}, M5: {len(m5) if m5 else 0}, D1: {len(d1) if d1 else 0}")
        
        if not all([h4, h1, m30, m5, d1]):
            logger.error(f"Insufficient data fetched! H4: {len(h4) if h4 else 0}, H1: {len(h1) if h1 else 0}, M30: {len(m30) if m30 else 0}, M5: {len(m5) if m5 else 0}, D1: {len(d1) if d1 else 0}")
            return
        
        # 2. RUN BASE BACKTEST
        print("DEBUG: Initializing BacktestEngine...")
        tester = BacktestEngine(strat_config, strategy) # Using strat_config as base
        print(f"DEBUG: Starting base backtest for {args.symbol}...")
        base_results = tester.run(args.symbol, h4, h1, m30, m5, d1, quiet=True)
        print("DEBUG: Base backtest completed successfully.")
        
        print("\n" + "="*50)
        print(f"BASE BACKTEST RESULTS: {args.symbol}")
        print("="*50)
        for k, v in base_results.items():
            if k != 'equity_curve' and k != 'trades':
                print(f"{k.replace('_', ' ').title():<20}: {v}")
        print("="*50)
        
        if args.full or args.walk_forward:
            if args.full:
                # 3. RUN VALIDATION SUITE
                logger.info("Running validation suite (stress tests)...")
                validator = ValidationSuite(strat_config, strategy)
                val_report = validator.run_all_tests(args.symbol, h4, h1, m30, m5, d1)
                
                print("\nVALIDATION REPORT:")
                print(f"STATUS: {val_report['status']}")
                if val_report['warnings']:
                    for w in val_report['warnings']:
                        print(f"  [!] {w}")
                else:
                    print("  [+] All robustness checks passed")
                
            # 4. RUN WALK-FORWARD
            logger.info("Running walk-forward validation...")
            wf = WalkForwardValidation(strat_config, strategy)
            wf_results = wf.run_validation(args.symbol, h4, h1, m30, m5, d1)
            
            print("\nWALK-FORWARD CONSISTENCY:")
            print("-" * 80)
            print(f"{'Window':<25} | {'IS Sharpe':<10} | {'OOS Sharpe':<10} | {'OOS Profit':<10}")
            print("-" * 80)
            for res in wf_results:
                is_m = res['is_metrics']
                oos_m = res['oos_metrics']
                print(f"{res['window']:<25} | {is_m.get('sharpe_ratio', 0):>10.2f} | {oos_m.get('sharpe_ratio', 0):>10.2f} | ${oos_m.get('net_profit', 0):>10.2f}")
            
            # Save results for analyze_results.py
            with pd.option_context('display.max_rows', None, 'display.max_columns', None):
                with open("wf_results.json", "w") as f:
                    # Basic serialization for now
                    json.dump(wf_results, f, indent=4, default=str)
            print("-" * 80)

    except Exception:
        print("\nFATAL ERROR in research_bt.py:")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
