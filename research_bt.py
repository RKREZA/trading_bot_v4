import argparse
import json
import logging
import os
import sys
from datetime import datetime

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

def load_config(path):
    with open(path, 'r') as f:
        return json.load(f)

def main():
    setup_logging()
    logger = logging.getLogger("trading_bot.research")
    
    parser = argparse.ArgumentParser(description="Research-Grade Backtesting System")
    parser.add_argument("--symbol", type=str, default="XAUUSD", help="Symbol to test")
    parser.add_argument("--config", type=str, default="config/backtest_config.json", help="Backtest config path")
    parser.add_argument("--strategy_config", type=str, default="config.json", help="Strategy config path")
    parser.add_argument("--full", action="store_true", help="Run full validation and walk-forward")
    args = parser.parse_args()
    
    bt_config = load_config(args.config)
    strat_config = load_config(args.strategy_config)
    
    # Initialize Strategy
    strategy = StrategyEngine(strat_config, None) # No logger for dashboard needed here
    
    # 1. FETCH DATA
    conn = MT5Connection()
    if not conn.connect():
        logger.error("Failed to connect to MT5")
        return
    
    fetcher = DataFetcher()
    logger.info(f"Fetching data for {args.symbol}...")
    h4 = fetcher.fetch_candles(args.symbol, "H4", 1000)
    m30 = fetcher.fetch_candles(args.symbol, "M30", 5000)
    m15 = fetcher.fetch_candles(args.symbol, "M15", 10000)
    conn.disconnect()
    
    if not h4 or not m30 or not m15:
        logger.error("Insufficient data fetched")
        return
    
    # 2. RUN BASE BACKTEST
    logger.info("Running base backtest...")
    tester = BacktestEngine(bt_config, strategy)
    base_results = tester.run(args.symbol, h4, m30, m15, quiet=True)
    
    print("\n" + "="*50)
    print(f"BASE BACKTEST RESULTS: {args.symbol}")
    print("="*50)
    for k, v in base_results.items():
        if k != 'equity_curve' and k != 'trades':
            print(f"{k.replace('_', ' ').title():<20}: {v}")
    print("="*50)
    
    if args.full:
        # 3. RUN VALIDATION SUITE
        logger.info("Running validation suite (stress tests)...")
        validator = ValidationSuite(bt_config, strategy)
        val_report = validator.run_all_tests(args.symbol, h4, m30, m15)
        
        print("\nVALIDATION REPORT:")
        print(f"STATUS: {val_report['status']}")
        if val_report['warnings']:
            for w in val_report['warnings']:
                print(f"  [!] {w}")
        else:
            print("  [+] All robustness checks passed")
            
        # 4. RUN WALK-FORWARD
        logger.info("Running walk-forward validation...")
        wf = WalkForwardValidation(bt_config, strategy)
        wf_results = wf.run_validation(args.symbol, h4, m30, m15)
        
        print("\nWALK-FORWARD CONSISTENCY:")
        for i, res in enumerate(wf_results):
            perf = res['performance']
            print(f"  Window {i+1}: Net Profit: {perf['net_profit']:>8} | Sharpe: {perf['sharpe_ratio']:>5}")

if __name__ == "__main__":
    main()
