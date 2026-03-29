import argparse
import json
import logging
import os
import sys
from datetime import datetime
import numpy as np

from dotenv import load_dotenv

# Add project root to sys.path
sys.path.append(os.getcwd())

# Load environment variables
load_dotenv()

from core.logger import setup_logging
from core.connection import MT5Connection
from core.data_fetcher import DataFetcher
from core.backtester import BacktestEngine
from core.strategy_engine import StrategyEngine
from core.validation import ValidationSuite
from core.walk_forward import WalkForwardValidation

def main():
    parser = argparse.ArgumentParser(description="Trading Bot Robustness Validator")
    parser.add_argument("--symbol", type=str, default="XAUUSDm", help="Symbol to validate")
    parser.add_argument("--config", type=str, default="config.json", help="Path to config file")
    args = parser.parse_args()

    setup_logging(level=logging.INFO, console=True)
    logger = logging.getLogger("trading_bot.robustness")

    # 1. Load Config
    with open(args.config, "r") as f:
        config = json.load(f)

    # 2. Fetch Data
    conn = MT5Connection()
    conn.config = config
    if not conn.connect():
        logger.error("Failed to connect to MT5 for data fetching")
        return

    fetcher = DataFetcher()
    logger.info(f"Fetching data for {args.symbol}...")
    
    # Need enough data for 3m IS + 1m OOS (at least 4 months, preferably more)
    h4 = fetcher.fetch_candles(args.symbol, "H4", 2500)
    h1 = fetcher.fetch_candles(args.symbol, "H1", 9000)
    m30 = fetcher.fetch_candles(args.symbol, "M30", 18000)
    m5 = fetcher.fetch_candles(args.symbol, "M5", 54000) # ~6-7 months of M5
    d1 = fetcher.fetch_candles(args.symbol, "D1", 500)
    conn.disconnect()

    # 3. Base Backtest
    logger.info("Running base backtest...")
    strategy = StrategyEngine(config)
    tester = BacktestEngine(config, strategy)
    base_perf = tester.run(args.symbol, h4, h1, m30, m5, d1, quiet=True)
    
    # 4. Monte Carlo Simulation
    logger.info("Running Monte Carlo simulation (1000 iterations)...")
    val_suite = ValidationSuite(config, strategy)
    mc_results = val_suite.monte_carlo_equity(base_perf['trades'], iterations=1000)
    
    # 5. Walk-Forward Validation
    logger.info("Running Walk-Forward Validation (3m IS / 1m OOS)...")
    wf = WalkForwardValidation(config, strategy)
    wf_results = wf.run_validation(args.symbol, h4, h1, m30, m5, d1)
    
    # 6. Calculate Results
    is_sharpes = [w['is_metrics'].get('sharpe_ratio', 0) for w in wf_results]
    oos_sharpes = [w['oos_metrics'].get('sharpe_ratio', 0) for w in wf_results]
    
    avg_is_sharpe = np.mean(is_sharpes) if is_sharpes else 0
    avg_oos_sharpe = np.mean(oos_sharpes) if oos_sharpes else 0
    
    overfitting_score = 1 - (avg_oos_sharpe / avg_is_sharpe) if avg_is_sharpe > 0 else 1.0
    overfitting_score = max(0, overfitting_score)
    
    # 7. Final Report
    print("\n" + "="*50)
    print("ROBUSTNESS VALIDATION REPORT")
    print("="*50)
    print(f"Symbol:                {args.symbol}")
    print(f"Period:                {wf_results[0]['window'].split(' to ')[0]} to {wf_results[-1]['window'].split(' to ')[1]}")
    print("-"*50)
    print(f"Base Backtest PF:      {base_perf['profit_factor']:.2f}")
    print(f"Base Backtest Sharpe:  {base_perf['sharpe_ratio']:.2f}")
    print(f"Win Rate:              {base_perf['win_rate']:.1f}%")
    print("-"*50)
    print(f"Avg IS Sharpe:         {avg_is_sharpe:.2f}")
    print(f"Avg OOS Sharpe:        {avg_oos_sharpe:.2f}")
    print(f"Overfitting Score:     {overfitting_score:.2f}")
    print("-"*50)
    print(f"MC 95% Max DD:         {mc_results['p95_max_drawdown']:.1f}%")
    print(f"MC Mean Max DD:        {mc_results['mean_max_drawdown']:.1f}%")
    print("-"*50)
    
    # Recommendation
    recommendation = "REJECT"
    if overfitting_score < 0.3 and mc_results['p95_max_drawdown'] <= 30:
        recommendation = "LIVE (Ready for deployment)"
    elif overfitting_score < 0.5 and mc_results['p95_max_drawdown'] <= 40:
        recommendation = "PAPER (Further monitoring required)"
    
    print(f"RECOMMENDATION:        {recommendation}")
    print("="*50)

    # Save summary
    summary = {
        "symbol": args.symbol,
        "overfitting_score": overfitting_score,
        "avg_oos_sharpe": avg_oos_sharpe,
        "mc_p95_dd": mc_results['p95_max_drawdown'],
        "recommendation": recommendation,
        "timestamp": datetime.now().isoformat()
    }
    with open("robustness_summary.json", "w") as f:
        json.dump(summary, f, indent=4)

if __name__ == "__main__":
    main()
