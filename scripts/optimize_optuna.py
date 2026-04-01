import optuna
import json
import logging
import copy
import argparse
from datetime import datetime, timezone
import os
import sys
from dotenv import load_dotenv

# Add project root to sys.path
sys.path.append(os.getcwd())
load_dotenv()

from core.strategy_engine import StrategyEngine
from core.backtester import BacktestEngine
from core.connection import MT5Connection
from core.data_fetcher import DataFetcher

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("optuna_optimizer")

def objective(trial, config, h4, h1, m30, m5, d1):
    # Clone and modify config with trial parameters
    local_cfg = copy.deepcopy(config)
    
    # Ensure strategy_defaults exists
    if "strategy_defaults" not in local_cfg:
        local_cfg["strategy_defaults"] = {}
    
    sd = local_cfg["strategy_defaults"]
    
    # Strategy Parameters
    sd["min_confluence_score"] = trial.suggest_int("min_confluence_score", 2, 6)
    sd["min_confidence"] = trial.suggest_int("min_confidence", 50, 85)
    sd["sl_atr_buffer"] = trial.suggest_float("sl_atr_buffer", 0.3, 1.2, step=0.1)
    
    # RR and ATR Optimization
    sd["rr_ratio"] = trial.suggest_float("rr_ratio", 1.5, 4.0, step=0.5)
    sd["atr_period"] = trial.suggest_int("atr_period", 10, 20)
    
    sd["pullback_distance_pct"] = trial.suggest_float("pullback_distance_pct", 0.2, 0.8, step=0.1)
    
    # MFE Trailing Base
    if "trailing_stop" not in local_cfg:
        local_cfg["trailing_stop"] = {}
    local_cfg["trailing_stop"]["mfe_trail_base"] = trial.suggest_float("mfe_trail_base", 0.3, 0.7, step=0.1)

    # Initialize engines
    strategy = StrategyEngine(local_cfg, silent=True)
    backtester = BacktestEngine(local_cfg, strategy)
    
    # Run backtest
    results = backtester.run("XAUUSDm", h4, h1, m30, m5, d1, quiet=True)
    
    # Objective: (Net Profit * Sharpe * PF) / MaxDrawdown
    profit = results.get("net_profit", 0)
    sharpe = results.get("sharpe_ratio", 0)
    pf = results.get("profit_factor", 0)
    mdd = results.get("max_drawdown", 100) # Fix key from max_drawdown_pct to max_drawdown
    
    if mdd <= 0: mdd = 0.01 # Avoid div by zero
    if profit <= 0: return -abs(profit) # Penalize losses
    
    # Reward for higher win rate to avoid curve fitting to 1 lucky trade
    win_rate = results.get("win_rate", 0) / 100.0
    
    score = (profit * sharpe * pf * win_rate) / mdd
    return score

def main():
    parser = argparse.ArgumentParser(description="Trading Bot Optuna Optimizer")
    parser.add_argument("--trials", type=int, default=50, help="Number of optimization trials")
    parser.add_argument("--symbol", type=str, default="XAUUSDm", help="Symbol to optimize")
    args = parser.parse_args()

    # 1. Load Config
    with open("config.json", "r") as f:
        config = json.load(f)
    
    # 2. Fetch Data (Once)
    conn = MT5Connection()
    conn.config = config
    if not conn.connect():
        logger.error("Could not connect to MT5 for data fetching.")
        return
    
    fetcher = DataFetcher()
    logger.info(f"Fetching historical data for {args.symbol}...")
    
    # Fetch ample data for a meaningful optimization (Last 3 months approx)
    h4 = fetcher.fetch_candles(args.symbol, "H4", 1000)
    h1 = fetcher.fetch_candles(args.symbol, "H1", 3000)
    m30 = fetcher.fetch_candles(args.symbol, "M30", 6000)
    m5 = fetcher.fetch_candles(args.symbol, "M5", 20000)
    d1 = fetcher.fetch_candles(args.symbol, "D1", 500)
    
    conn.disconnect()
    
    if not all([h4, h1, m30, m5, d1]):
        logger.error("Failed to fetch all required timeframes.")
        return

    logger.info(f"Starting Optuna Study with {args.trials} trials...")
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, config, h4, h1, m30, m5, d1), n_trials=args.trials)

    logger.info("Optimization Complete!")
    logger.info(f"Best Score: {study.best_value}")
    logger.info(f"Best Params: {study.best_params}")
    
    # 3. Save optimized config
    best_cfg = copy.deepcopy(config)
    for k, v in study.best_params.items():
        if k == "mfe_trail_base":
            best_cfg.setdefault("trailing_stop", {})[k] = v
        else:
            best_cfg.setdefault("strategy_defaults", {})[k] = v
        
    with open("config_optimized.json", "w") as f:
        json.dump(best_cfg, f, indent=4)
    logger.info("Optimized config saved to config_optimized.json")
    print(f"\n[SUCCESS] Optimization done. Run backtest with: python main.py --symbol {args.symbol} --backtest")

if __name__ == "__main__":
    main()
