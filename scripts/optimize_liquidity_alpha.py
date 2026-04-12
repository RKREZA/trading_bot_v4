import optuna
import json
import os
import sys
import logging
import pandas as pd
import numpy as np
import copy
from datetime import datetime, timezone, timedelta
from typing import Dict, Any

# Add project root to path
sys.path.append(os.getcwd())

from core.config.loader import ConfigLoader
from backtesting.backtester import PortfolioBacktester
from comprehensive_backtest import ComprehensiveBacktestSuite
from strategies import create_strategy, STRATEGY_REGISTRY
from core.performance_tracker import PerformanceTracker
from core.indicator_engine import IndicatorEngine

# Disable excessive logging
logging.getLogger("trading_bot").setLevel(logging.WARNING)
optuna.logging.set_verbosity(optuna.logging.WARNING)

def objective(trial: optuna.Trial, master_config, m1, m5, m15, h1, symbol="XAUUSDm"):
    config = copy.deepcopy(master_config)
    
    # Suggest LiquiditySweepBreakout Parameters
    lookback = trial.suggest_int("lookback", 10, 60)
    body_thresh = trial.suggest_float("body_thresh", 0.4, 0.9, step=0.05)
    h1_strength_thresh = trial.suggest_float("h1_strength_thresh", 0.4, 0.9, step=0.05)
    sl_atr = trial.suggest_float("sl_atr", 1.5, 4.0, step=0.1)
    tp_atr = trial.suggest_float("tp_atr", 3.0, 12.0, step=0.5)
    min_conf = trial.suggest_float("min_confidence", 0.2, 0.9, step=0.05)
    min_bars_between = trial.suggest_int("min_bars_between_signals", 1, 100)
    
    # Update Config for specific strategy
    if "strategies" not in config:
        config["strategies"] = {}
        
    if "LiquiditySweepBreakout" not in config["strategies"]:
        config["strategies"]["LiquiditySweepBreakout"] = {}
        
    config["strategies"]["LiquiditySweepBreakout"].update({
        "lookback": lookback,
        "body_thresh": body_thresh,
        "h1_strength_thresh": h1_strength_thresh,
        "sl_atr": sl_atr,
        "tp_atr": tp_atr,
        "min_confidence": min_conf,
        "min_bars_between_signals": min_bars_between,
        "enabled": True
    })
    
    # Isolate LiquiditySweepBreakout for Alpha finding
    config["portfolio_allocations"] = {"LiquiditySweepBreakout": 1.0}
    
    # Ensure balance is set to $1000 for the simulation
    config["backtest"]["initial_balance_per_strategy"] = 1000.0
    
    try:
        backtester = PortfolioBacktester(config)
        
        # Instantiate strategy correctly
        norm_name = "LIQUIDITYSWEEPBREAKOUT"
        strategy_instance = STRATEGY_REGISTRY[norm_name]("LiquiditySweepBreakout", config=config)
        
        backtester.run(
            symbol=symbol,
            strategies=[strategy_instance],
            target_tf_data=m5,
            h1_data=h1,
            m15_data=m15,
            m5_data=m5,
            m1_data=m1
        )
        
        history = backtester.history
        if not history:
            return -100.0
        
        initial_bal = 1000.0
        metrics = PerformanceTracker.calculate_metrics(history, initial_bal)
        sharpe = metrics.get("sharpe_ratio", 0)
        max_dd_str = str(metrics.get("max_drawdown", "100%"))
        max_dd = float(max_dd_str.replace("%", ""))
        trades = len(history)
        
        # Penalty for low frequency
        if trades < 10:
            return -50.0 + (trades * 2.0)
            
        # Hard Penalty for Drawdown (Aggressive for $1000 balance)
        if max_dd > 10.0:
            return -200.0 - (max_dd - 10.0)
            
        # Reward profitability + Sharpe
        profit = metrics.get("net_profit", 0)
        if profit <= 0:
            return -10.0 + sharpe # Small reward for sharpe if just below zero
            
        return sharpe + (profit / 100.0)
        
    except Exception as e:
        # print(f"Trial Error: {e}")
        return -500.0

def run_alpha_optimization(n_trials=100, symbol="XAUUSDm"):
    print(f"--- Starting LiquiditySweep Alpha Optimization for Micro ($1000) [Trials: {n_trials}] ---")
    
    suite = ComprehensiveBacktestSuite()
    # Use symbol config but we'll override balance
    master_config = suite.config_loader.get_symbol_config(symbol)
    
    print(f"Preparing data via local parquet cache...")
    # Use a longer window for better robustness
    m5 = suite.load_real_data(symbol=symbol, timeframe="M5", n_bars=35000)
    m15 = suite.load_real_data(symbol=symbol, timeframe="M15", n_bars=12000)
    h1 = suite.load_real_data(symbol=symbol, timeframe="H1", n_bars=3000)
    m1 = suite.load_real_data(symbol=symbol, timeframe="M1", n_bars=200000)
    
    # Pre-calculate indicators
    m5._indicators = IndicatorEngine.precalculate_all(symbol, "M5", m5)
    m15._indicators = IndicatorEngine.precalculate_all(symbol, "M15", m15)
    h1._indicators = IndicatorEngine.precalculate_all(symbol, "H1", h1)
    
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, master_config, m1, m5, m15, h1, symbol), n_trials=n_trials)
    
    print("\n" + "=" * 40)
    print(f"BEST LIQUIDITY ALPHA FOUND")
    print(f"Objective Value: {study.best_value:.2f}")
    print("Best Params:", json.dumps(study.best_params, indent=2))
    print("=" * 40)
    
    # Save best alpha
    os.makedirs("config", exist_ok=True)
    out_file = "config/alpha_liquidity_micro.json"
    with open(out_file, "w") as f:
        json.dump(study.best_params, f, indent=2)
    print(f"Results saved to {out_file}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=100)
    args = parser.parse_args()
    run_alpha_optimization(n_trials=args.trials)
