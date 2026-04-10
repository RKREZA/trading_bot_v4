import optuna
import json
import os
import sys
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
from typing import Dict, Any
from dotenv import load_dotenv

# Add project root to path
sys.path.append(os.getcwd())
load_dotenv()

from core.data.manager import DataManager
from core.connection import MT5Connection
from backtesting.backtester import PortfolioBacktester
from strategies import create_strategy, STRATEGY_REGISTRY
from core.performance_tracker import PerformanceTracker
from core.indicator_engine import IndicatorEngine

# Disable excessive logging
logging.getLogger("trading_bot").setLevel(logging.WARNING)
optuna.logging.set_verbosity(optuna.logging.WARNING)

def objective(trial: optuna.Trial, master_config, m1, m5, m15, h1, symbol="XAUUSDm"):
    import copy
    config = copy.deepcopy(master_config)
    
    # Suggest TrendFollowing Parameters
    adx_threshold = trial.suggest_int("adx_threshold", 20, 35)
    tp_atr = trial.suggest_float("tp_atr", 3.0, 8.0, step=0.5)
    sl_atr = trial.suggest_float("sl_atr", 1.5, 3.5, step=0.2)
    min_trend_maturity = trial.suggest_int("min_trend_maturity", 2, 6)
    min_conf = trial.suggest_float("min_confidence", 0.65, 0.85, step=0.05)
    
    # Trailing Stop Parameters
    be_rr = trial.suggest_float("phase1_rr_threshold", 1.0, 2.0, step=0.1)
    trail_mult = trial.suggest_float("phase3_trail_mult", 1.5, 3.5, step=0.1)
    
    # Update Config for specific strategy
    config["TrendFollowing"].update({
        "adx_threshold": adx_threshold,
        "tp_atr": tp_atr,
        "sl_atr": sl_atr,
        "min_trend_maturity": min_trend_maturity,
        "min_confidence": min_conf,
        "enabled": True
    })
    
    config["trailing_stop"].update({
        "phase1_rr_threshold": be_rr,
        "phase3_trail_mult": trail_mult,
        "enabled": True
    })
    
    # Isolate TrendFollowing for Alpha finding
    config["portfolio_allocations"] = {"TrendFollowing": 1.0}
    
    try:
        backtester = PortfolioBacktester(config)
        
        # Instantiate strategy correctly
        norm_name = "TRENDFOLLOWING"
        strategy_instance = STRATEGY_REGISTRY[norm_name]("TrendFollowing", config=config)
        
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
        
        initial_bal = backtester.initial_partition_balance
        metrics = PerformanceTracker.calculate_metrics(history, initial_bal)
        sharpe = metrics.get("sharpe_ratio", 0)
        max_dd_str = str(metrics.get("max_drawdown", "100%"))
        max_dd = float(max_dd_str.replace("%", ""))
        trades = len(history)
        
        # Penalty for low frequency (Alpha must be tradeable)
        if trades < 25:
            return -50.0 + (trades * 1.5)
            
        # Hard Penalty for Drawdown
        if max_dd > 15.0:
            return -150.0 - (max_dd - 15.0)
            
        return sharpe
        
    except Exception as e:
        return -200.0

def run_alpha_optimization(n_trials=50, symbol="XAUUSDm"):
    print(f"--- Starting TrendFollowing Alpha Optimization [Trials: {n_trials}] ---")
    
    with open("config.json", "r") as f:
        master_config = json.load(f)
    
    connection = MT5Connection()
    if not connection.connect():
        print("Error: Could not connect to MT5.")
        return
    
    data_manager = DataManager(master_config)
    end_dt = datetime.now(timezone.utc)
    start_dt = end_dt - timedelta(days=365)
    
    print(f"Preparing data...")
    m1 = data_manager.prepare_data(symbol, "M1", start_dt)
    m5 = data_manager.prepare_data(symbol, "M5", start_dt)
    m15 = data_manager.prepare_data(symbol, "M15", start_dt)
    h1 = data_manager.prepare_data(symbol, "H1", start_dt)
    
    # Pre-calculate indicators
    m5._indicators = IndicatorEngine.precalculate_all(symbol, "M5", m5)
    m15._indicators = IndicatorEngine.precalculate_all(symbol, "M15", m15)
    h1._indicators = IndicatorEngine.precalculate_all(symbol, "H1", h1)
    
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, master_config, m1, m5, m15, h1, symbol), n_trials=n_trials)
    
    print("\n" + "=" * 40)
    print(f"BEST ALPHA FOUND FOR TRENDFOLLOWING")
    print(f"Objective Value (Sharpe): {study.best_value:.2f}")
    print("Best Params:", json.dumps(study.best_params, indent=2))
    print("=" * 40)
    
    # Save best alpha
    os.makedirs("config", exist_ok=True)
    with open("config/alpha_trend.json", "w") as f:
        json.dump(study.best_params, f, indent=2)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=50)
    args = parser.parse_args()
    run_alpha_optimization(n_trials=args.trials)
