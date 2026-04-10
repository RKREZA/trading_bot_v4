"""
V4-ULTRA Institutional Optimization Suite
Uses Optuna to find global optima for strategy parameters.
Focus: XAUUSDm | Constraint: Max Drawdown < 15% | Metric: Sharpe Ratio
"""
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
from strategies import create_strategy
from core.performance_tracker import PerformanceTracker

# Disable excessive logging
logging.getLogger("trading_bot").setLevel(logging.WARNING)

def optimize_strategy(strategy_name: str, symbol: str = "XAUUSDm", n_trials: int = 50):
    print(f"--- Starting Optimization for {strategy_name} on {symbol} ---")
    
    # 1. Setup Environment
    with open("config.json", "r") as f:
        master_config = json.load(f)
    
    # Force single strategy allocation for optimization
    master_config["portfolio_allocations"] = {strategy_name: 1.0}
    master_config["backtest"]["debug_signals"] = False
    
    connection = MT5Connection()
    if not connection.connect():
        print("Error: Could not connect to MT5.")
        return
    
    data_manager = DataManager(master_config)
    
    # 2. Prepare Data (1 year window)
    end_dt = datetime.now(timezone.utc)
    start_dt = end_dt - timedelta(days=365)
    
    print(f"Preparing data from {start_dt.date()} to {end_dt.date()}...")
    m1 = data_manager.prepare_data(symbol, "M1", start_dt)
    m5 = data_manager.prepare_data(symbol, "M5", start_dt)
    m15 = data_manager.prepare_data(symbol, "M15", start_dt)
    h1 = data_manager.prepare_data(symbol, "H1", start_dt)
    
    # Alignment
    if len(m1) > 0:
        max_t = m1.time[-1]
        m5 = m5[m5.time <= max_t]
        m15 = m15[m15.time <= max_t]
        h1 = h1[h1.time <= max_t]
    
    def objective(trial: optuna.Trial):
        # 3. Suggest Parameters
        config = master_config.copy()
        
        # Generic strategy params
        sl_atr = trial.suggest_float("sl_atr", 1.5, 5.0, step=0.1)
        tp_atr = trial.suggest_float("tp_atr", 2.0, 8.0, step=0.1)
        min_conf = trial.suggest_float("min_confidence", 0.5, 0.9, step=0.05)
        
        # Strategy-specific params
        strat_cfg = config.get(strategy_name, {})
        strat_cfg.update({
            "sl_atr": sl_atr,
            "tp_atr": tp_atr,
            "min_confidence": min_conf,
            "enabled": True
        })
        
        if strategy_name == "RangeBounce":
            strat_cfg["bb_std"] = trial.suggest_float("bb_std", 1.5, 3.0, step=0.1)
            strat_cfg["rsi_oversold"] = trial.suggest_int("rsi_oversold", 20, 45) # Widened
            strat_cfg["rsi_overbought"] = trial.suggest_int("rsi_overbought", 55, 80) # Widened
            strat_cfg["adx_threshold"] = trial.suggest_int("adx_threshold", 20, 40) # Widened
            strat_cfg["max_vol_ratio"] = trial.suggest_float("max_vol_ratio", 1.5, 3.5, step=0.1)
            strat_cfg["max_adx_slope"] = trial.suggest_float("max_adx_slope", 3.0, 15.0, step=1.0)
        elif strategy_name == "LiquiditySweepBreakout":
            strat_cfg["body_thresh"] = trial.suggest_float("body_thresh", 0.5, 0.8, step=0.05)
            strat_cfg["lookback"] = trial.suggest_int("lookback", 10, 40)
            
        config[strategy_name] = strat_cfg
        
        # 4. Run Backtest
        try:
            backtester = PortfolioBacktester(config)
            strategy_instance = create_strategy(strategy_name, config=config)
            
            history, _ = backtester.run(
                symbol=symbol,
                strategies=[strategy_instance],
                target_tf_data=m5,
                h1_data=h1,
                m15_data=m15,
                m5_data=m5,
                m1_data=m1
            )
            
            if not history:
                return -100.0
            
            # 5. Calculate Metrics
            metrics = PerformanceTracker.calculate_metrics(history, config.get("initial_balance", 1000.0))
            sharpe = metrics.get("sharpe_ratio", 0)
            max_dd = metrics.get("max_drawdown", 100.0)
            trades = len(history)
            
            # Institutional Constraint: Hard Penalty for DD > 15%
            if max_dd > 15.0:
                return -100.0 - (max_dd - 15.0)
            
            # Institutional Objective: Balanced Sharpe & Frequency
            # Target > 24 trades/year (2 per month)
            if trades < 24:
                # Penalize severely if frequency is too low for a range strategy
                # return sharpe * (trades / 24.0) 
                return -50.0 + (trades * 2) # Force exploration of higher frequency
                
            return sharpe
            
        except Exception as e:
            return -200.0

    # 6. Optimize
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    
    print("\n" + "=" * 40)
    print(f"OPTIMIZATION COMPLETE FOR {strategy_name}")
    print(f"Best Sharpe: {study.best_value:.2f}")
    print("Best Params:", json.dumps(study.best_params, indent=2))
    print("=" * 40)
    
    # Save best params
    output_file = f"config/opt_{strategy_name.lower()}.json"
    os.makedirs("config", exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(study.best_params, f, indent=2)
    
    return study.best_params

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", type=str, required=True, help="Strategy name to optimize")
    parser.add_argument("--trials", type=int, default=30, help="Number of trials")
    args = parser.parse_args()
    
    optimize_strategy(args.strategy, n_trials=args.trials)
