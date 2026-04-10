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
from strategies import create_strategy, STRATEGY_REGISTRY
from core.performance_tracker import PerformanceTracker
from core.indicator_engine import IndicatorEngine

# Disable excessive logging
logging.getLogger("trading_bot").setLevel(logging.WARNING)
optuna.logging.set_verbosity(optuna.logging.WARNING)

def optimize_strategy(strategy_name: str, symbol: str = "XAUUSDm", n_trials: int = 50):
    print(f"--- Starting Optimization for {strategy_name} on {symbol} ---")
    
    # 1. Setup Environment
    with open("config.json", "r") as f:
        master_config = json.load(f)
    
    # Force single strategy allocation for optimization
    master_config["portfolio_allocations"] = {strategy_name: 1.0}
    master_config["backtest"]["debug_signals"] = False
    master_config["backtest"]["adaptive_strategy"] = False
    
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
    
    # Pre-calculate indicators once (shared across trials)
    print("Pre-calculating indicators...")
    m5._indicators = IndicatorEngine.precalculate_all(symbol, "M5", m5)
    m15._indicators = IndicatorEngine.precalculate_all(symbol, "M15", m15)
    h1._indicators = IndicatorEngine.precalculate_all(symbol, "H1", h1)
    print(f"Data ready: M5={len(m5)}, M15={len(m15)}, H1={len(h1)}, M1={len(m1)}")

    def objective(trial: optuna.Trial):
        # 3. Suggest Parameters
        import copy
        config = copy.deepcopy(master_config)
        
        # Generic strategy params
        sl_atr = trial.suggest_float("sl_atr", 1.5, 5.0, step=0.1)
        tp_atr = trial.suggest_float("tp_atr", 2.0, 8.0, step=0.1)
        min_conf = trial.suggest_float("min_confidence", 0.5, 0.85, step=0.05)
        
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
            strat_cfg["rsi_oversold"] = trial.suggest_int("rsi_oversold", 20, 45)
            strat_cfg["rsi_overbought"] = trial.suggest_int("rsi_overbought", 55, 80)
            strat_cfg["adx_threshold"] = trial.suggest_int("adx_threshold", 20, 40)
            strat_cfg["max_vol_ratio"] = trial.suggest_float("max_vol_ratio", 1.5, 3.5, step=0.1)
            strat_cfg["max_adx_slope"] = trial.suggest_float("max_adx_slope", 3.0, 15.0, step=1.0)
        elif strategy_name == "LiquiditySweepBreakout":
            strat_cfg["body_thresh"] = trial.suggest_float("body_thresh", 0.5, 0.8, step=0.05)
            strat_cfg["lookback"] = trial.suggest_int("lookback", 10, 40)
        elif strategy_name == "TrendFollowing":
            strat_cfg["adx_threshold"] = trial.suggest_int("adx_threshold", 15, 30)
            strat_cfg["adx_strong"] = trial.suggest_int("adx_strong", 20, 40)
            strat_cfg["min_bars_between_signals"] = trial.suggest_int("min_bars_between_signals", 10, 50)
            strat_cfg["max_vol_ratio"] = trial.suggest_float("max_vol_ratio", 1.5, 3.5, step=0.1)
            strat_cfg["min_trend_maturity"] = trial.suggest_int("min_trend_maturity", 1, 5)
            
        config[strategy_name] = strat_cfg
        
        # 4. Run Backtest
        try:
            backtester = PortfolioBacktester(config)
            
            # Find the correct strategy type key
            norm_name = strategy_name.upper().replace("_", "")
            strategy_instance = STRATEGY_REGISTRY[norm_name](strategy_name, config=config)
            
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
            
            # 5. Calculate Metrics
            initial_bal = backtester.initial_partition_balance
            metrics = PerformanceTracker.calculate_metrics(history, initial_bal)
            sharpe = metrics.get("sharpe_ratio", 0)
            max_dd_str = str(metrics.get("max_drawdown", "100%"))
            max_dd = float(max_dd_str.replace("%", ""))
            trades = len(history)
            
            # Institutional Constraint: Hard Penalty for DD > 15%
            if max_dd > 15.0:
                return -100.0 - (max_dd - 15.0)
            
            # Institutional Objective: Balanced Sharpe & Frequency
            if trades < 20:
                return -50.0 + (trades * 2)
                
            return sharpe
            
        except Exception as e:
            import traceback
            print(f"  TRIAL ERROR: {e}")
            traceback.print_exc()
            return -200.0

    # 6. Optimize
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
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
