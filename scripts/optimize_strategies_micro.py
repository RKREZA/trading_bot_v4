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

def objective(trial: optuna.Trial, strategy_type: str, master_config, m1, m5, m15, h1, symbol="XAUUSDm"):
    config = copy.deepcopy(master_config)
    
    # 1. Parameter suggestions based on strategy type
    params = {}
    if strategy_type == "TRENDFOLLOWING":
        params = {
            "adx_threshold": trial.suggest_int("adx_threshold", 15, 45),
            "tp_atr": trial.suggest_float("tp_atr", 3.0, 15.0, step=0.5),
            "sl_atr": trial.suggest_float("sl_atr", 1.5, 5.0, step=0.1),
            "min_trend_maturity": trial.suggest_int("min_trend_maturity", 1, 10),
            "min_confidence": trial.suggest_float("min_confidence", 0.5, 0.95, step=0.05)
        }
    elif strategy_type == "LIQUIDITYSWEEPBREAKOUT":
        params = {
            "lookback": trial.suggest_int("lookback", 10, 80),
            "body_thresh": trial.suggest_float("body_thresh", 0.3, 0.9, step=0.05),
            "h1_strength_thresh": trial.suggest_float("h1_strength_thresh", 0.3, 0.9, step=0.05),
            "sl_atr": trial.suggest_float("sl_atr", 1.0, 4.0, step=0.1),
            "tp_atr": trial.suggest_float("tp_atr", 3.0, 12.0, step=0.5),
            "min_confidence": trial.suggest_float("min_confidence", 0.2, 0.9, step=0.05),
            "min_bars_between_signals": trial.suggest_int("min_bars_between_signals", 1, 100)
        }
    elif strategy_type == "SMARTMEANREVERSION":
        params = {
            "bb_period": trial.suggest_int("bb_period", 10, 40),
            "bb_std": trial.suggest_float("bb_std", 1.5, 3.5, step=0.1),
            "rsi_period": 14,
            "rsi_overbought": trial.suggest_int("rsi_overbought", 65, 85),
            "rsi_oversold": trial.suggest_int("rsi_oversold", 15, 35),
            "sl_atr": trial.suggest_float("sl_atr", 1.0, 3.5, step=0.1),
            "tp_atr": trial.suggest_float("tp_atr", 2.0, 8.0, step=0.5),
            "min_confidence": trial.suggest_float("min_confidence", 0.2, 0.9, step=0.05)
        }
    elif strategy_type == "RANGEBOUNCE":
        params = {
            "bb_period": trial.suggest_int("bb_period", 10, 40),
            "bb_std": trial.suggest_float("bb_std", 1.5, 3.0, step=0.1),
            "rsi_period": 14,
            "rsi_oversold": trial.suggest_int("rsi_oversold", 20, 40),
            "rsi_overbought": trial.suggest_int("rsi_overbought", 60, 80),
            "sl_atr": trial.suggest_float("sl_atr", 1.0, 4.0, step=0.1),
            "tp_atr": trial.suggest_float("tp_atr", 2.0, 10.0, step=0.5),
            "min_confidence": trial.suggest_float("min_confidence", 0.2, 0.9, step=0.05)
        }
    
    # Update Config
    strategy_id = strategy_type.title().replace(" ", "")
    if "strategies" not in config: config["strategies"] = {}
    if strategy_id not in config["strategies"]: config["strategies"][strategy_id] = {}
    config["strategies"][strategy_id].update(params)
    config["strategies"][strategy_id]["enabled"] = True
    
    # Force isolate this strategy
    config["portfolio_allocations"] = {strategy_id: 1.0}
    config["backtest"]["initial_balance_per_strategy"] = 1000.0
    config["backtest"]["disable_checkpoint"] = True
    
    try:
        backtester = PortfolioBacktester(config)
        strategy_instance = STRATEGY_REGISTRY[strategy_type](strategy_id, config=config)
        
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
        if not history: return -1000.0
        
        metrics = PerformanceTracker.calculate_metrics(history, 1000.0)
        sharpe = metrics.get("sharpe_ratio", 0)
        max_dd = float(str(metrics.get("max_drawdown", "100")).replace("%", ""))
        trades = len(history)
        profit = metrics.get("net_profit", 0)
        
        # Micro-account safety hard-penalties
        if max_dd > 8.0: return -500.0 - max_dd # Strict DD bound for $1000
        if trades < 15: return -100.0 + (trades * 2.0)
        
        # Objective: Balance of Sharpe + Profitability Factor
        score = (sharpe * 10.0) + (profit / 100.0)
        return score
        
    except Exception: return -2000.0

def run_optimization(strategy_type: str, n_trials=50, symbol="XAUUSDm"):
    strategy_type = strategy_type.upper()
    print(f"\n>>> OPTIMIZING: {strategy_type} [Trials: {n_trials}] [Balance: $1000] <<<")
    
    suite = ComprehensiveBacktestSuite()
    master_config = suite.config_loader.get_symbol_config(symbol)
    
    m5 = suite.load_real_data(symbol=symbol, timeframe="M5", n_bars=35000)
    m15 = suite.load_real_data(symbol=symbol, timeframe="M15", n_bars=12000)
    h1 = suite.load_real_data(symbol=symbol, timeframe="H1", n_bars=3000)
    m1 = suite.load_real_data(symbol=symbol, timeframe="M1", n_bars=200000)
    
    m5._indicators = IndicatorEngine.precalculate_all(symbol, "M5", m5)
    m15._indicators = IndicatorEngine.precalculate_all(symbol, "M15", m15)
    h1._indicators = IndicatorEngine.precalculate_all(symbol, "H1", h1)
    
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, strategy_type, master_config, m1, m5, m15, h1, symbol), n_trials=n_trials)
    
    print("\n" + "="*50)
    print(f"RESULTS FOR {strategy_type}")
    print(f"Best Score: {study.best_value:.4f}")
    print(f"Best Params: {json.dumps(study.best_params, indent=2)}")
    print("="*50)
    
    os.makedirs("config/micro_alpha", exist_ok=True)
    out_file = f"config/micro_alpha/{strategy_type.lower()}.json"
    with open(out_file, "w") as f:
        json.dump(study.best_params, f, indent=2)
    print(f"Alpha saved to: {out_file}")
    return study.best_params

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", type=str, default="all", help="Strategy type or 'all'")
    parser.add_argument("--trials", type=int, default=50)
    args = parser.parse_args()
    
    enabled_strats = ["TRENDFOLLOWING", "LIQUIDITYSWEEPBREAKOUT", "SMARTMEANREVERSION", "RANGEBOUNCE"]
    
    if args.strategy.upper() == "ALL":
        for s in enabled_strats:
            run_optimization(s, n_trials=args.trials)
    else:
        run_optimization(args.strategy, n_trials=args.trials)
