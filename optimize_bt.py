import optuna
import json
import os
import logging
from datetime import datetime
from core.connection import MT5Connection
from core.data_fetcher import DataFetcher
from core.strategy_engine import StrategyEngine
from core.backtester import BacktestEngine

logger = logging.getLogger("trading_bot.optimizer")

def objective(trial):
    # Load base config
    with open("config.json", "r") as f:
        config = json.load(f)
    
    # Define search space
    config["strategy"]["min_confluence_score"] = trial.suggest_int("min_confluence", 2, 5)
    config["strategy"]["min_confidence"] = trial.suggest_int("min_confidence", 60, 80)
    config["strategy"]["sl_atr_buffer"] = trial.suggest_float("sl_buffer", 0.1, 1.0)
    config["strategy"]["pullback_distance_pct"] = trial.suggest_float("pullback", 0.2, 0.8)
    config["strategy"]["atr_period"] = trial.suggest_int("atr_period", 10, 24)
    
    # Initialize components
    conn = MT5Connection()
    if not conn.connect():
        return 0.0
    
    fetcher = DataFetcher(config)
    symbol = "XAUUSDm"
    
    # Fetch data (Sync for optimization)
    h4 = fetcher.fetch_candles(symbol, "H4", 500)
    h1 = fetcher.fetch_candles(symbol, "H1", 500)
    m30 = fetcher.fetch_candles(symbol, "M30", 500)
    m5 = fetcher.fetch_candles(symbol, "M5", 2000)
    d1 = fetcher.fetch_candles(symbol, "D1", 100)
    
    if not all([h4, h1, m30, m5, d1]):
        return 0.0
        
    strategy = StrategyEngine(config)
    tester = BacktestEngine(config, strategy)
    
    # Run backtest
    results = tester.run(symbol, h4, h1, m30, m5, d1, quiet=True)
    
    # Goal: Maximize Sharpe but penalize low trade count
    sharpe = results.get("sharpe_ratio", 0)
    pf = results.get("profit_factor", 0)
    trades = results.get("total_trades", 0)
    dd = results.get("max_drawdown", 100)
    
    if trades < 30: # Minimum sample size
        return 0.0
        
    if dd > 15: # Maximum allowed drawdown
        return 0.0
        
    # Multi-objective heuristic
    score = sharpe * (1.0 if pf > 1.5 else 0.5)
    return score

def run_optimization(n_trials=100):
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    
    print("\nOPTIMIZATION COMPLETE")
    print(f"Best Score: {study.best_value:.4f}")
    print("Best Params:")
    print(json.dumps(study.best_params, indent=2))
    
    # Save best params to a file
    with open("best_params.json", "w") as f:
        json.dump(study.best_params, f, indent=2)

if __name__ == "__main__":
    run_optimization()
