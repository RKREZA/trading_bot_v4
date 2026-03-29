import optuna
import json
import logging
import copy
from datetime import datetime, timezone
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
    
    # Strategy Parameters
    local_cfg["strategy"]["min_confluence_score"] = trial.suggest_int("min_confluence_score", 3, 6)
    local_cfg["strategy"]["min_confidence"] = trial.suggest_int("min_confidence", 40, 75)
    local_cfg["strategy"]["sl_atr_buffer"] = trial.suggest_float("sl_atr_buffer", 0.2, 1.0, step=0.1)
    local_cfg["strategy"]["tp_multiplier"] = trial.suggest_float("tp_multiplier", 1.5, 4.0, step=0.1)
    local_cfg["strategy"]["pullback_distance_pct"] = trial.suggest_float("pullback_distance_pct", 0.3, 0.8, step=0.05)
    
    # MFE Trailing Base
    local_cfg["strategy"]["mfe_trail_base"] = trial.suggest_float("mfe_trail_base", 0.3, 0.7, step=0.05)

    # Initialize engines
    strategy = StrategyEngine(local_cfg)
    backtester = BacktestEngine(local_cfg, strategy)
    
    # Run backtest
    results = backtester.run("XAUUSDm", h4, h1, m30, m5, d1, quiet=True)
    
    # Objective: (Net Profit * Sharpe * PF) / MaxDrawdown
    profit = results.get("net_profit", 0)
    sharpe = results.get("sharpe_ratio", 0)
    pf = results.get("profit_factor", 0)
    mdd = results.get("max_drawdown_pct", 100)
    
    if mdd == 0: mdd = 0.01 # Avoid div by zero
    if profit <= 0: return -abs(profit) # Penalize losses
    
    score = (profit * sharpe * pf) / mdd
    return score

def main():
    # 1. Load Config
    with open("config.json", "r") as f:
        config = json.load(f)
    
    symbol = "XAUUSDm"
    
    # 2. Fetch Data (Once)
    conn = MT5Connection()
    conn.config = config
    if not conn.connect():
        logger.error("Could not connect to MT5 for data fetching.")
        return
    
    fetcher = DataFetcher()
    logger.info(f"Fetching historical data for {symbol}...")
    
    # Fetch ample data for a meaningful optimization
    h4 = fetcher.fetch_candles(symbol, "H4", 1000)
    h1 = fetcher.fetch_candles(symbol, "H1", 2000)
    m30 = fetcher.fetch_candles(symbol, "M30", 5000)
    m5 = fetcher.fetch_candles(symbol, "M5", 15000)
    d1 = fetcher.fetch_candles(symbol, "D1", 500)
    
    conn.disconnect()
    
    if not all([h4, h1, m30, m5, d1]):
        logger.error("Failed to fetch all required timeframes.")
        return

    logger.info("Starting Optuna Study...")
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, config, h4, h1, m30, m5, d1), n_trials=100)

    logger.info("Optimization Complete!")
    logger.info(f"Best Score: {study.best_value}")
    logger.info(f"Best Params: {study.best_params}")
    
    # 3. Save optimized config
    best_cfg = copy.deepcopy(config)
    for k, v in study.best_params.items():
        best_cfg["strategy"][k] = v
        
    with open("config_optimized.json", "w") as f:
        json.dump(best_cfg, f, indent=4)
    logger.info("Optimized config saved to config_optimized.json")

if __name__ == "__main__":
    main()
