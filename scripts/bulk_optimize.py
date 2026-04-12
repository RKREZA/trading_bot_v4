import optuna
import json
import os
import sys
import logging
from datetime import datetime, timezone, timedelta
sys.path.append(os.getcwd())

from core.data.manager import DataManager
from core.connection import MT5Connection
from backtesting.backtester import PortfolioBacktester
from strategies import create_strategy, STRATEGY_REGISTRY
from core.performance_tracker import PerformanceTracker
from core.indicator_engine import IndicatorEngine

# Institutional Fast-Track Optimizer
class FastOptimizer:
    def __init__(self, trials=20, window_days=120):
        self.trials = trials
        self.window_days = window_days
        self.conn = MT5Connection()
        self.conn.connect()
        
    def optimize(self, symbol, strategy_name):
        print(f"\n>>> Optimizing {strategy_name} on {symbol} (Window: {self.window_days} days)")
        
        with open("config/config.json", "r") as f:
            base_config = json.load(f)
            
        data_manager = DataManager(base_config)
        end_dt = datetime.now(timezone.utc)
        start_dt = end_dt - timedelta(days=self.window_days)
        
        # Load data once
        m1 = data_manager.prepare_data(symbol, "M1", start_dt)
        m5 = data_manager.prepare_data(symbol, "M5", start_dt)
        m15 = data_manager.prepare_data(symbol, "M15", start_dt)
        h1 = data_manager.prepare_data(symbol, "H1", start_dt)
        
        # Pre-calculate indicators
        print("Pre-calculating indicators...")
        m5._indicators = IndicatorEngine.precalculate_all(symbol, "M5", m5)
        m15._indicators = IndicatorEngine.precalculate_all(symbol, "M15", m15)
        h1._indicators = IndicatorEngine.precalculate_all(symbol, "H1", h1)
        
        def objective(trial):
            config = json.loads(json.dumps(base_config)) # Deep copy
            config["portfolio_allocations"] = {strategy_name: 1.0}
            config["checkpointing"] = {"enabled": False} # Disable to avoid file locking issues
            
            # Param Grid
            strat_cfg = config.get(strategy_name, {})
            strat_cfg.update({
                "sl_atr": trial.suggest_float("sl_atr", 1.5, 4.0, step=0.1),
                "tp_atr": trial.suggest_float("tp_atr", 2.0, 7.0, step=0.1),
                "min_confidence": trial.suggest_float("min_confidence", 0.6, 0.85, step=0.05),
                "enabled": True
            })
            
            if strategy_name == "RangeBounce":
                strat_cfg["bb_std"] = trial.suggest_float("bb_std", 2.0, 3.0, step=0.1)
                strat_cfg["rsi_oversold"] = trial.suggest_int("rsi_oversold", 20, 35)
                strat_cfg["rsi_overbought"] = trial.suggest_int("rsi_overbought", 65, 80)
            elif strategy_name == "SmartMeanReversion":
                strat_cfg["bb_std"] = trial.suggest_float("bb_std", 2.0, 3.0, step=0.1)
                strat_cfg["rsi_oversold"] = trial.suggest_int("rsi_oversold", 15, 30)
                strat_cfg["rsi_overbought"] = trial.suggest_int("rsi_overbought", 70, 85)
            elif strategy_name == "TrendFollowing":
                strat_cfg["adx_threshold"] = trial.suggest_int("adx_threshold", 20, 35)
                strat_cfg["min_trend_maturity"] = trial.suggest_int("min_trend_maturity", 2, 6)
            elif strategy_name == "LiquiditySweepBreakout":
                strat_cfg["lookback"] = trial.suggest_int("lookback", 10, 30)
                strat_cfg["body_thresh"] = trial.suggest_float("body_thresh", 0.4, 0.7, step=0.05)
            elif strategy_name == "LiquiditySession":
                strat_cfg["range_maturity_limit"] = trial.suggest_float("range_maturity_limit", 1.5, 4.0, step=0.1)
                strat_cfg["vol_trigger_mult"] = trial.suggest_float("vol_trigger_mult", 0.2, 0.6, step=0.05)
                strat_cfg["min_range_bars"] = trial.suggest_int("min_range_bars", 10, 30)
            
            config[strategy_name] = strat_cfg
            
            try:
                backtester = PortfolioBacktester(config)
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
                
                if not backtester.history: return -100.0
                
                metrics = PerformanceTracker.calculate_metrics(backtester.history, 10000.0)
                sharpe = metrics.get("sharpe_ratio", 0)
                max_dd = float(str(metrics.get("max_drawdown", "100")).replace("%", ""))
                
                if max_dd > 10.0: return -100.0 - max_dd
                return sharpe
            except Exception as e:
                import traceback
                print(f"Exception in objective: {e}")
                traceback.print_exc()
                return -200.0

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=self.trials)
        
        print(f"Best Sharpe for {symbol}/{strategy_name}: {study.best_value:.2f}")
        return study.best_params

    def run_all(self, tasks):
        for symbol, strategies in tasks.items():
            for strategy in strategies:
                try:
                    best_params = self.optimize(symbol, strategy)
                    self.apply_params(symbol, strategy, best_params)
                except Exception as e:
                    print(f"Failed to optimize {strategy} on {symbol}: {e}")
                
    def apply_params(self, symbol, strategy, params):
        # Update symbol-specific config
        path = f"configs/symbols/{symbol}.json"
        if os.path.exists(path):
            with open(path, "r") as f:
                cfg = json.load(f)
            if "strategies" in cfg and strategy in cfg["strategies"]:
                cfg["strategies"][strategy].update(params)
                with open(path, "w") as f:
                    json.dump(cfg, f, indent=2)
                print(f"Updated {path} with best params.")

if __name__ == "__main__":
    optimizer = FastOptimizer(trials=30, window_days=180) # Increased trials and window
    tasks = {
        "GBPJPYm": ["TrendFollowing", "RangeBounce", "SmartMeanReversion", "LiquiditySession"],
        "EURUSDm": ["TrendFollowing", "RangeBounce", "SmartMeanReversion", "LiquiditySession"]
    }
    optimizer.run_all(tasks)
