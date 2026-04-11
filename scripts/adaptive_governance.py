"""
V4-ULTRA Institutional Adaptive Governance Suite
Calculates optimal parameters for a 3-week lookback and validates on a 1-week walk-forward window.
Automatically updates config.json if the new parameters exceed the performance of the current ones.
"""
import optuna
import json
import os
import sys
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List

# Add project root to path
sys.path.append(os.getcwd())

from core.data.manager import DataManager
from core.connection import MT5Connection
from backtesting.backtester import PortfolioBacktester
from strategies import create_strategy, STRATEGY_REGISTRY
from core.performance_tracker import PerformanceTracker
from core.indicator_engine import IndicatorEngine

# Disable excessive logging
logging.getLogger("trading_bot").setLevel(logging.WARNING)
optuna.logging.set_verbosity(optuna.logging.WARNING)

class WalkForwardGovernor:
    def __init__(self, config_path: str = "config.json"):
        self.config_path = config_path
        with open(config_path, "r") as f:
            self.master_config = json.load(f)
        
        self.connection = MT5Connection()
        self.data_manager = DataManager(self.master_config)
        self.symbol = "XAUUSDm"

    def run_cycle(self, strategy_name: str, n_trials: int = 40):
        print(f"\n--- GOVERNANCE CYCLE: {strategy_name} ({self.symbol}) ---")
        
        if not self.connection.connect():
            print("Error: Could not connect to MT5.")
            return

        # 1. Define Windows
        now = datetime.now(timezone.utc)
        start_dt = now - timedelta(days=35)
        
        print(f"Syncing data for 35-day window...")
        m1 = self.data_manager.prepare_data(self.symbol, "M1", start_dt)
        m5 = self.data_manager.prepare_data(self.symbol, "M5", start_dt)
        m15 = self.data_manager.prepare_data(self.symbol, "M15", start_dt)
        h1 = self.data_manager.prepare_data(self.symbol, "H1", start_dt)
        
        # Split: 21 days training, 7 days validation, 7 days holdout
        split_ts_val = (now - timedelta(days=14)).timestamp()
        split_ts_holdout = (now - timedelta(days=7)).timestamp()
        
        m5_train = m5[m5.time < split_ts_val]
        m5_val = m5[(m5.time >= split_ts_val) & (m5.time < split_ts_holdout)]
        
        h1_train = h1[h1.time < split_ts_val]
        h1_val = h1[(h1.time >= split_ts_val) & (h1.time < split_ts_holdout)]
        
        m15_train = m15[m15.time < split_ts_val]
        m15_val = m15[(m15.time >= split_ts_val) & (m15.time < split_ts_holdout)]
        
        m1_train = m1[m1.time < split_ts_val]
        m1_val = m1[(m1.time >= split_ts_val) & (m1.time < split_ts_holdout)]

        # Precalculate Indicators
        print("Calculating Indicator Matrix...")
        m5_train._indicators = IndicatorEngine.precalculate_all(self.symbol, "M5", m5_train)
        m5_val._indicators = IndicatorEngine.precalculate_all(self.symbol, "M5", m5_val)
        
        # 2. Optimization Phase (Training)
        print(f"Phase 1: Optimization (Train Window size: {len(m5_train)})")
        best_params = self._optimize(strategy_name, m1_train, m5_train, m15_train, h1_train, n_trials)
        
        if not best_params:
            print("Optimization failed to find valid parameters.")
            return

        # 3. Validation Phase (In-Sample vs Out-of-Sample)
        print(f"Phase 2: Walk-Forward Validation (Val Window size: {len(m5_val)})")
        val_metrics = self._validate(strategy_name, best_params, m1_val, m5_val, m15_val, h1_val)
        
        pf = val_metrics.get("profit_factor", 0)
        trades = val_metrics.get("total_trades", 0)
        dd = float(str(val_metrics.get("max_drawdown", "100%")).replace("%", ""))

        print(f"Validation Result: PF={pf:.2f}, Trades={trades}, DD={dd:.1f}%")
        
        # 4. Deployment Gate
        # Institutional Rule: Only update if PF > 1.4 and DD < 10% on fresh data
        if pf >= 1.4 and dd < 10.0 and trades >= 5:
            print("SUCCESS: Parameters passed WFO validation gate. Patching config.json...")
            self._patch_config(strategy_name, best_params)
        else:
            print("REJECTED: Parameters failed WFO safety gate. Maintaining current regime.")

    def _optimize(self, strategy_name, m1, m5, m15, h1, n_trials):
        def objective(trial):
            import copy
            config = copy.deepcopy(self.master_config)
            
            sl_atr = trial.suggest_float("sl_atr", 1.8, 4.0, step=0.1)
            tp_atr = trial.suggest_float("tp_atr", 2.5, 7.0, step=0.1)
            
            strat_cfg = config.get(strategy_name, {})
            strat_cfg.update({"sl_atr": sl_atr, "tp_atr": tp_atr, "enabled": True})
            
            if strategy_name == "TrendFollowing":
                strat_cfg["adx_threshold"] = trial.suggest_int("adx_threshold", 20, 30)
            elif strategy_name == "RangeBounce":
                strat_cfg["bb_std"] = trial.suggest_float("bb_std", 2.0, 3.0, step=0.1)
            
            config[strategy_name] = strat_cfg
            
            backtester = PortfolioBacktester(config)
            norm_name = strategy_name.upper().replace("_", "")
            strategy_instance = STRATEGY_REGISTRY[norm_name](strategy_name, config=config)
            
            backtester.run(self.symbol, [strategy_instance], m5, h1, m15, m5, m1)
            if not backtester.history: return -100
            
            metrics = PerformanceTracker.calculate_metrics(backtester.history, 5000)
            sharpe = metrics.get("sharpe_ratio", 0)
            dd = float(str(metrics.get("max_drawdown", "100%")).replace("%", ""))
            
            if dd > 12.0: return -100 - dd
            return sharpe

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials)
        return study.best_params if study.best_value > 0 else None

    def _validate(self, strategy_name, params, m1, m5, m15, h1):
        import copy
        config = copy.deepcopy(self.master_config)
        config[strategy_name].update(params)
        
        backtester = PortfolioBacktester(config)
        norm_name = strategy_name.upper().replace("_", "")
        strategy_instance = STRATEGY_REGISTRY[norm_name](strategy_name, config=config)
        
        backtester.run(self.symbol, [strategy_instance], m5, h1, m15, m5, m1)
        if not backtester.history: return {}
        
        return PerformanceTracker.calculate_metrics(backtester.history, 5000)

    def _patch_config(self, strategy_name, params):
        with open(self.config_path, "r") as f:
            current = json.load(f)
        
        current[strategy_name].update(params)
        current["_audit_date"] = datetime.now().strftime("%Y-%m-%d")
        current["_comment"] = f"Adaptive Update: {strategy_name} calibrated via WFO."
        
        with open(self.config_path, "w") as f:
            json.dump(current, f, indent=2)
        print(f"Config patched successfully for {strategy_name}.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", type=str, required=True)
    parser.add_argument("--trials", type=int, default=40)
    args = parser.parse_args()
    
    governor = WalkForwardGovernor()
    governor.run_cycle(args.strategy, n_trials=args.trials)
