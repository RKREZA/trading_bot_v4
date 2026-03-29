import itertools
import copy
import logging
from typing import List, Dict, Tuple
from core.backtester import BacktestEngine
from core.strategy_engine import StrategyEngine
from datetime import datetime, timezone
import pandas as pd

logger = logging.getLogger("trading_bot.walk_forward")

class WalkForwardValidation:
    def __init__(self, config: dict, strategy: StrategyEngine):
        self.config = config
        self.strategy = strategy
        self.last_train_perf = {}

    def run_validation(self, symbol: str, h4: List[dict], h1: List[dict], m30: List[dict], m5: List[dict], d1: List[dict], 
                       train_days: int = 90, test_days: int = 30) -> List[Dict]:
        """
        Implementation of rolling window walk-forward validation (3m IS / 1m OOS).
        """
        results = []
        
        # Use M30 as the anchor for window slicing
        start_date = datetime.fromtimestamp(m30[0]['time'], tz=timezone.utc)
        end_date = datetime.fromtimestamp(m30[-1]['time'], tz=timezone.utc)
        
        current_train_start = start_date
        
        while True:
            current_train_end = current_train_start + pd.DateOffset(days=train_days)
            current_test_end = current_train_end + pd.DateOffset(days=test_days)
            
            if current_test_end > end_date:
                break
                
            logger.info(f"WFV Window: Train {current_train_start.date()} -> {current_train_end.date()} | Test {current_train_end.date()} -> {current_test_end.date()}")
            
            # Slice all timeframes
            train_data = {
                "h4": self._filter_by_time(h4, current_train_start, current_train_end),
                "h1": self._filter_by_time(h1, current_train_start, current_train_end),
                "m30": self._filter_by_time(m30, current_train_start, current_train_end),
                "m5": self._filter_by_time(m5, current_train_start, current_train_end),
                "d1": self._filter_by_time(d1, current_train_start, current_train_end)
            }
            
            test_data = {
                "h4": self._filter_by_time(h4, current_train_end, current_test_end),
                "h1": self._filter_by_time(h1, current_train_end, current_test_end),
                "m30": self._filter_by_time(m30, current_train_end, current_test_end),
                "m5": self._filter_by_time(m5, current_train_end, current_test_end),
                "d1": self._filter_by_time(d1, current_train_end, current_test_end)
            }

            if not train_data["m30"] or not test_data["m30"]:
                current_train_start += pd.DateOffset(days=test_days)
                continue

            # 1. OPTIMIZATION (IS) - Use Optuna or Grid
            # For simplicity in this first update, we use a small grid
            param_grid = {
                "min_confluence_score": [4, 5],
                "min_confidence": [65, 75],
                "sl_atr_buffer": [0.4, 0.6],
            }
            
            best_params = self._optimize(symbol, train_data, param_grid)
            
            # 2. VALIDATION (OOS)
            test_config = copy.deepcopy(self.config)
            test_config["strategy_defaults"].update(best_params)
            
            test_strategy = StrategyEngine(test_config)
            tester = BacktestEngine(test_config, test_strategy)
            
            test_perf = tester.run(symbol, 
                                   test_data["h4"], test_data["h1"], 
                                   test_data["m30"], test_data["m5"], 
                                   test_data["d1"], quiet=True)
            
            # Check for Overfitting (Decay Ratio)
            is_sharpe = self.last_train_perf.get("sharpe_ratio", 0)
            oos_sharpe = test_perf.get("sharpe_ratio", 0)
            decay_ratio = oos_sharpe / is_sharpe if is_sharpe > 0 else 0
            
            if decay_ratio < 0.5:
                logger.warning(f"HIGH OVERFIT RISK: Window {current_train_end.date()} OOS decay ratio: {decay_ratio:.2f}")

            # 3. RECORD RESULTS
            record = {
                "window": f"{current_train_end.date()} to {current_test_end.date()}",
                "best_params": best_params,
                "is_metrics": self.last_train_perf,
                "oos_metrics": test_perf,
                "decay_ratio": round(float(decay_ratio), 4)
            }
            results.append(record)
            
            current_train_start += pd.DateOffset(days=test_days) # Slide by test window
            
        # Save to file
        import json
        def datetime_handler(x):
            if isinstance(x, datetime):
                return x.isoformat()
            raise TypeError("Unknown type")

        with open("wf_robustness.json", "w") as f:
            json.dump(results, f, indent=4, default=datetime_handler)
        logger.info(f"Walk-forward results saved to wf_robustness.json")
            
        return results

    def _filter_by_time(self, candles: List[dict], start: datetime, end: datetime) -> List[dict]:
        return [c for c in candles if start <= datetime.fromtimestamp(c['time'], tz=timezone.utc) < end]

    def _optimize(self, symbol, data, grid) -> dict:
        best_metric = -999999
        best_params = {}
        self.last_train_perf = {}
        
        keys, values = zip(*grid.items())
        total_combinations = len(list(itertools.product(*values)))
        logger.info(f"Optimizing window with {total_combinations} combinations...")
        
        for v in itertools.product(*values):
            params = dict(zip(keys, v))
            tmp_config = copy.deepcopy(self.config)
            tmp_config["strategy_defaults"].update(params)
            
            strat = StrategyEngine(tmp_config)
            tester = BacktestEngine(tmp_config, strat)
            perf = tester.run(symbol, data["h4"], data["h1"], data["m30"], data["m5"], data["d1"], quiet=True)
            
            # Score: Sharpe * Profit Factor (basic but effective for IS)
            score = (perf.get("sharpe_ratio", 0) * perf.get("profit_factor", 0))
            if score > best_metric:
                best_metric = score
                best_params = params
                self.last_train_perf = perf
        
        return best_params
