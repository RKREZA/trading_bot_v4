import itertools
import copy
import logging
from typing import List, Dict, Tuple
from core.backtester import BacktestEngine
from core.strategy_engine import StrategyEngine
from core.types import CandleArray
from datetime import datetime, timezone
import pandas as pd

logger = logging.getLogger("trading_bot.walk_forward")

class WalkForwardValidation:
    def __init__(self, config: dict, strategy: StrategyEngine):
        self.config = config
        self.strategy = strategy
        self.last_train_perf = {}

    def run_validation(self, symbol: str, h4: List[dict], h1: List[dict], m30: List[dict], m5: List[dict], d1: List[dict], 
                       train_days: int = 90, test_days: int = 30, mode: str = "anchored") -> List[Dict]:
        """
        Institutional-grade Walk-Forward Optimization (WFO).
        - Anchored: Training start stays at the beginning.
        - Rolling: Training window slides with the test window.
        """
        results = []
        start_date = datetime.fromtimestamp(m30[0]['time'], tz=timezone.utc)
        end_date = datetime.fromtimestamp(m30[-1]['time'], tz=timezone.utc)
        
        current_train_start = start_date
        param_popularity = {} # Track which params win most windows
        
        while True:
            current_train_end = current_train_start + pd.DateOffset(days=train_days)
            current_test_end = current_train_end + pd.DateOffset(days=test_days)
            
            if current_test_end > end_date:
                break
                
            logger.info(f"--- WFO WINDOW: Train {current_train_start.date()} -> {current_train_end.date()} | Test {current_train_end.date()} -> {current_test_end.date()} ---")
            
            train_data = {
                "h4": CandleArray.from_dicts(self._filter_by_time(h4, current_train_start, current_train_end)),
                "h1": CandleArray.from_dicts(self._filter_by_time(h1, current_train_start, current_train_end)),
                "m30": CandleArray.from_dicts(self._filter_by_time(m30, current_train_start, current_train_end)),
                "m5": CandleArray.from_dicts(self._filter_by_time(m5, current_train_start, current_train_end)),
                "d1": CandleArray.from_dicts(self._filter_by_time(d1, current_train_start, current_train_end))
            }
            
            test_data = {
                "h4": CandleArray.from_dicts(self._filter_by_time(h4, current_train_end, current_test_end)),
                "h1": CandleArray.from_dicts(self._filter_by_time(h1, current_train_end, current_test_end)),
                "m30": CandleArray.from_dicts(self._filter_by_time(m30, current_train_end, current_test_end)),
                "m5": CandleArray.from_dicts(self._filter_by_time(m5, current_train_end, current_test_end)),
                "d1": CandleArray.from_dicts(self._filter_by_time(d1, current_train_end, current_test_end))
            }

            if not train_data["m5"] or not test_data["m5"]:
                if mode == "rolling": current_train_start += pd.DateOffset(days=test_days)
                else: current_train_start = start_date # Should not change in anchored
                continue

            # 1. OPTIMIZATION (IS) - Grid Search with Stability Emphasis
            param_grid = {
                "min_confluence_score": [3, 4, 5],
                "min_confidence": [60, 70, 80],
                "sl_atr_buffer": [0.6, 0.8, 1.0],
                "swing_lookback": [3, 5, 8]
            }
            
            best_params = self._optimize(symbol, train_data, param_grid)
            
            # 2. VALIDATION (OOS)
            test_config = copy.deepcopy(self.config)
            test_config["strategy_defaults"].update(best_params)
            
            test_strategy = StrategyEngine(test_config)
            tester = BacktestEngine(test_config, test_strategy)
            test_perf = tester.run(symbol, test_data["h4"], test_data["h1"], 
                                   test_data["m30"], test_data["m5"], test_data["d1"], quiet=True)
            
            # 3. SCORE & RECORD
            is_sharpe = self.last_train_perf.get("sharpe_ratio", 0)
            oos_sharpe = test_perf.get("sharpe_ratio", 0)
            consistency = max(0.0, min(1.0, oos_sharpe / is_sharpe)) if is_sharpe > 0 else 0
            
            # Param Popularity (Winner selection)
            p_key = str(best_params)
            param_popularity[p_key] = param_popularity.get(p_key, 0) + (1 * consistency)

            results.append({
                "window": f"{current_train_end.date()} to {current_test_end.date()}",
                "best_params": best_params,
                "is_metrics": self.last_train_perf,
                "oos_metrics": test_perf,
                "consistency": round(float(consistency), 4)
            })
            
            if mode == "rolling":
                current_train_start += pd.DateOffset(days=test_days)
            else:
                # Anchored: Train end moves forward, Train start stays fixed
                # In our loop, we just need to increment the days for the next iteration's end calculation
                train_days += test_days 

        # 4. EXPORT OPTIMIZED CONFIG
        if param_popularity:
            best_stable_params_str = max(param_popularity, key=param_popularity.get)
            import ast
            best_stable_params = ast.literal_eval(best_stable_params_str)
            
            final_config = copy.deepcopy(self.config)
            final_config["strategy_defaults"].update(best_stable_params)
            
            with open("config_optimized.json", "w") as f:
                import json
                json.dump(final_config, f, indent=4)
            logger.info("PHASE 11: Optimized parameters saved to config_optimized.json")

        # Save WF Robustness Log
        with open("wf_robustness.json", "w") as f:
            import json
            def dt_h(x): return x.isoformat() if isinstance(x, datetime) else str(x)
            json.dump(results, f, indent=4, default=dt_h)
            
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
            
            # Fitness: Sharpe * Profit Factor / |MaxDrawdown|
            # Penalize high drawdown heavily
            sharpe = perf.get("sharpe_ratio", 0)
            pf = perf.get("profit_factor", 0)
            mdd = perf.get("max_drawdown", 100)
            
            score = (sharpe * pf) / (mdd + 0.1)
            
            if score > best_metric:
                best_metric = score
                best_params = params
                self.last_train_perf = perf
        
        return best_params
