import itertools
import copy
import logging
from typing import List, Dict, Tuple, Any
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

    def run_validation(self, symbol: str, h1: Any, m15: Any, m5: Any, d1: Any, 
                       train_days: int = 14, test_days: int = 7, mode: str = "anchored") -> List[Dict]:
        """
        Institutional-grade Walk-Forward Optimization (WFO) for M5 Sniper.
        Hierarchy: H1 (Zones), M15 (Bias), M5 (Entries)
        """
        results = []
        if not m5: return []
        
        # M5 is now the primary time-base for windows
        m5_list = [{"time": t} for t in m5.time] # for time filtering
        start_date = datetime.fromtimestamp(m5.time[0], tz=timezone.utc)
        end_date = datetime.fromtimestamp(m5.time[-1], tz=timezone.utc)
        
        current_train_start = start_date
        param_popularity = {} 
        
        while True:
            current_train_end = current_train_start + pd.DateOffset(days=train_days)
            current_test_end = current_train_end + pd.DateOffset(days=test_days)
            
            if current_test_end > end_date: break
                
            logger.info(f"--- WFO WINDOW: Train {current_train_start.date()} -> {current_train_end.date()} | Test {current_train_end.date()} -> {current_test_end.date()} ---")
            
            train_data = {
                "h1": self._filter_arr(h1, current_train_start, current_train_end),
                "m15": self._filter_arr(m15, current_train_start, current_train_end),
                "m5": self._filter_arr(m5, current_train_start, current_train_end),
                "d1": self._filter_arr(d1, current_train_start, current_train_end)
            }
            
            test_data = {
                "h1": self._filter_arr(h1, current_train_end, current_test_end),
                "m15": self._filter_arr(m15, current_train_end, current_test_end),
                "m5": self._filter_arr(m5, current_train_end, current_test_end),
                "d1": self._filter_arr(d1, current_train_end, current_test_end)
            }

            if len(train_data["m5"].time) < 100 or len(test_data["m5"].time) < 50:
                if mode == "rolling": current_train_start += pd.DateOffset(days=test_days)
                else: train_days += test_days
                continue

            # 1. OPTIMIZATION (IS)
            param_grid = {
                "swing_lookback": [7, 12, 18],
                "min_wick_pct": [30.0, 40.0, 50.0],
                "min_body_pct": [15.0, 25.0],
                "fixed_rr": [2.0, 3.0, 5.0]
            }
            
            best_params = self._optimize(symbol, train_data, param_grid)
            
            # 2. VALIDATION (OOS)
            test_config = copy.deepcopy(self.config)
            if "price_action" not in test_config["strategy_defaults"]:
                test_config["strategy_defaults"]["price_action"] = {}
            test_config["strategy_defaults"]["price_action"].update(best_params)
            
            test_strategy = StrategyEngine(test_config)
            tester = BacktestEngine(test_config, test_strategy)
            test_perf = tester.run(symbol, test_data["h1"], test_data["m15"], 
                                   test_data["m5"], test_data["d1"], quiet=True)
            
            # 3. SCORE & RECORD
            results.append({
                "window": f"{current_train_end.date()} to {current_test_end.date()}",
                "best_params": best_params,
                "is_metrics": self.last_train_perf,
                "oos_metrics": test_perf,
                "consistency": 1.0 # placeholder
            })
            
            if mode == "rolling": current_train_start += pd.DateOffset(days=test_days)
            else: train_days += test_days 

        return results

    def _filter_arr(self, arr: CandleArray, start: datetime, end: datetime) -> CandleArray:
        mask = (arr.time >= start.timestamp()) & (arr.time < end.timestamp())
        return arr[mask]

    def _optimize(self, symbol, data, grid) -> dict:
        best_metric = -999999; best_params = {}
        keys, values = zip(*grid.items())
        
        for v in itertools.product(*values):
            params = dict(zip(keys, v))
            tmp_config = copy.deepcopy(self.config)
            if "price_action" not in tmp_config["strategy_defaults"]:
                tmp_config["strategy_defaults"]["price_action"] = {}
            tmp_config["strategy_defaults"]["price_action"].update(params)
            
            strat = StrategyEngine(tmp_config)
            tester = BacktestEngine(tmp_config, strat)
            perf = tester.run(symbol, data["h1"], data["m15"], data["m5"], data["d1"], quiet=True)
            
            score = perf.get("net_profit", 0)
            if score > best_metric:
                best_metric = score; best_params = params; self.last_train_perf = perf
        
        return best_params
