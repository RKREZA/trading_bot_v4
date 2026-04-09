"""
TRADING BOT V4 — Stress Test Suite
==================================
Institutional "Worst Case" execution simulation.
Evaluates strategy robustness under toxic market conditions (high spread, high slippage).
"""

import logging
import copy
import numpy as np
from typing import List, Dict, Any
from backtesting.backtester import PortfolioBacktester
from core.performance_tracker import PerformanceTracker
from strategies import STRATEGY_REGISTRY

logger = logging.getLogger("trading_bot.stress_test")

class StressTester:
    """
    V4 Institutional Stress Tester.
    Runs multiple backtest passes with degraded execution conditions.
    """

    def __init__(self, config: Dict):
        self.config = config

    def run_stress_test(self, symbol: str, strategies: list, data: dict) -> Dict[str, Dict]:
        """
        Executes a battery of stress scenarios.
        
        Args:
            symbol (str): Trading pair.
            strategies (list): List of strategy instances.
            data (dict): Dict containing CandleArrays for all timeframes.
            
        Returns:
            Dict[str, Dict]: Results for each scenario.
        """
        results = {}
        initial_balance = len(strategies) * float(self.config.get("backtest", {}).get("initial_balance_per_strategy", 10000.0))
        
        fresh_strategies = self._create_fresh_strategies(strategies)
        
        baseline_history, baseline_equity = self._run_pass(symbol, fresh_strategies, data, self.config)
        baseline_metrics = PerformanceTracker.calculate_metrics(
            baseline_history, 
            initial_balance, 
            equity_curve=self._aggregate_equity(baseline_equity, len(fresh_strategies))
        )
        baseline_profit = baseline_metrics.get("net_profit", 0.0)
        
        results["BASELINE"] = {
            "metrics": baseline_metrics,
            "profit_retention": 100.0
        }

        fresh_strategies = self._create_fresh_strategies(strategies)
        shock_data = self._apply_spread_multiplier(data, 3.0)
        spread_config = copy.deepcopy(self.config)
        spread_config["symbols_config"][symbol]["spread_pips"] = spread_config["symbols_config"].get(symbol, {}).get("spread_pips", 6) * 3
        
        spread_history, spread_equity = self._run_pass(symbol, fresh_strategies, shock_data, spread_config)
        spread_metrics = PerformanceTracker.calculate_metrics(
            spread_history, 
            initial_balance, 
            equity_curve=self._aggregate_equity(spread_equity, len(fresh_strategies))
        )
        spread_profit = spread_metrics.get("net_profit", 0.0)
        results["SPREAD SHOCK"] = {
            "metrics": spread_metrics,
            "profit_retention": (spread_profit / baseline_profit * 100) if baseline_profit != 0 else 0.0
        }

        fresh_strategies = self._create_fresh_strategies(strategies)
        slip_config = copy.deepcopy(self.config)
        slip_config["symbols_config"][symbol]["spread_pips"] = slip_config["symbols_config"].get(symbol, {}).get("spread_pips", 6) * 5
        slip_config["execution"]["entry_slippage_points"] = slip_config["execution"].get("entry_slippage_points", 1.5) * 5
        slip_config["execution"]["sl_exit_slippage_points"] = slip_config["execution"].get("sl_exit_slippage_points", 2.5) * 5
        
        slip_history, slip_equity = self._run_pass(symbol, fresh_strategies, data, slip_config)
        slip_metrics = PerformanceTracker.calculate_metrics(
            slip_history, 
            initial_balance, 
            equity_curve=self._aggregate_equity(slip_equity, len(fresh_strategies))
        )
        slip_profit = slip_metrics.get("net_profit", 0.0)
        results["SLIPPAGE SHOCK"] = {
            "metrics": slip_metrics,
            "profit_retention": (slip_profit / baseline_profit * 100) if baseline_profit != 0 else 0.0
        }

        fresh_strategies = self._create_fresh_strategies(strategies)
        toxic_config = copy.deepcopy(self.config)
        toxic_config["symbols_config"][symbol]["spread_pips"] = toxic_config["symbols_config"].get(symbol, {}).get("spread_pips", 6) * 5
        toxic_config["execution"]["entry_slippage_points"] = toxic_config["execution"].get("entry_slippage_points", 1.5) * 5
        toxic_config["execution"]["sl_exit_slippage_points"] = toxic_config["execution"].get("sl_exit_slippage_points", 2.5) * 5
        
        toxic_history, toxic_equity = self._run_pass(symbol, fresh_strategies, shock_data, toxic_config)
        toxic_metrics = PerformanceTracker.calculate_metrics(
            toxic_history, 
            initial_balance, 
            equity_curve=self._aggregate_equity(toxic_equity, len(fresh_strategies))
        )
        toxic_profit = toxic_metrics.get("net_profit", 0.0)
        results["TOXIC FLOW"] = {
            "metrics": toxic_metrics,
            "profit_retention": (toxic_profit / baseline_profit * 100) if baseline_profit != 0 else 0.0
        }

        return results

    def _create_fresh_strategies(self, original_strategies: list) -> list:
        """Creates fresh strategy instances to avoid state pollution."""
        fresh = []
        for strat in original_strategies:
            st_type = strat.__class__.__name__.replace("Strategy", "").upper()
            if st_type in STRATEGY_REGISTRY:
                fresh.append(STRATEGY_REGISTRY[st_type](strat.strategy_id, self.config))
        return fresh

    def _run_pass(self, symbol, strategies, data, config):
        """Helper to run a single backtest pass."""
        backtester = PortfolioBacktester(config)
        
        symbol_cfg = config.get("symbols_config", {}).get(symbol, {})
        primary_tf = symbol_cfg.get("backtest_timeframe", "M5")
        primary_data = data.get(primary_tf, data.get("M5"))
        
        return backtester.run(
            symbol, 
            strategies, 
            primary_data, 
            data.get("H1"), 
            data.get("M15"), 
            data.get("M5"), 
            data.get("M1")
        )

    def _apply_spread_multiplier(self, data, multiplier):
        """Creates a copy of data with boosted spreads."""
        new_data = {}
        for tf, candles in data.items():
            c_copy = copy.copy(candles)
            c_copy.spread = np.array(candles.spread) * multiplier
            new_data[tf] = c_copy
        return new_data

    def _aggregate_equity(self, equity_history, num_strategies):
        """Helper to aggregate multi-strategy equity into a single curve."""
        if not equity_history:
            return []
        import pandas as pd
        df = pd.DataFrame(equity_history)
        return df.groupby('time')['equity'].sum().tolist()
