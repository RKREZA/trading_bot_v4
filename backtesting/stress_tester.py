"""
TRADING BOT V4 — Stress Test Suite
==================================
Institutional "Worst Case" execution simulation.
Evaluates strategy robustness under toxic market conditions (high spread, high slippage).
"""

import logging
import copy
from typing import List, Dict, Any
from backtesting.backtester import PortfolioBacktester
from core.performance_tracker import PerformanceTracker

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
        
        # 1. Baseline Performance
        logger.info("Stress Test: Running Baseline Scenario...")
        baseline_history, baseline_equity = self._run_pass(symbol, strategies, data, self.config)
        baseline_metrics = PerformanceTracker.calculate_metrics(
            baseline_history, 
            len(strategies) * 1000.0, 
            equity_curve=self._aggregate_equity(baseline_equity, len(strategies))
        )
        baseline_profit = baseline_metrics.get("net_profit", 0.0)
        
        results["baseline"] = {
            "metrics": baseline_metrics,
            "profit_retention": 100.0
        }

        # 2. Scenario: Spread Shock (3x Spread)
        logger.info("Stress Test: Running Spread Shock (3x)...")
        shock_data = self._apply_spread_multiplier(data, 3.0)
        s_history, s_equity = self._run_pass(symbol, strategies, shock_data, self.config)
        s_metrics = PerformanceTracker.calculate_metrics(
            s_history, 
            len(strategies) * 1000.0, 
            equity_curve=self._aggregate_equity(s_equity, len(strategies))
        )
        results["spread_shock"] = {
            "metrics": s_metrics,
            "profit_retention": (s_metrics.get("net_profit", 0.0) / baseline_profit * 100) if baseline_profit > 0 else 0.0
        }

        # 3. Scenario: Slippage Shock (5x Slippage)
        logger.info("Stress Test: Running Slippage Shock (5x)...")
        slip_config = copy.deepcopy(self.config)
        exec_cfg = slip_config.setdefault("execution", {})
        exec_cfg["entry_slippage_pips"] = exec_cfg.get("entry_slippage_pips", 0.15) * 5.0
        exec_cfg["sl_exit_slippage_pips"] = exec_cfg.get("sl_exit_slippage_pips", 0.25) * 5.0
        
        sl_history, sl_equity = self._run_pass(symbol, strategies, data, slip_config)
        sl_metrics = PerformanceTracker.calculate_metrics(
            sl_history, 
            len(strategies) * 1000.0, 
            equity_curve=self._aggregate_equity(sl_equity, len(strategies))
        )
        results["slippage_shock"] = {
            "metrics": sl_metrics,
            "profit_retention": (sl_metrics.get("net_profit", 0.0) / baseline_profit * 100) if baseline_profit > 0 else 0.0
        }

        # 4. Scenario: Toxic Flow (Combined Shock)
        logger.info("Stress Test: Running Toxic Flow Scenario...")
        toxic_history, toxic_equity = self._run_pass(symbol, strategies, shock_data, slip_config)
        toxic_metrics = PerformanceTracker.calculate_metrics(
            toxic_history, 
            len(strategies) * 1000.0, 
            equity_curve=self._aggregate_equity(toxic_equity, len(strategies))
        )
        results["toxic_flow"] = {
            "metrics": toxic_metrics,
            "profit_retention": (toxic_metrics.get("net_profit", 0.0) / baseline_profit * 100) if baseline_profit > 0 else 0.0
        }

        return results

    def _run_pass(self, symbol, strategies, data, config):
        """Helper to run a single backtest pass."""
        backtester = PortfolioBacktester(config)
        
        # Determine primary timeframe from config (Harmonize with backtest.py)
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
            # Deep copy the CandleArray to avoid mutating original data
            c_copy = copy.copy(candles)
            c_copy.spread = candles.spread * multiplier
            new_data[tf] = c_copy
        return new_data

    def _aggregate_equity(self, equity_history, num_strategies):
        """Helper to aggregate multi-strategy equity into a single curve."""
        if not equity_history:
            return []
        import pandas as pd
        df = pd.DataFrame(equity_history)
        return df.groupby('time')['equity'].sum().tolist()
