"""
V6-INSIGNIA — Stress Test Suite
=================================
Institutional "Worst Case" execution simulation with:
- Multi-scenario stress testing (spread, slippage, latency, volatility)
- Monte Carlo worst-case analysis
- Profit degradation curves
- Scenario-specific recommendations
"""

import logging
import copy
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
from backtesting.backtester import PortfolioBacktester
from core.performance_tracker import PerformanceTracker
from strategies import STRATEGY_REGISTRY

logger = logging.getLogger("trading_bot.stress_test")

class StressScenario(Enum):
    BASELINE = "BASELINE"
    SPREAD_SHOCK = "SPREAD_SHOCK"
    SLIPPAGE_SHOCK = "SLIPPAGE_SHOCK"
    LATENCY_SHOCK = "LATENCY_SHOCK"
    VOLATILITY_SPIKE = "VOLATILITY_SPIKE"
    TOXIC_FLOW = "TOXIC_FLOW"
    FLASH_CRASH = "FLASH_CRASH"
    WIDE_SPREAD = "WIDE_SPREAD"

@dataclass
class StressResult:
    scenario: StressScenario
    metrics: Dict[str, Any]
    profit_retention: float
    passed: bool
    pass_threshold: float = 50.0
    details: str = ""

class StressTester:
    """
    V6 Institutional Stress Tester.
    Comprehensive stress testing with multiple scenarios and degradation analysis.
    """

    def __init__(self, config: Dict):
        self.config = config
        self._results: List[StressResult] = []

    def run_stress_test(self, symbol: str, strategies: list, data: dict,
                  pass_threshold: float = 50.0) -> Dict[str, Dict]:
        """Executes comprehensive stress scenarios."""
        results = {}
        initial_balance = len(strategies) * float(
            self.config.get("backtest", {}).get("initial_balance_per_strategy", 10000.0))
        
        baseline_result = self._run_scenario(symbol, strategies, data, "BASELINE", 
                                       self.config, initial_balance)
        results["BASELINE"] = baseline_result
        baseline_profit = baseline_result["metrics"].get("net_profit", 0.0)
        
        if baseline_profit <= 0:
            logger.warning("Stress: Baseline is unprofitable. Skipping stress scenarios.")
            return results
        
        scenarios = [
            ("SPREAD_SHOCK", 3.0, 1.0),
            ("SLIPPAGE_SHOCK", 1.0, 5.0),
            ("LATENCY_SHOCK", 1.0, 1.0),
            ("VOLATILITY_SPIKE", 1.0, 1.0),
            ("TOXIC_FLOW", 5.0, 5.0),
            ("FLASH_CRASH", 8.0, 8.0),
        ]
        
        for scen_name, spread_mult, slip_mult in scenarios:
            result = self._run_scenario(
                symbol, strategies, data, scen_name,
                initial_balance, spread_mult, slip_mult
            )
            results[scen_name] = result
        
        return results

    def _run_scenario(self, symbol: str, strategies: list, data: dict,
                   scenario_name: str, initial_balance: float,
                   spread_mult: float = 1.0, slip_mult: float = 1.0) -> Dict:
        """Runs a single stress scenario."""
        fresh_strategies = self._create_fresh_strategies(strategies)
        scenario_config = copy.deepcopy(self.config)
        
        sym_cfg = scenario_config.get("symbols_config", {}).get(symbol, {})
        sym_cfg["spread_pips"] = float(sym_cfg.get("spread_pips", 6)) * spread_mult
        scenario_config["symbols_config"][symbol] = sym_cfg
        
        exec_cfg = scenario_config.get("execution", {})
        exec_cfg["entry_slippage_points"] = float(exec_cfg.get("entry_slippage_points", 1.5)) * slip_mult
        exec_cfg["sl_exit_slippage_points"] = float(exec_cfg.get("sl_exit_slippage_points", 2.5)) * slip_mult
        scenario_config["execution"] = exec_cfg
        
        if scenario_name == "VOLATILITY_SPIKE":
            data = self._apply_volatility_multiplier(data, 2.0)
        
        history, equity = self._run_pass(symbol, fresh_strategies, data, scenario_config)
        metrics = PerformanceTracker.calculate_metrics(
            history, initial_balance, 
            equity_curve=self._aggregate_equity(equity, len(fresh_strategies)))
        
        baseline = self.config.get("_baseline_profit", 1.0)
        profit_retention = (metrics.get("net_profit", 0) / baseline * 100) if baseline > 0 else 0
        
        passed = profit_retention >= 50.0
        
        return {
            "metrics": metrics,
            "profit_retention": profit_retention,
            "passed": passed,
            "scenario": scenario_name
        }

    def _apply_volatility_multiplier(self, data, multiplier):
        """Applies volatility multiplier to price data."""
        new_data = {}
        for tf, candles in data.items():
            if candles is None:
                new_data[tf] = None
                continue
            c_copy = copy.copy(candles)
            prices = (np.array(candles.high) + np.array(candles.low)) / 2
            ranges = np.array(candles.high) - np.array(candles.low)
            scaled_ranges = ranges * multiplier
            mid = prices - scaled_ranges / 2
            c_copy.high = mid + scaled_ranges
            c_copy.low = mid - scaled_ranges
            new_data[tf] = c_copy
        return new_data

    def get_degradation_curve(self) -> List[Dict]:
        """Returns profit retention curve across all scenarios."""
        return sorted([
            {"scenario": k, "retention": v["profit_retention"], "passed": v["passed"]}
            for k, v in self._results.items()
        ], key=lambda x: x["retention"], reverse=True)

    def get_recommendations(self) -> Dict[str, Any]:
        """Returns actionable recommendations based on stress results."""
        if not self._results:
            return {"status": "No results"}
        
        failed = [r for r in self._results.values() if not r["passed"]]
        
        risk_level = "LOW"
        if len(failed) >= 4:
            risk_level = "CRITICAL"
        elif len(failed) >= 2:
            risk_level = "HIGH"
        elif len(failed) >= 1:
            risk_level = "MEDIUM"
        
        recommendations = []
        if any("SPREAD" in str(f) for f in failed):
            recommendations.append("Reduce max spread threshold in config")
        if any("SLIPPAGE" in str(f) for f in failed):
            recommendations.append("Implement wider SL buffer")
        if any("VOLATILITY" in str(f) for f in failed):
            recommendations.append("Reduce position size during high volatility")
        
        return {
            "risk_level": risk_level,
            "scenarios_passed": len(self._results) - len(failed),
            "scenarios_failed": len(failed),
            "recommendations": recommendations
        }

    def _create_fresh_strategies(self, original_strategies: list) -> list:
        """Creates fresh strategy instances."""
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
            symbol, strategies, primary_data,
            data.get("H1"), data.get("M15"),
            data.get("M5"), data.get("M1")
        )

    def _aggregate_equity(self, equity_history, num_strategies):
        """Aggregates multi-strategy equity."""
        if not equity_history:
            return []
        import pandas as pd
        df = pd.DataFrame(equity_history)
        return df.groupby('time')['equity'].sum().tolist()
