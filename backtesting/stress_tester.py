import logging
import copy
from typing import List, Dict, Any
from backtesting.backtester import PortfolioBacktester
from core.performance_tracker import PerformanceTracker

logger = logging.getLogger("trading_bot.stress_tester")

class StressTester:
    """
    V4 Institutional Stress Testing Engine.
    Runs 'Pessimistic' simulations to find the breaking point of strategies.
    """

    def __init__(self, config: dict):
        self.config = config

    def run_stress_test(self, symbol: str, strategies: list, data: dict) -> Dict[str, Any]:
        """
        Runs multiple backtest passes with increasingly degraded conditions.
        """
        results = {}
        
        # Scenarios to test (Step 8.3)
        scenarios = {
            "baseline": {"spread_mult": 1.0, "slip_mult": 1.0},
            "high_spread": {"spread_mult": 2.0, "slip_mult": 1.0},
            "spread_blowout": {"spread_mult": 3.0, "slip_mult": 1.0},
            "high_slippage": {"spread_mult": 1.0, "slip_mult": 2.5},
            "black_swan_gap": {"spread_mult": 2.0, "slip_mult": 5.0},
            "pessimistic_bundle": {"spread_mult": 3.0, "slip_mult": 2.5},
        }

        m5 = data.get("M5")
        h1 = data.get("H1")
        m15 = data.get("M15")
        m1 = data.get("M1")

        for name, params in scenarios.items():
            logger.info(f"Running Stress Scenario: {name}...")
            
            # Create a degraded config
            stress_config = copy.deepcopy(self.config)
            
            # Inject stress into symbol config
            if "symbols_config" not in stress_config:
                stress_config["symbols_config"] = {}
            if symbol not in stress_config["symbols_config"]:
                stress_config["symbols_config"][symbol] = {}
            
            s_cfg = stress_config["symbols_config"][symbol]
            s_cfg["spread_pips"] = float(s_cfg.get("spread_pips", 20.0)) * params["spread_mult"]
            
            # Create specialized backtester
            bt = PortfolioBacktester(stress_config)
            
            # Reset strategy status for the new run
            scenario_strategies = copy.deepcopy(strategies)
            for s in scenario_strategies:
                s.enabled = True
                
            # Inject slippage multiplier directly into simulator
            mult = params["slip_mult"]
            bt.simulator.entry_slip_pips *= mult
            bt.simulator.tp_exit_slip_pips *= mult
            bt.simulator.sl_exit_slip_pips *= mult
            
            # Resolve target timeframe data
            target_tf = self.config.get("symbols_config", {}).get(symbol, {}).get("backtest_timeframe", "M5")
            target_tf_data = m5 if target_tf == "M5" else (m15 if target_tf == "M15" else h1)
            
            history, equity_history = bt.run(symbol, scenario_strategies, target_tf_data, h1, m15, m5, m1)
            
            partition_initial = float(self.config.get("initial_balance", 1000.0))
            total_initial = len(strategies) * partition_initial
            
            # Aggregate metrics
            metrics = PerformanceTracker.calculate_metrics(history, total_initial)
            
            results[name] = {
                "metrics": metrics,
                "trade_count": len(history),
                "profit_retention": 0.0
            }

        # Calculate retention relative to baseline
        baseline_profit = results["baseline"]["metrics"].get("net_profit", 0)
        for name in results:
            if baseline_profit != 0:
                results[name]["profit_retention"] = (results[name]["metrics"].get("net_profit", 0) / baseline_profit) * 100
            else:
                results[name]["profit_retention"] = 0.0

        return results

    def summarize(self, stress_results: dict):
        """Prints a professional stress test summary."""
        print("\n" + "!"*50)
        print("INSTITUTIONAL STRESS TEST SUMMARY")
        print("!"*50)
        
        for name, res in stress_results.items():
            m = res["metrics"]
            profit = m.get("net_profit", 0)
            retention = res["profit_retention"]
            status = "[PASS]" if profit > 0 else "[FAIL]"
            
            print(f"{status} Scenario: {name.upper()}")
            print(f"      Net Profit: ${profit:,.2f} ({retention:.1f}% retention)")
            print(f"      Max Drawdown: {m.get('max_drawdown', 'N/A')}")
            print("-" * 30)
