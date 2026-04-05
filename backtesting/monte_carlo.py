import numpy as np
import random
from typing import List, Dict
from core.performance_tracker import PerformanceTracker

class MonteCarloSimulator:
    """
    Institutional Robustness Simulation.
    Randomizes trade order and applies execution noise to history.
    """

    def __init__(self, iterations: int = 2000):
        self.iterations = iterations

    def run(self, history: List[Dict], initial_balance: float = 1000.0) -> Dict:
        """
        Runs multiple simulation paths.
        """
        if not history:
            return {"status": "No history to simulate"}

        pnls = [t['pnl'] for t in history]
        all_final_balances = []
        all_max_drawdowns = []

        for _ in range(self.iterations):
            # 1. Randomize Trade Sequence
            sim_pnls = pnls.copy()
            random.shuffle(sim_pnls)
            
            # 2. Add Execution Noise (±0.5 pip randomized slippage penalty)
            # This would subtract a small amount from each trade to simulate worst-case.
            noise = [p - random.uniform(0, 5) for p in sim_pnls] # Rough dollar-value noise
            
            # 3. Calculate Path Stats
            cum_pnl = np.cumsum(noise)
            equity = initial_balance + cum_pnl
            final_balance = equity[-1]
            
            peak = np.maximum.accumulate(equity)
            dd = (peak - equity) / peak * 100
            max_dd = np.max(dd)

            all_final_balances.append(final_balance)
            all_max_drawdowns.append(max_dd)

        # Confidence Interval (95th percentile Worst Case)
        all_final_balances.sort()
        worst_case_balance = all_final_balances[int(self.iterations * 0.05)]
        worst_case_dd = np.percentile(all_max_drawdowns, 95)

        return {
            "iterations": self.iterations,
            "median_final_balance": round(np.median(all_final_balances), 2),
            "worst_case_balance_95ci": round(worst_case_balance, 2),
            "worst_case_dd_95ci": f"{worst_case_dd:.2f}%",
            "probability_of_ruin": f"{(len([b for b in all_final_balances if b < initial_balance]) / self.iterations * 100):.2f}%"
        }
