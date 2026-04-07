import numpy as np
import random
from typing import List, Dict
from core.performance_tracker import PerformanceTracker

class MonteCarloSimulator:
    """
    Institutional Robustness Simulation.
    Uses Bootstrap Resampling (sampling with replacement) and Execution Shock injection.
    """

    def __init__(self, iterations: int = 2500):
        self.iterations = iterations

    def run(self, history: List[Dict], initial_balance: float = 1000.0) -> Dict:
        """
        Institutional Monte Carlo Suite (Step 8.2).
        Includes Bootstrap, Permutation, and Jitter tests.
        """
        if not history or len(history) < 10:
            return {
                "status": "INSUFFICIENT_DATA", 
                "robustness_score": 0,
                "message": f"Institutional certification requires at least 10 trades (Found: {len(history)}). Small sample size precludes Stress Testing."
            }

        pnls = np.array([t['pnl'] for t in history])
        all_final_balances = []
        all_max_drawdowns = []

        num_trades = len(pnls)
        
        for i in range(self.iterations):
            # 1. Randomized Approach Select
            # Alternating between Bootstrap (with replacement) and Permutation (without replacement/shuffle)
            if i % 2 == 0:
                indices = np.random.randint(0, num_trades, size=num_trades)
            else:
                indices = np.random.permutation(num_trades)
            
            sim_pnls = pnls[indices].copy()
            
            # 2. Jitter Cost Injection (Institutional Realism)
            # Randomly subtract 0.1 to 1.5 pips of 'unseen friction' (Step 8)
            jitter_penalty = np.random.uniform(0.1, 1.5, size=num_trades)
            sim_pnls -= jitter_penalty

            # 3. Path Stats
            cum_pnl = np.cumsum(sim_pnls)
            equity = initial_balance + cum_pnl
            
            # Risk of Ruin check
            if np.any(equity <= 0):
                all_final_balances.append(0.0)
                all_max_drawdowns.append(100.0)
                continue

            final_balance = equity[-1]
            peak = np.maximum.accumulate(equity)
            dd = (peak - equity) / peak * 100
            max_dd = np.max(dd)

            all_final_balances.append(final_balance)
            all_max_drawdowns.append(max_dd)

        # Confidence Intervals
        all_final_balances.sort()
        all_max_drawdowns.sort()
        
        # 95th Percentile Worst Case (5% mark for balance, 95% mark for DD)
        worst_case_balance = all_final_balances[int(self.iterations * 0.05)]
        worst_case_dd = all_max_drawdowns[int(self.iterations * 0.95)]
        
        # Expectancy under stress
        median_balance = np.median(all_final_balances)
        prob_of_ruin = (len([b for b in all_final_balances if b <= 0]) / self.iterations) * 100

        return {
            "status": "SUCCESS",
            "iterations": self.iterations,
            "median_final_balance": round(median_balance, 2),
            "worst_case_balance_95ci": round(worst_case_balance, 2),
            "worst_case_dd_95ci": f"{worst_case_dd:.2f}%",
            "probability_of_ruin": f"{prob_of_ruin:.2f}%",
            "robustness_score": self._calculate_robustness_score(history, worst_case_dd, prob_of_ruin)
        }

    def _calculate_robustness_score(self, history, worst_dd, ruin_prob):
        """
        Institutional Robustness Scoring (V4-ULTRA).
        Targets: Worst Case DD < 15%, Probability of Ruin = 0.00%
        """
        if not history: return 0
        
        # Base score starts at 100
        score = 100.0
        
        # 1. DD Penalty (Institutional target < 15%)
        if worst_dd > 15:
            excess = worst_dd - 15
            score -= (excess ** 1.5) * 2.5
        
        # 2. Ruin Penalty (Extreme sensitivity to ruin)
        if ruin_prob > 0:
            score -= (ruin_prob * 10) + 50 
        
        # 3. Sample Size Filter (Tiered Institutional Quality)
        # 100+ trades = Full Certification
        # 50-100 trades = Moderate Confidence (-0.1 per missing trade)
        # < 50 trades = Low Confidence (-0.2 per missing trade)
        count = len(history)
        if count < 100:
            if count >= 50:
                score -= (100 - count) * 0.1
            else:
                score -= (50 - count) * 0.2 + 5.0 # Baseline penalty for low count
        
        return max(0, min(100, round(score, 1)))

