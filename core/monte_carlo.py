"""
TRADING BOT V3 — Monte Carlo Simulation Engine
================================================
Simulates thousands of randomized trade sequence permutations to answer:

  "If these same trades happened in a different order, how bad could it get?"

This separates skill (edge) from luck (sequence order) and produces:
  - P5 / P50 / P95 equity curves
  - Drawdown distribution (worst-case at various confidence levels)
  - Probability of ruin (balance < 50% of starting capital)
  - Confidence intervals for profit factor and win rate
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple


class MonteCarlo:
    """
    Trade-sequence Monte Carlo simulator.

    Usage:
        mc = MonteCarlo(trades, initial_balance=1000, n_simulations=2000)
        result = mc.run()
        mc.print_report(result)
    """

    def __init__(self, trades: List[dict], initial_balance: float = 1000.0,
                 n_simulations: int = 2000, ruin_threshold_pct: float = 50.0):
        """
        Args:
            trades: List of completed trade dicts with 'pnl' key
            initial_balance: Starting capital
            n_simulations: Number of random permutations to run
            ruin_threshold_pct: % of starting balance below which = "ruin"
        """
        self.pnl_list = [t['pnl'] for t in trades if 'pnl' in t]
        self.initial_balance = initial_balance
        self.n_sims = n_simulations
        self.ruin_level = initial_balance * (1 - ruin_threshold_pct / 100)
        self.n_trades = len(self.pnl_list)

    def run(self) -> Dict:
        """
        Run Monte Carlo simulation.

        Returns dict with:
            - equity_percentiles: {p5, p50, p95} equity curves
            - final_balance: {p5, p50, p95} final balance
            - max_drawdown: {p5, p50, p95, worst} max drawdown %
            - prob_ruin: probability of hitting ruin threshold
            - prob_profit: probability of ending profitable
            - expected_profit: median final profit $
            - confidence_interval: (lower, upper) 90% CI for net profit
        """
        if self.n_trades == 0:
            return self._empty_result()

        rng = np.random.default_rng(seed=42)
        pnl_arr = np.array(self.pnl_list)

        final_balances = []
        max_drawdowns  = []
        ruin_count     = 0
        all_equity     = []

        for _ in range(self.n_sims):
            # Shuffle trade order
            shuffled = rng.permutation(pnl_arr)
            equity   = np.concatenate([[self.initial_balance],
                                       self.initial_balance + np.cumsum(shuffled)])

            # Max drawdown for this path
            rolling_max = np.maximum.accumulate(equity)
            dd_pct      = (rolling_max - equity) / rolling_max * 100
            max_dd      = float(dd_pct.max())

            final = float(equity[-1])
            final_balances.append(final)
            max_drawdowns.append(max_dd)

            if np.any(equity <= self.ruin_level):
                ruin_count += 1

            all_equity.append(equity)

        # Stack all equity curves (same length — same trade count)
        eq_matrix = np.array(all_equity)  # shape: (n_sims, n_trades+1)

        # Percentile equity curves
        p5_curve  = np.percentile(eq_matrix, 5,  axis=0).tolist()
        p50_curve = np.percentile(eq_matrix, 50, axis=0).tolist()
        p95_curve = np.percentile(eq_matrix, 95, axis=0).tolist()

        final_arr = np.array(final_balances)
        dd_arr    = np.array(max_drawdowns)

        profits = final_arr - self.initial_balance
        ci_low  = float(np.percentile(profits, 5))
        ci_high = float(np.percentile(profits, 95))

        return {
            "n_simulations":   self.n_sims,
            "n_trades":        self.n_trades,
            "initial_balance": self.initial_balance,
            "equity_p5":       p5_curve,
            "equity_p50":      p50_curve,
            "equity_p95":      p95_curve,
            "final_p5":        round(float(np.percentile(final_arr, 5)),  2),
            "final_p50":       round(float(np.percentile(final_arr, 50)), 2),
            "final_p95":       round(float(np.percentile(final_arr, 95)), 2),
            "final_worst":     round(float(final_arr.min()),  2),
            "final_best":      round(float(final_arr.max()),  2),
            "max_dd_p50":      round(float(np.percentile(dd_arr, 50)), 2),
            "max_dd_p95":      round(float(np.percentile(dd_arr, 95)), 2),
            "max_dd_worst":    round(float(dd_arr.max()),  2),
            "prob_ruin_pct":   round(ruin_count / self.n_sims * 100, 2),
            "prob_profit_pct": round(float((final_arr > self.initial_balance).mean() * 100), 2),
            "expected_profit": round(float(np.median(profits)), 2),
            "ci_90_low":       round(ci_low,  2),
            "ci_90_high":      round(ci_high, 2),
        }

    def _empty_result(self) -> Dict:
        return {
            "n_simulations": 0, "n_trades": 0,
            "prob_ruin_pct": 0, "prob_profit_pct": 0,
            "expected_profit": 0, "ci_90_low": 0, "ci_90_high": 0,
            "max_dd_p50": 0, "max_dd_p95": 0, "max_dd_worst": 0,
        }

    @staticmethod
    def print_report(result: Dict, strategy_id: str = "") -> None:
        """Print a formatted Monte Carlo report to stdout."""
        label = f" [{strategy_id}]" if strategy_id else ""
        sep = "=" * 55
        print(f"\n{sep}")
        print(f"  MONTE CARLO SIMULATION{label}")
        print(f"  {result['n_simulations']} permutations × {result['n_trades']} trades")
        print(sep)
        print(f"  Final Balance (P5  / worst case):  ${result['final_p5']}")
        print(f"  Final Balance (P50 / median):       ${result['final_p50']}")
        print(f"  Final Balance (P95 / best case):    ${result['final_p95']}")
        print(f"  90% Confidence Interval (profit):   ${result['ci_90_low']} -> ${result['ci_90_high']}")
        print(f"")
        print(f"  Max Drawdown (P50 / typical):       {result['max_dd_p50']}%")
        print(f"  Max Drawdown (P95 / near-worst):    {result['max_dd_p95']}%")
        print(f"  Max Drawdown (absolute worst):      {result['max_dd_worst']}%")
        print(f"")
        print(f"  Probability of Profit:              {result['prob_profit_pct']}%")
        print(f"  Probability of Ruin (<50% capital): {result['prob_ruin_pct']}%")
        print(f"  Expected Profit (median):           ${result['expected_profit']}")
        print(sep)

        # Grade
        ruin_ok   = result["prob_ruin_pct"] < 5
        profit_ok = result["prob_profit_pct"] > 70
        dd_ok     = result["max_dd_p95"] < 25

        grade_pts = sum([ruin_ok, profit_ok, dd_ok])
        grade = {3: "A (Institutional Grade)",
                 2: "B (Retail Acceptable)",
                 1: "C (Needs Improvement)",
                 0: "D (Not Viable)"}.get(grade_pts, "?")
        print(f"  Robustness Grade: {grade}")
        if not ruin_ok:
            print(f"  ⚠  Ruin probability {result['prob_ruin_pct']}% exceeds 5% threshold")
        if not profit_ok:
            print(f"  ⚠  Win probability {result['prob_profit_pct']}% below 70% target")
        if not dd_ok:
            print(f"  ⚠  Near-worst drawdown {result['max_dd_p95']}% exceeds 25% threshold")
        print(sep)
