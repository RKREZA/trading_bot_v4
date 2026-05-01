import numpy as np
from typing import List, Dict, Any


class MonteCarloSimulator:
    """
    Institutional Robustness Simulation.
    Bootstrap resampling + permutation + execution shock injection.
    Returns full distribution data for frontend charting.
    """

    MIN_ITERATIONS = 100
    MAX_ITERATIONS = 5000

    def __init__(self, iterations: int = 2500, seed: int = 42):
        self.iterations = max(self.MIN_ITERATIONS, min(self.MAX_ITERATIONS, iterations))
        self._rng = np.random.default_rng(seed)

    def run(self, history: List[Dict], initial_balance: float = 1000.0) -> Dict[str, Any]:
        if not history or len(history) < 10:
            return {
                "status": "INSUFFICIENT_DATA",
                "robustness_score": 0,
                "message": f"Requires at least 10 trades (found: {len(history)}).",
            }

        returns = np.array([
            t["pnl"] / max(
                t.get("balance_at_start", t.get("final_balance", initial_balance) - t.get("pnl", 0)),
                initial_balance * 0.01,
            )
            for t in history
        ])
        num_trades = len(returns)

        all_final_balances = np.empty(self.iterations)
        all_max_drawdowns = np.empty(self.iterations)
        all_sharpe = np.empty(self.iterations)
        equity_paths: List[np.ndarray] = []
        drawdown_paths: List[np.ndarray] = []

        sample_every = max(1, self.iterations // 200)

        for i in range(self.iterations):
            if i % 2 == 0:
                indices = self._rng.integers(0, num_trades, size=num_trades)
            else:
                indices = self._rng.permutation(num_trades)

            sim_returns = returns[indices].copy()

            jitter = self._rng.uniform(0.0005, 0.0020, size=num_trades)
            sim_returns -= jitter

            equity = initial_balance * np.cumprod(1 + sim_returns)

            if np.any(equity <= 0):
                all_final_balances[i] = 0.0
                all_max_drawdowns[i] = 100.0
                all_sharpe[i] = -10.0
                continue

            all_final_balances[i] = equity[-1]

            peak = np.maximum.accumulate(equity)
            dd = (peak - equity) / peak * 100
            all_max_drawdowns[i] = np.max(dd)

            mean_r = np.mean(sim_returns)
            std_r = np.std(sim_returns)
            all_sharpe[i] = (mean_r / std_r * np.sqrt(252)) if std_r > 0 else 0.0

            if i % sample_every == 0:
                equity_paths.append(equity.tolist())
                drawdown_paths.append(dd.tolist())

        sorted_bal = np.sort(all_final_balances)
        sorted_dd = np.sort(all_max_drawdowns)
        sorted_sharpe = np.sort(all_sharpe)

        p05 = int(self.iterations * 0.05)
        p25 = int(self.iterations * 0.25)
        p50 = int(self.iterations * 0.50)
        p75 = int(self.iterations * 0.75)
        p95 = int(self.iterations * 0.95)

        prob_ruin = float(np.sum(all_final_balances <= 0) / self.iterations * 100)
        prob_profit = float(np.sum(all_final_balances > initial_balance) / self.iterations * 100)

        bal_hist, bal_edges = np.histogram(all_final_balances[all_final_balances > 0], bins=50)
        dd_hist, dd_edges = np.histogram(all_max_drawdowns, bins=50)
        sharpe_hist, sharpe_edges = np.histogram(all_sharpe[all_sharpe > -10], bins=50)

        return {
            "status": "SUCCESS",
            "iterations": self.iterations,
            "num_trades": num_trades,

            "summary": {
                "median_final_balance": round(float(sorted_bal[p50]), 2),
                "worst_case_balance_5pct": round(float(sorted_bal[p05]), 2),
                "best_case_balance_95pct": round(float(sorted_bal[p95]), 2),
                "worst_case_dd_95pct": round(float(sorted_dd[p95]), 2),
                "median_dd": round(float(sorted_dd[p50]), 2),
                "median_sharpe": round(float(sorted_sharpe[p50]), 2),
                "probability_of_ruin": round(prob_ruin, 2),
                "probability_of_profit": round(prob_profit, 2),
                "robustness_score": self._calculate_robustness_score(
                    num_trades, float(sorted_dd[p95]), prob_ruin
                ),
            },

            "distributions": {
                "balance": {
                    "percentiles": {
                        "p5": round(float(sorted_bal[p05]), 2),
                        "p25": round(float(sorted_bal[p25]), 2),
                        "p50": round(float(sorted_bal[p50]), 2),
                        "p75": round(float(sorted_bal[p75]), 2),
                        "p95": round(float(sorted_bal[p95]), 2),
                    },
                    "histogram": {
                        "counts": bal_hist.tolist(),
                        "edges": [round(float(e), 2) for e in bal_edges.tolist()],
                    },
                },
                "drawdown": {
                    "percentiles": {
                        "p5": round(float(sorted_dd[p05]), 2),
                        "p25": round(float(sorted_dd[p25]), 2),
                        "p50": round(float(sorted_dd[p50]), 2),
                        "p75": round(float(sorted_dd[p75]), 2),
                        "p95": round(float(sorted_dd[p95]), 2),
                    },
                    "histogram": {
                        "counts": dd_hist.tolist(),
                        "edges": [round(float(e), 2) for e in dd_edges.tolist()],
                    },
                },
                "sharpe": {
                    "percentiles": {
                        "p5": round(float(sorted_sharpe[p05]), 2),
                        "p25": round(float(sorted_sharpe[p25]), 2),
                        "p50": round(float(sorted_sharpe[p50]), 2),
                        "p75": round(float(sorted_sharpe[p75]), 2),
                        "p95": round(float(sorted_sharpe[p95]), 2),
                    },
                    "histogram": {
                        "counts": sharpe_hist.tolist(),
                        "edges": [round(float(e), 2) for e in sharpe_edges.tolist()],
                    },
                },
            },

            "equity_paths": equity_paths[:50],
            "drawdown_paths": drawdown_paths[:50],

            # Legacy flat fields for backward compat with existing callers
            "robustness_score": self._calculate_robustness_score(
                num_trades, float(sorted_dd[p95]), prob_ruin
            ),
            "median_final_balance": round(float(sorted_bal[p50]), 2),
            "worst_case_balance_95ci": round(float(sorted_bal[p05]), 2),
            "worst_case_dd_95ci": f"{sorted_dd[p95]:.2f}%",
            "probability_of_ruin": f"{prob_ruin:.2f}%",
        }

    def _calculate_robustness_score(self, num_trades: int, worst_dd: float, ruin_prob: float) -> float:
        score = 100.0

        if worst_dd > 15:
            score -= (worst_dd - 15) ** 1.5 * 2.5

        if ruin_prob > 0:
            score -= (ruin_prob * 10) + 50

        if num_trades < 100:
            if num_trades >= 50:
                score -= (100 - num_trades) * 0.1
            else:
                score -= (50 - num_trades) * 0.2 + 5.0

        return max(0, min(100, round(score, 1)))
