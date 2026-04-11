import numpy as np
import logging
from typing import Dict, Any
from core.common.types import CandleArray

logger = logging.getLogger("trading_bot.fidelity")

class FidelityEngine:
    """
    V5-INSIGNIA Data Fidelity Score (DFS) Engine.
    Institutional Grade: Grades market data on 5 dimensions of integrity.
    DFS ∈ [0.0, 1.0]
    """

    @staticmethod
    def calculate_dfs(candles: CandleArray, timeframe: str, prev_dfs: float = 1.0) -> float:
        """
        Calculates the composite DFS with Rule 2 (Stability & Smoothing).
        """
        if len(candles) < 20:
            return prev_dfs
        
        # 1. Density Score
        avg_ticks = np.mean(candles.v)
        baseline = {"M1": 50, "M5": 200, "M15": 500, "H1": 2000}.get(timeframe, 200)
        density_score = min(1.0, avg_ticks / baseline)

        # 2. Stability Score
        spread_std = np.std(candles.s)
        spread_mean = np.max([np.mean(candles.s), 1.0])
        stability_score = max(0.0, 1.0 - (spread_std / spread_mean))

        # 3. Gap Score
        tf_seconds = {"M1": 60, "M5": 300, "M15": 900, "H1": 3600, "D1": 86400}.get(timeframe, 300)
        time_diffs = np.diff(candles.t)
        gaps = np.sum(time_diffs > (tf_seconds * 1.1))
        gap_score = max(0.0, 1.0 - (gaps / len(candles)))

        # 4. Spread Noise Score
        spread_diffs = np.diff(candles.s)
        oscillations = np.sum(np.diff(np.sign(spread_diffs[spread_diffs != 0])) != 0)
        noise_score = max(0.0, 1.0 - (oscillations / len(candles)))

        # 5. Regularity Score
        jitter = np.std(time_diffs % tf_seconds)
        regularity_score = max(0.0, 1.0 - (jitter / 10.0))

        # Rule 2.1: Weight Normalization (Σw = 1.0)
        w = {
            "density": 0.2,
            "stability": 0.15,
            "gap": 0.35,
            "noise": 0.15,
            "regularity": 0.15
        }
        
        dfs_raw = (
            w["density"] * density_score +
            w["stability"] * stability_score +
            w["gap"] * gap_score +
            w["noise"] * noise_score +
            w["regularity"] * regularity_score
        )

        # Rule 2.2: EMA Smoothing (α=0.7) to prevent regime flicker
        alpha = 0.7
        dfs_smooth = alpha * dfs_raw + (1.0 - alpha) * prev_dfs
        
        # Rule 2.2: Hard Stability Constraint (|Δ| < 0.2)
        final_dfs = np.clip(dfs_smooth, prev_dfs - 0.2, prev_dfs + 0.2)

        return float(np.clip(final_dfs, 0.0, 1.0))

    @staticmethod
    def get_classification(dfs: float) -> str:
        """Categorizes the run based on the DFS."""
        if dfs >= 0.75: return "PRODUCTION_VALID"
        if dfs >= 0.50: return "DEGRADED_FIDELITY"
        return "INVALID"
