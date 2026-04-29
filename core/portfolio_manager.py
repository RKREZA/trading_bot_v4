import logging
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from core.common.types import TradeSignal

logger = logging.getLogger("trading_bot.portfolio")

class PortfolioManager:
    """
    V6-LIVE: Institutional Dynamic Portfolio Manager.
    Allocates capital using a Performance-Responsive Framework.
    Features: Equity-Growth Scaling, Drawdown De-scaling, and Adaptive Ranking.
    """

    def __init__(self, config: dict):
        self.config = config
        self.initial_balance = float(config.get("initial_balance", 1000.0))
        
        # 1. Base Institutional Allocation Weights
        self.base_allocations = config.get("portfolio_allocations", {
            "LiquidityPriceAction": 1.0
        })
        
        # 2. Performance Tracking (Institutional State)
        self.strategy_performance = {} # {sid: {"peak": float, "current_dd": float}}
        self.scaling_enabled = config.get("portfolio_scaling", {}).get("enabled", True)
        
        # Rule 3.1: Stability History & Guards
        self.allocation_history = {} # {sid: last_weight}
        self.portfolio_peak = self.initial_balance
        
        self._normalize_base_weights()

    def _normalize_base_weights(self):
        total = sum(self.base_allocations.values())
        if total > 0.001:
            self.base_allocations = {k: v/total for k, v in self.base_allocations.items()}

    def resolve_signals(self, signals: Dict[str, TradeSignal], state: Dict[str, Any] = None) -> List[tuple[str, TradeSignal]]:
        """
        Institutional Dynamic Signal Auction.
        1. Filters by direction and allocation.
        2. Applies Dynamic Scaling (Drawdown/Performance).
        3. Ranks by Performance Efficiency (PnL/Risk).
        """
        active_signals = {sid: sig for sid, sig in signals.items() if sig.direction != "NONE"}
        if not active_signals:
            return []

        # 1. Filter out Zero-Allocation Strategies
        eligible = {}
        for sid, sig in active_signals.items():
            base_weight = self.get_strategy_allocation(sid, dynamic=False)
            if base_weight > 0:
                eligible[sid] = sig
            else:
                logger.info("Signal from %s REJECTED: 0.0 core allocation.", sid)

        # 2. Institutional Conflict Resolution (No Hedging)
        # Prevent opposing trades on the same symbol to avoid commission traps
        buy_sigs = {sid: sig for sid, sig in eligible.items() if sig.direction == "BUY"}
        sell_sigs = {sid: sig for sid, sig in eligible.items() if sig.direction == "SELL"}
        if buy_sigs and sell_sigs:
            logger.warning("Signal Conflict! Opposing trades detected. Canceling all to prevent hedging traps.")
            return []

        # 3. Dynamic Performance Ranking
        # We sort by (Confidence * Dynamic_Weight) to give better allocation to performing strats
        approved = []
        for sid, sig in eligible.items():
            dynamic_weight = self.get_strategy_allocation(sid, dynamic=True)
            # Ranking Score = Confidence (Strategy) * Efficiency (Portfolio)
            score = sig.confidence * (dynamic_weight / self.base_allocations.get(sid, 0.1))
            approved.append((sid, sig, score))
            
        approved.sort(key=lambda x: x[2], reverse=True)
        return [(sid, sig) for sid, sig, score in approved]

    def get_strategy_allocation(self, strategy_id: str, dynamic: bool = True, current_portfolio_equity: float = 0.0) -> float:
        """
        Returns the final allocation weight for a strategy.
        If dynamic=True, applies Drawdown De-scaling and Performance Adjustment.
        Includes EMA Damping, Peak Lock, and Allocation Floors (v6-LOCKED).
        """
        base_weight = self.base_allocations.get(strategy_id, 0.0)
        if base_weight == 0:
            # [ Institutional Resiliency ]: Enhanced Fuzzy Naming Resolution
            # Handles: TrendFollowing vs trend_following vs TrendFollowing_v4
            normalized = strategy_id.replace("_v4", "").replace("_v5", "").replace("_", "").lower()
            for key, weight in self.base_allocations.items():
                k_norm = key.replace("_", "").lower()
                if k_norm == normalized or normalized in k_norm or k_norm in normalized:
                    base_weight = weight
                    logger.info(f"[PORTFOLIO] Resilient Match: '{strategy_id}' resolved to '{key}' ({base_weight:.2%})")
                    break
        
        if not dynamic or not self.scaling_enabled:
            return base_weight

        # --- Rule 3.3: Institutional Drawdown De-scaling ---
        perf = self.strategy_performance.get(strategy_id, {"peak": 0.0, "current_dd": 0.0})
        dd = perf.get("current_dd", 0.0)
        
        # Penalty Function: 
        # If DD < 3%, No penalty.
        # If DD > 10%, Weight -> 0.
        penalty = 1.0
        if dd > 3.0:
            penalty = max(0.0, 1.0 - (dd - 3.0) / 7.0)
            
        # --- Rule 3.4/6.2: Equity-Growth Scaling (PEAK LOCK) ---
        # Only scale up if we are at NEW PORTFOLIO PEAK
        boost = 1.0
        if current_portfolio_equity > self.portfolio_peak:
            self.portfolio_peak = current_portfolio_equity
            if dd < 0.5: boost = 1.1 
        
        # Calculate raw dynamic target
        raw_target = base_weight * penalty * boost
        
        # --- Rule 6.1: Floor Constraint (0.2x Base) ---
        # Prevents strategy starvation during minor drawdown
        floor = base_weight * 0.2
        target = max(floor, raw_target) if base_weight > 0 else 0.0
        
        # --- Rule 3.1: EMA Damping (0.8 Prev / 0.2 New) ---
        # Prevents violent allocation swings
        prev = self.allocation_history.get(strategy_id, target)
        # Cap Delta Change (Rule 3.2: 15% per step)
        delta_max = base_weight * 0.15
        diff = target - prev
        if abs(diff) > delta_max:
            target = prev + (np.sign(diff) * delta_max)
            
        final_weight = (0.8 * prev) + (0.2 * target)
        self.allocation_history[strategy_id] = final_weight
        
        return final_weight

    def update_performance_state(self, strategy_id: str, current_equity: float, total_history: List[float]):
        """Updates internal strategy health state used for next-cycle scaling."""
        if strategy_id not in self.strategy_performance:
            self.strategy_performance[strategy_id] = {"peak": current_equity, "current_dd": 0.0}
            
        perf = self.strategy_performance[strategy_id]
        if current_equity > perf["peak"]:
            perf["peak"] = current_equity
            
        perf["current_dd"] = ((perf["peak"] - current_equity) / perf["peak"] * 100) if perf["peak"] > 0 else 0.0
        
    def get_strategy_balance(self, total_balance: float, strategy_id: str) -> float:
        """Calculates final USD-equivalent balance for sizing calculation."""
        weight = self.get_strategy_allocation(strategy_id, dynamic=True)
        return total_balance * weight
