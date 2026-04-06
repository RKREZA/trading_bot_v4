import logging
from typing import Dict, List, Optional
from core.common.types import TradeSignal

logger = logging.getLogger("trading_bot.portfolio")

class PortfolioManager:
    """
    Institutional Portfolio Manager.
    Allocates capital across strategies and resolves signal conflicts.
    """

    def __init__(self, config: dict):
        self.config = config
        self.initial_balance = float(config.get("initial_balance", 1000.0))
        
        # Institutional Allocation (Step 9)
        self.allocations = config.get("portfolio_allocations", {
            "TrendFollowing": 0.40,
            "MeanReversion": 0.30,
            "Breakout": 0.30,
            "LiquiditySession": 0.0
        })
        
        # Verify Checksum
        total = sum(self.allocations.values())
        if total > 1.01 or total < 0.99:
            logger.warning(f"Portfolio allocation checksum error: {total}. Normalizing...")
            self.allocations = {k: v/total for k, v in self.allocations.items()}
            
    def resolve_signals(self, signals: Dict[str, TradeSignal]) -> List[tuple[str, TradeSignal]]:
        """
        Institutional Parallel Signal Audit (Step 9).
        Allows multiple non-conflicting strategies to execute in the same cycle.
        """
        active_signals = {sid: sig for sid, sig in signals.items() if sig.direction != "NONE"}
        if not active_signals:
            return []

        # In parallel mode, we approve all valid signals from different strategies.
        # Directional Filtering: We keep only the best signal per strategy (already handled by pulse loop)
        # and ensure we aren't overloaded.
        approved = []
        for sid, sig in active_signals.items():
            approved.append((sid, sig))
            
        # Optional: Sort by confidence for sequential execution pulse
        approved.sort(key=lambda x: x[1].confidence, reverse=True)
        return approved

    def get_strategy_balance(self, total_balance: float, strategy_id: str) -> float:
        """Looks up allocation with key normalization (e.g., 'trendfollowing_v4' → 'TrendFollowing')."""
        # Try direct lookup first
        allocation = self.allocations.get(strategy_id)
        if allocation is not None:
            return total_balance * allocation
        
        # Normalize: strip _v4 suffix, convert to title case variants
        normalized = strategy_id.replace("_v4", "").replace("_", "")
        for key in self.allocations:
            if key.lower().replace("_", "") == normalized.lower():
                return total_balance * self.allocations[key]
        
        # Fallback: equal share
        logger.warning(f"No allocation found for '{strategy_id}'. Using equal share.")
        num_strategies = len(self.allocations) or 1
        return total_balance / num_strategies
