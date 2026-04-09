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
        Now enforces allocation filters and confidence prioritizing.
        """
        # 1. Filter by Direction
        active_signals = {sid: sig for sid, sig in signals.items() if sig.direction != "NONE"}
        if not active_signals:
            return []

        # 2. Institutional Allocation Audit (Audit Bug #4 Fix)
        # Filter out strategies with 0% allocation
        eligible_signals = {}
        for sid, sig in active_signals.items():
            # Calculate mock balance to check if strategy is enabled/allocated
            allocation = self.get_strategy_allocation(sid)
            if allocation > 0:
                eligible_signals[sid] = sig
            else:
                logger.info("Signal from %s REJECTED: 0.0 core allocation.", sid)

        if not eligible_signals:
            return []

        # 3. Institutional Conflict Resolution (Rule: No Hegded positions on same symbol)
        buy_signals = {sid: sig for sid, sig in eligible_signals.items() if sig.direction == "BUY"}
        sell_signals = {sid: sig for sid, sig in eligible_signals.items() if sig.direction == "SELL"}

        # If conflicting signals exist on the same symbol, cancel both to prevent hedging traps
        if buy_signals and sell_signals:
            logger.warning("Signal Conflict detected! Canceling opposing trades on same symbol to prevent hedging.")
            return []

        # 4. Final Approval & Ranking
        approved = []
        for sid, sig in eligible_signals.items():
            approved.append((sid, sig))
            
        # Optional: Sort by confidence for sequential execution pulse
        approved.sort(key=lambda x: x[1].confidence, reverse=True)
        return approved

    def get_strategy_allocation(self, strategy_id: str) -> float:
        """Looks up raw allocation weight (0.0 to 1.0)."""
        allocation = self.allocations.get(strategy_id)
        if allocation is not None:
            return allocation
        
        normalized = strategy_id.replace("_v4", "").replace("_", "").lower()
        for key in self.allocations:
            if key.lower().replace("_", "") == normalized:
                return self.allocations[key]
        return 0.0 # Default to 0 if not found in config

    def get_strategy_balance(self, total_balance: float, strategy_id: str) -> float:
        """Calculates allocated balance for a strategy."""
        allocation = self.get_strategy_allocation(strategy_id)
        if allocation > 0:
            return total_balance * allocation
        
        # Fallback: equal share if somehow matched but not in dict (Should not happen with get_strategy_allocation)
        logger.warning(f"No allocation found for '{strategy_id}'. Using minimum share.")
        return total_balance * (1.0 / (len(self.allocations) or 4))
