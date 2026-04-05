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
            
    def resolve_signals(self, signals: Dict[str, TradeSignal]) -> Optional[tuple[str, TradeSignal]]:
        """
        Institutional Correlation Control (Step 9).
        1. Groups signals by direction.
        2. Keeps only the highest confidence trade in the same direction.
        3. Returns the absolute winner.
        """
        active_signals = {sid: sig for sid, sig in signals.items() if sig.direction != "NONE"}
        if not active_signals:
            return None

        # Sort all signals by confidence
        sorted_signals = sorted(active_signals.items(), key=lambda x: x[1].confidence, reverse=True)
        
        # Directional Filtering (Same Direction Correlation Control)
        # We only look at the 'top' signal for each direction
        directions = {"BUY": None, "SELL": None}
        for sid, sig in sorted_signals:
            if directions[sig.direction] is None:
                directions[sig.direction] = (sid, sig)

        # Final winner resolve (BUY vs SELL)
        buy_win = directions["BUY"]
        sell_win = directions["SELL"]

        if buy_win and sell_win:
            return buy_win if buy_win[1].confidence >= sell_win[1].confidence else sell_win
        
        return buy_win or sell_win

    def get_strategy_balance(self, total_balance: float, strategy_id: str) -> float:
        allocation = self.allocations.get(strategy_id, 0.25)
        return total_balance * allocation
