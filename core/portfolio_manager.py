import logging
from typing import Dict, List, Optional
from core.types import TradeSignal

logger = logging.getLogger("trading_bot.portfolio")

class PortfolioManager:
    """
    Institutional Portfolio Manager.
    Allocates capital across strategies and resolves signal conflicts.
    """

    def __init__(self, config: dict):
        self.config = config
        self.initial_balance = float(config.get("initial_balance", 1000.0))
        
        self.allocations = {
            "trend_following": 0.40,
            "mean_reversion": 0.30,
            "breakout": 0.30,
            "liquidity_session": 0.0
        }
        
    def resolve_signals(self, signals: Dict[str, TradeSignal]) -> Optional[tuple[str, TradeSignal]]:
        active_signals = {sid: sig for sid, sig in signals.items() if sig.direction != "NONE"}
        if not active_signals:
            return None

        sorted_signals = sorted(active_signals.items(), key=lambda x: x[1].confidence, reverse=True)
        
        buys = [s for s in sorted_signals if s[1].direction == "BUY"]
        sells = [s for s in sorted_signals if s[1].direction == "SELL"]
        
        if buys and sells:
            if buys[0][1].confidence > sells[0][1].confidence:
                return buys[0]
            else:
                return sells[0]
        
        return sorted_signals[0]

    def get_strategy_balance(self, total_balance: float, strategy_id: str) -> float:
        allocation = self.allocations.get(strategy_id, 0.25)
        return total_balance * allocation
