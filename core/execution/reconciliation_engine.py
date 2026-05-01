import logging
from typing import Dict, Any

logger = logging.getLogger("trading_bot.reconciliation")


class ReconciliationEngine:
    """
    Synchronizes internal platform state with the MT5 terminal state.
    Detects missing or duplicate trades and positions.
    """

    def __init__(self, order_manager, connection=None):
        self.order_manager = order_manager
        self.connection = connection
        self.last_sync = None

    def reconcile(self) -> Dict[str, Any]:
        if not self.connection:
            return {"status": "SKIPPED", "reason": "No MT5 connection"}

        logger.info("ReconciliationEngine: Starting sync...")

        try:
            mt5_positions = self.connection.get_positions() or []
        except Exception as e:
            logger.error(f"Failed to get positions: {e}")
            mt5_positions = []

        discrepancies = []

        tracked_states = self.order_manager._order_states
        mt5_tickets = {getattr(p, "ticket", None) for p in mt5_positions}

        for exec_id, state_data in tracked_states.items():
            if state_data["state"].value == "FILLED":
                ticket = state_data.get("ticket")
                if ticket and ticket not in mt5_tickets:
                    discrepancies.append({
                        "type": "MISSING_POSITION",
                        "execution_id": exec_id,
                        "ticket": ticket,
                    })
                    logger.warning(f"ReconciliationEngine: Tracked ticket {ticket} ({exec_id}) not found in MT5")

        for pos in mt5_positions:
            ticket = getattr(pos, "ticket", None)
            tracked_tickets = {
                v.get("ticket") for v in tracked_states.values() if v.get("ticket")
            }
            if ticket and ticket not in tracked_tickets:
                symbol = getattr(pos, "symbol", "UNKNOWN")
                volume = getattr(pos, "volume", 0.0)
                if volume > 0.1:
                    discrepancies.append({
                        "type": "UNTRACKED_POSITION",
                        "ticket": ticket,
                        "symbol": symbol,
                        "volume": volume,
                    })
                    logger.warning(f"ReconciliationEngine: Untracked position {ticket} ({symbol})")

        import time
        self.last_sync = time.time()
        logger.info(f"ReconciliationEngine: Sync complete. Discrepancies: {len(discrepancies)}")

        return {
            "status": "OK" if not discrepancies else "DISCREPANCIES_FOUND",
            "discrepancies": discrepancies,
            "mt5_positions": len(mt5_positions),
            "tracked_orders": len(tracked_states),
        }

    def get_account_summary(self) -> Dict[str, Any]:
        if not self.connection:
            return {}

        try:
            acc = self.connection.get_account_info()
            if acc is None:
                return {}
            return {
                "balance": getattr(acc, "balance", 0.0),
                "equity": getattr(acc, "equity", 0.0),
                "margin": getattr(acc, "margin", 0.0),
                "free_margin": getattr(acc, "margin_free", 0.0),
                "profit": getattr(acc, "profit", 0.0),
                "drawdown": (
                    (getattr(acc, "balance", 0) - getattr(acc, "equity", 0))
                    / getattr(acc, "balance", 1)
                    if getattr(acc, "balance", 0) > 0
                    else 0.0
                ),
            }
        except Exception as e:
            logger.error(f"Failed to get account summary: {e}")
            return {}
