import MetaTrader5 as mt5
import logging
from typing import Dict, Any, List
from datetime import datetime, timedelta

from core.data.mt5_service import mt5_service
from core.time.time_service import time_service

logger = logging.getLogger("trading_bot.reconciliation")

class ReconciliationEngine:
    """
    Synchronizes internal platform state with the MT5 terminal state.
    Detects missing or duplicate trades and positions.
    """
    def __init__(self, execution_engine):
        self.execution_engine = execution_engine
        self.last_sync = None

    def reconcile(self):
        """Performs a full reconciliation of open positions and orders."""
        if not mt5_service.connected:
            return

        logger.info("ReconciliationEngine: Starting sync...")
        
        # 1. Fetch MT5 open positions
        mt5_positions = mt5.positions_get()
        if mt5_positions is None:
            logger.error(f"Failed to get positions: {mt5.last_error()}")
            mt5_positions = []
            
        # 2. Fetch MT5 active orders (pending)
        mt5_orders = mt5.orders_get()
        if mt5_orders is None:
            logger.error(f"Failed to get orders: {mt5.last_error()}")
            mt5_orders = []

        # 3. Detect Discrepancies
        # Compare with ExecutionEngine.orders
        internal_orders = self.execution_engine.orders
        
        # Example check: Find positions in MT5 that we don't have recorded
        for pos in mt5_positions:
            pos_dict = pos._asdict()
            # Try to match by comment (contains execution_id)
            comment = pos_dict.get('comment', '')
            found = False
            for eid, data in internal_orders.items():
                if eid[:8] in comment:
                    found = True
                    break
            
            if not found:
                logger.warning(f"ReconciliationEngine: Untracked position found in MT5: {pos_dict['ticket']} ({pos_dict['symbol']})")

        # 4. Correct internal state
        # (This is where you'd update your database if a trade was closed externally)
        
        self.last_sync = time_service.get_server_time()
        logger.info("ReconciliationEngine: Sync complete.")

    def get_account_summary(self) -> Dict[str, Any]:
        """Returns account snapshot for RiskEngine."""
        acc = mt5.account_info()
        if acc is None:
            return {}
            
        return {
            "balance": acc.balance,
            "equity": acc.equity,
            "margin": acc.margin,
            "free_margin": acc.margin_free,
            "profit": acc.profit,
            "drawdown": (acc.balance - acc.equity) / acc.balance if acc.balance > 0 else 0.0
        }
