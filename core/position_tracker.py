"""
TRADING BOT V3 — Per-Strategy Position Tracker
Tracks which MT5 tickets belong to a specific strategy.
Zero shared state with other strategies' trackers.
"""

import logging
import threading
from typing import Any, Dict, List, Optional

logger = logging.getLogger("trading_bot.position_tracker")


class PositionTracker:
    """
    Per-strategy position tracking and metadata management.
    
    Each StrategyRuntime gets its own PositionTracker instance.
    Positions are identified by MT5 ticket number and attributed
    via the order comment tagging system.
    
    Thread-safe: all mutations protected by internal lock.
    """

    def __init__(self, strategy_id: str):
        """
        Args:
            strategy_id: The owning strategy's unique identifier
        """
        self.strategy_id = strategy_id
        self._positions: Dict[int, dict] = {}  # ticket -> metadata
        self._trade_history: List[dict] = []
        self._lock = threading.Lock()

    @property
    def open_count(self) -> int:
        """Number of currently open positions for this strategy."""
        with self._lock:
            return len(self._positions)

    @property
    def has_open_position(self) -> bool:
        return self.open_count > 0

    def add_position(self, ticket: int, metadata: dict) -> None:
        """
        Register a new position owned by this strategy.
        
        Args:
            ticket: MT5 position ticket number
            metadata: Dict with keys: entry_price, direction, session, risk, etc.
        """
        with self._lock:
            self._positions[ticket] = {
                "ticket": ticket,
                "strategy_id": self.strategy_id,
                **metadata
            }
        logger.info("[%s] Position opened: ticket=%d", self.strategy_id, ticket)

    def remove_position(self, ticket: int) -> Optional[dict]:
        """
        Remove a position (on close). Returns the metadata or None.
        """
        with self._lock:
            return self._positions.pop(ticket, None)

    def get_position(self, ticket: int) -> Optional[dict]:
        """Get metadata for a specific ticket."""
        with self._lock:
            return self._positions.get(ticket)

    def get_all_positions(self) -> Dict[int, dict]:
        """Returns a snapshot copy of all open positions."""
        with self._lock:
            return dict(self._positions)

    def get_tickets(self) -> List[int]:
        """Returns list of all open ticket IDs."""
        with self._lock:
            return list(self._positions.keys())

    def update_position(self, ticket: int, updates: dict) -> bool:
        """
        Update metadata for an open position.
        
        Args:
            ticket: Position ticket
            updates: Dict of fields to update (e.g. best_price, partial_closed_count)
            
        Returns:
            True if position exists and was updated.
        """
        with self._lock:
            if ticket in self._positions:
                self._positions[ticket].update(updates)
                return True
            return False

    def record_trade(self, trade_record: dict) -> None:
        """
        Add a completed trade to this strategy's history.
        
        Args:
            trade_record: Dict with keys: ticket, pnl, result, entry, exit, etc.
        """
        with self._lock:
            trade_record["strategy_id"] = self.strategy_id
            self._trade_history.append(trade_record)
            # Keep window to 500 trades
            if len(self._trade_history) > 500:
                self._trade_history = self._trade_history[-500:]

    def get_trade_history(self) -> List[dict]:
        """Returns a copy of trade history."""
        with self._lock:
            return list(self._trade_history)

    def get_state(self) -> dict:
        """Serialize state for persistence."""
        with self._lock:
            return {
                "strategy_id": self.strategy_id,
                "positions": dict(self._positions),
                "trade_count": len(self._trade_history),
            }

    def load_state(self, state: dict) -> None:
        """Restore state from persistence."""
        with self._lock:
            self._positions = {
                int(k): v for k, v in state.get("positions", {}).items()
            }

    def reconcile(self, live_tickets: set) -> List[int]:
        """
        Reconcile tracker state against live MT5 tickets.
        Removes any tracked positions that no longer exist in MT5.
        
        Args:
            live_tickets: Set of ticket IDs currently alive in MT5
            
        Returns:
            List of tickets that were removed (closed externally)
        """
        closed = []
        with self._lock:
            stale = [t for t in self._positions if t not in live_tickets]
            for t in stale:
                del self._positions[t]
                closed.append(t)
        if closed:
            logger.info("[%s] Reconciled: removed %d stale positions", self.strategy_id, len(closed))
        return closed
