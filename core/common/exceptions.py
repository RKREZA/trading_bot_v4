"""
V5-INSIGNIA Institutional Exception Hierarchy.
===============================================
Custom exceptions for graceful error handling in safety-critical trading paths.
These replace raw sys.exit() calls to allow forensic logging and connection cleanup.
"""

from datetime import datetime, timezone


class CriticalRiskViolationError(Exception):
    """
    INSTITUTIONAL HARD BLOCK: Raised when a safety-critical risk limit is breached.
    
    This replaces sys.exit() in the execution pipeline to allow the orchestrator
    to perform graceful shutdown: closing open connections, flushing audit logs,
    and executing emergency flatten procedures before process termination.
    
    Attributes:
        lot_size: The violating lot size that triggered the breach.
        max_allowed: The maximum allowed lot size at the time of violation.
        symbol: The instrument symbol involved.
        strategy_id: The strategy that generated the violating intent.
        timestamp: UTC timestamp of the violation event.
        detail: Human-readable forensic description.
    """

    def __init__(
        self,
        lot_size: float,
        max_allowed: float,
        symbol: str = "UNKNOWN",
        strategy_id: str = "UNKNOWN",
        detail: str = "",
    ):
        self.lot_size = lot_size
        self.max_allowed = max_allowed
        self.symbol = symbol
        self.strategy_id = strategy_id
        self.timestamp = datetime.now(timezone.utc)
        self.detail = detail or (
            f"PHASE 1 SAFETY VIOLATION: Intent {lot_size:.4f} lots > "
            f"{max_allowed:.4f} max on {symbol} [{strategy_id}]"
        )
        super().__init__(self.detail)

    def forensic_dict(self) -> dict:
        """Returns a JSON-serializable forensic trail for crash logging."""
        return {
            "error_type": "CriticalRiskViolationError",
            "lot_size": self.lot_size,
            "max_allowed": self.max_allowed,
            "symbol": self.symbol,
            "strategy_id": self.strategy_id,
            "timestamp": self.timestamp.isoformat(),
            "detail": self.detail,
        }
