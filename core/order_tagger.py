"""
TRADING BOT V3 — Order Tagging System
Creates and parses strategy-attributed order comments for MT5.
MT5 comment field is limited to ~31 characters.
"""

from typing import Optional, Tuple


class OrderTagger:
    """
    Utility for encoding/decoding strategy attribution in MT5 order comments.
    
    Format: "BV3|{strategy_id}|{trade_id_short}"
    Example: "BV3|sniper_v1|a1b2c3d4"
    
    This ensures:
        - Trade attribution to a specific strategy
        - Independent PnL calculation per strategy
        - No cross-strategy contamination
    """

    PREFIX = "BV3"
    SEPARATOR = "|"

    @staticmethod
    def create_comment(strategy_id: str, trade_id: str) -> str:
        """
        Creates an MT5 order comment with strategy attribution.
        
        Args:
            strategy_id: Unique strategy identifier
            trade_id: Unique trade identifier (will be truncated to 8 chars)
            
        Returns:
            Formatted comment string (max ~31 chars for MT5 compatibility)
        """
        # Truncate strategy_id to keep within MT5's ~31 char limit
        # Format: "BV3|" (4) + strategy_id (max 18) + "|" (1) + trade_id (8) = 31
        sid = strategy_id[:18]
        tid = trade_id[:8]
        return f"{OrderTagger.PREFIX}{OrderTagger.SEPARATOR}{sid}{OrderTagger.SEPARATOR}{tid}"

    @staticmethod
    def parse_comment(comment: str) -> Optional[Tuple[str, str]]:
        """
        Parses an MT5 order comment to extract strategy attribution.
        
        Args:
            comment: The order comment string from MT5
            
        Returns:
            Tuple of (strategy_id, trade_id) if valid, None otherwise
        """
        if not comment:
            return None
        parts = comment.split(OrderTagger.SEPARATOR)
        if len(parts) >= 3 and parts[0] == OrderTagger.PREFIX:
            return parts[1], parts[2]
        return None

    @staticmethod
    def is_tagged(comment: str) -> bool:
        """Check if a comment was created by this tagging system."""
        return comment is not None and comment.startswith(OrderTagger.PREFIX + OrderTagger.SEPARATOR)

    @staticmethod
    def extract_strategy_id(comment: str) -> Optional[str]:
        """Extract just the strategy_id from a tagged comment."""
        result = OrderTagger.parse_comment(comment)
        return result[0] if result else None
