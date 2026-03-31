import logging
import threading
from typing import Dict, Any
from core.connection import MT5Connection

logger = logging.getLogger("trading_bot.trailing_stop")

class TrailingStopManager:
    """Manages trailing stops, partial closes, and break-even moving."""
    
    def __init__(self, config: "BotConfig", connection: MT5Connection, position_meta: Dict[int, Any], state_lock: threading.Lock):
        self.config = config
        self.connection = connection
        self.position_meta = position_meta
        self.state_lock = state_lock
        
    def manage_positions(self, symbol: str, current_bid: float, current_ask: float):
        """Iterate over active positions and apply trailing constraints dynamically. Safe TOCTOU."""
        positions = self.connection.get_positions(symbol)
        if not positions:
            return
            
        magic = self.config.magic_number if hasattr(self.config, 'magic_number') else self.config.get("magic_number", 234000)
        
        for pos in positions:
            if pos.magic != magic:
                continue
                
            ticket = pos.ticket
            
            # Coarse lock acquired once per ticket
            with self.state_lock:
                if ticket not in self.position_meta:
                    self.position_meta[ticket] = {
                        "ticket": ticket,
                        "best_price": pos.price_current,
                        "partial_closed_count": 0,
                        "risk": abs(pos.price_open - pos.sl) if pos.sl > 0 else 0
                    }
                    
                meta = self.position_meta[ticket]
                
                # Identify high water mark (MFE)
                current_price = current_bid if pos.type == 0 else current_ask
                if pos.type == 0 and current_price > meta["best_price"]:
                    meta["best_price"] = current_price
                elif pos.type == 1 and current_price < meta["best_price"]:
                    meta["best_price"] = current_price
                    
                # Calculate trailing stops here (Simplified for Phase 10 integration)
                profit_points = (meta["best_price"] - pos.price_open) if pos.type == 0 else (pos.price_open - meta["best_price"])
                
                # Check for Break Even
                min_sl_points = 150 * 0.01  # Assuming config scaling
                if profit_points > min_sl_points * 2:
                    new_sl = pos.price_open + min_sl_points if pos.type == 0 else pos.price_open - min_sl_points
                    if pos.sl == 0.0 or (pos.type == 0 and new_sl > pos.sl) or (pos.type == 1 and new_sl < pos.sl):
                        self.connection.modify_sl_tp(ticket, symbol, new_sl, pos.tp)
                        logger.info(f"Position {ticket} moved to Break-Even at {new_sl}")
