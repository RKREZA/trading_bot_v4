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
        
    def manage_positions(self, symbol: str, current_bid: float, current_ask: float, atr: float):
        """
        Institutional-grade trailing stop management with Chandelier Exit (ATR-based).
        Ensures a One-Way Ratchet (stop only moves forward).
        """
        positions = self.connection.get_positions(symbol)
        if not positions:
            return
            
        magic = self.config.magic_number if hasattr(self.config, 'magic_number') else self.config.get("magic_number", 234000)
        
        for pos in positions:
            if pos.magic != magic:
                continue
                
            ticket = pos.ticket
            
            with self.state_lock:
                if ticket not in self.position_meta:
                    self.position_meta[ticket] = {
                        "ticket": ticket,
                        "best_price": pos.price_current,
                        "partial_closed_count": 0,
                        "risk": abs(pos.price_open - pos.sl) if pos.sl > 0 else 0,
                        "entry_time": getattr(pos, 'time_setup', 0)
                    }
                    
                meta = self.position_meta[ticket]
                
                # 1. Update High-Water Mark (MFE - Maximum Favorable Excursion)
                is_buy = (pos.type == 0)
                current_price = current_bid if is_buy else current_ask
                
                if is_buy and current_price > meta["best_price"]:
                    meta["best_price"] = current_price
                elif not is_buy and current_price < meta["best_price"]:
                    meta["best_price"] = current_price
                    
                # 2. Partial Profit Check (50% at 1.0R)
                # This protects capital before the 1.5R Break-Even trigger or the wider Chandelier takes over.
                if risk > 0 and profit_points > risk and meta["partial_closed_count"] == 0:
                    lot_to_close = pos.volume * 0.5
                    lot_to_close = max(0.01, round(lot_to_close, 2))
                    
                    if pos.volume > 0.01:
                        success = self.connection.close_position_partial(ticket, lot_to_close)
                        if success:
                            meta["partial_closed_count"] = 1
                            logger.info(f"Position {ticket} Partial Close (1.0R): {lot_to_close} lots closed.")
                    
                # 3. Break-Even Check (Minimum 1.5R Profit)
                if risk > 0 and profit_points > (risk * 1.5):
                    # Move SL to Entry + small buffer (0.1R) or just Entry
                    new_sl = pos.price_open + (risk * 0.1) if is_buy else pos.price_open - (risk * 0.1)
                    
                    # One-Way Ratchet Check
                    if (is_buy and new_sl > pos.sl) or (not is_buy and (pos.sl == 0.0 or new_sl < pos.sl)):
                        self.connection.modify_sl_tp(ticket, symbol, new_sl, pos.tp)
                        logger.info(f"Position {ticket} moved to Break-Even (1.5R reached) at {new_sl}")

                # 3. Chandelier Exit (ATR-based Trailing)
                # Multiplier 3.5x is used for XAUUSDm to filter out market noise and prevent "Stop Choking"
                multiplier = 3.5 
                chandelier_dist = atr * multiplier
                
                if is_buy:
                    new_sl = meta["best_price"] - chandelier_dist
                    # One-Way Ratchet: Only move the stop up
                    if new_sl > pos.sl:
                        self.connection.modify_sl_tp(ticket, symbol, new_sl, pos.tp)
                        # logger.debug(f"Position {ticket} Trailed (Chandelier) to {new_sl}")
                else:
                    new_sl = meta["best_price"] + chandelier_dist
                    # One-Way Ratchet: Only move the stop down
                    if pos.sl == 0.0 or new_sl < pos.sl:
                        self.connection.modify_sl_tp(ticket, symbol, new_sl, pos.tp)
                        # logger.debug(f"Position {ticket} Trailed (Chandelier) to {new_sl}")
