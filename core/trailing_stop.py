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
        Institutional-grade trailing stop management with adaptive session logic.
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
                    # [PHASE 7] Recover missing metadata if possible (default to GLOBAL)
                    self.position_meta[ticket] = {
                        "ticket": ticket,
                        "session": "LONDON", # Default fallback
                        "best_price": pos.price_current,
                        "partial_closed_count": 0,
                        "risk": abs(pos.price_open - pos.sl) if pos.sl > 0 else 0,
                        "entry_time": getattr(pos, 'time_setup', 0)
                    }
                    
                meta = self.position_meta[ticket]
                session = meta.get("session", "LONDON")
                session_conf = self.config.get("session_config", {}).get(session, {})
                
                # 1. Update High-Water Mark (MFE)
                is_buy = (pos.type == 0)
                current_price = current_bid if is_buy else current_ask
                
                if is_buy and current_price > meta["best_price"]:
                    meta["best_price"] = current_price
                elif not is_buy and current_price < meta["best_price"]:
                    meta["best_price"] = current_price
                    
                # 2. Risk Metrics
                risk = meta.get("risk", 0)
                if risk <= 0: risk = abs(pos.price_open - pos.sl) if pos.sl > 0 else 0
                
                profit_points = (current_price - pos.price_open) if is_buy else (pos.price_open - current_price)
                
                # 3. Partial Profit Check (Scale-out)
                # Parameters could be session-aware in the future
                if risk > 0 and profit_points > (risk * 1.0) and meta["partial_closed_count"] == 0:
                    lot_to_close = pos.volume * 0.5
                    lot_to_close = max(0.01, round(lot_to_close, 2))
                    if pos.volume > 0.01:
                        success = self.connection.close_position_partial(ticket, lot_to_close)
                        if success:
                            meta["partial_closed_count"] = 1
                            logger.info(f"Position {ticket} Partial Close (1.0R): {lot_to_close} lots closed.")
                    
                # 4. Break-Even Check (Institutional: 1.0 RR + Spread)
                # We move to BE once 1.0R is achieved to protect capital
                be_activation = 1.0 # [INSTITUTIONAL] 1.0R
                if risk > 0 and profit_points > (risk * be_activation):
                    # Fetch current spread to add as buffer
                    with self.connection.MT5_LOCK:
                        tick = self.connection.mt5.symbol_info_tick(symbol) if hasattr(self.connection, 'mt5') else None
                    spread = (tick.ask - tick.bid) if tick else (atr * 0.2) # Fallback to 0.2 ATR if tick fails
                    
                    # BE + Spread to ensure "Break-even is not a loss"
                    new_sl = pos.price_open + spread if is_buy else pos.price_open - spread
                    
                    if (is_buy and new_sl > pos.sl) or (not is_buy and (pos.sl == 0.0 or new_sl < pos.sl)):
                        self.connection.modify_sl_tp(ticket, symbol, new_sl, pos.tp)
                        logger.info(f"Position {ticket} moved to BE+Spread (1.0R achieved): {new_sl}")

                # 5. Chandelier Exit (ATR-based Trailing)
                # Multiplier varies by session to handle noise
                multiplier = session_conf.get("trailing_atr_mult", 3.5)
                chandelier_dist = atr * multiplier
                
                if is_buy:
                    new_sl = meta["best_price"] - chandelier_dist
                    if new_sl > pos.sl:
                        self.connection.modify_sl_tp(ticket, symbol, new_sl, pos.tp)
                else:
                    new_sl = meta["best_price"] + chandelier_dist
                    if pos.sl == 0.0 or new_sl < pos.sl:
                        self.connection.modify_sl_tp(ticket, symbol, new_sl, pos.tp)
