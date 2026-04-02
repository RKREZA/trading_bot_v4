import logging
import threading
from typing import Dict, Any, Optional

logger = logging.getLogger("trading_bot.trailing_stop")

class TrailingStopManager:
    """Manages aggressive runaway trailing stops and risk protection."""
    
    def __init__(self, config: dict, connection: Any, position_meta: Dict[int, Any], state_lock: threading.Lock):
        self.config = config
        self.connection = connection
        self.position_meta = position_meta
        self.state_lock = state_lock
        
    def manage_positions(self, symbol: str, current_bid: float, current_ask: float, atr: float, last_candle: Optional[dict] = None):
        """
        Implements Sniper-grade Runaway Trailing (up to 1:15 RR).
        - 1.0R: Break-Even
        - 2.0R+: aggressive M1 Candle Trailing
        """
        if not self.config.get("trailing_stop_enabled", True):
            return

        positions = self.connection.get_positions(symbol)
        if not positions: return
            
        magic = self.config.get("magic_number", 234000)
        
        for pos in positions:
            if pos.magic != magic: continue
                
            ticket = pos.ticket
            with self.state_lock:
                if ticket not in self.position_meta:
                    self.position_meta[ticket] = {
                        "ticket": ticket, "session": "GLOBAL", "best_price": pos.price_current,
                        "partial_closed_count": 0, "risk": abs(pos.price_open - pos.sl) if pos.sl > 0 else 0
                    }
                    
                meta = self.position_meta[ticket]
                is_buy = (pos.type == 0)
                current_price = current_bid if is_buy else current_ask
                
                # 1. Update High-Water Mark 
                if is_buy and current_price > meta["best_price"]: meta["best_price"] = current_price
                elif not is_buy and current_price < meta["best_price"]: meta["best_price"] = current_price
                    
                risk = meta.get("risk", 0)
                if risk <= 0: risk = abs(pos.price_open - pos.sl) if pos.sl > 0 else 0
                if risk <= 0: continue
                
                profit_points = (current_price - pos.price_open) if is_buy else (pos.price_open - current_price)
                rr = profit_points / risk
                
                # 2. RUNAWAY TRAILING LOGIC
                new_sl = None
                
                # Phase A: Break-Even (Move to BE+0.1R at 1.0R profit)
                if rr >= 1.0 and pos.sl != (pos.price_open + (risk * 0.1) if is_buy else pos.price_open - (risk * 0.1)):
                    new_sl = pos.price_open + (risk * 0.1) if is_buy else pos.price_open - (risk * 0.1)
                    if (is_buy and new_sl > pos.sl) or (not is_buy and (pos.sl == 0.0 or new_sl < pos.sl)):
                        self.connection.modify_sl_tp(ticket, symbol, new_sl, pos.tp)
                        logger.info(f"[Runaway] Trade {ticket} moved to BE (1.0R hit)")

                # Phase B: Aggressive M1 Candle Trailing (at 2.0R+ profit)
                if rr >= 2.0 and last_candle:
                    # Move SL to the low/high of the previous M1 candle
                    suggested_sl = last_candle['low'] if is_buy else last_candle['high']
                    
                    # Ensure we only move SL in profit direction (tightening)
                    if is_buy and suggested_sl > pos.sl:
                        new_sl = suggested_sl
                    elif not is_buy and (pos.sl == 0.0 or suggested_sl < pos.sl):
                        new_sl = suggested_sl
                        
                    if new_sl and new_sl != pos.sl:
                        self.connection.modify_sl_tp(ticket, symbol, new_sl, pos.tp)
                        logger.info(f"[Runaway] Trade {ticket} Trailing M1 Candle: {new_sl}")

                # 3. Fallback Chandelier Exit (If M1 Trail not active or less aggressive)
                if rr < 2.0:
                    multiplier = 3.5
                    chandelier_dist = atr * multiplier
                    c_sl = (meta["best_price"] - chandelier_dist) if is_buy else (meta["best_price"] + chandelier_dist)
                    
                    if (is_buy and c_sl > pos.sl) or (not is_buy and (pos.sl == 0.0 or c_sl < pos.sl)):
                        self.connection.modify_sl_tp(ticket, symbol, c_sl, pos.tp)
