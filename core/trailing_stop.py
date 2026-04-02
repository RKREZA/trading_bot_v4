import logging
import threading
from typing import Dict, Any, Optional

logger = logging.getLogger("trading_bot.trailing_stop")

class TrailingStopManager:
    """
    Manages adaptive trailing stops for active positions.
    Implements a 3-phase risk reduction strategy:
    1. Chandelier: Initial buffer based on ATR.
    2. Break-Even: Moved to entry + buffer after 1.5R profit.
    3. Structural: Locked in using candle extremes after 3.0R profit.
    """
    
    def __init__(self, config: dict, connection: Any, position_meta: Dict[int, Any], state_lock: threading.Lock):
        """
        Initializes the TrailingStopManager.
        
        Args:
            config (dict): Global configuration.
            connection (MT5Connection): terminal connection.
            position_meta (dict): Shared position metadata.
            state_lock (Lock): Thread-safety lock for metadata access.
        """
        self.config = config
        self.connection = connection
        self.position_meta = position_meta
        self.state_lock = state_lock
        
    def manage_positions(self, symbol: str, current_bid: float, current_ask: float, atr: float, last_candle: Optional[dict] = None):
        """
        Performs trailing stop logic for all active positions on a symbol.
        
        Args:
            symbol (str): Symbol name.
            current_bid (float): Current bid price.
            current_ask (float): Current ask price.
            atr (float): Current ATR for buffer calculation.
            last_candle (Optional[dict]): Most recent closed candle for structural stops.
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
                        "ticket": ticket, "best_price": pos.price_current,
                        "risk": abs(pos.price_open - pos.sl) if pos.sl > 0 else 0
                    }
                meta = self.position_meta[ticket]
                
            is_buy = (pos.type == 0)
            current_price = current_bid if is_buy else current_ask
            
            # 1. High-Water Mark Update
            if (is_buy and current_price > meta["best_price"]) or (not is_buy and current_price < meta["best_price"]):
                meta["best_price"] = current_price
                
            # 2. Optimized Trailing Logic (Shared with Backtest)
            new_sl = self.calculate_new_sl(
                is_buy, pos.price_open, pos.sl, meta["best_price"], atr, meta["risk"], last_candle
            )
            
            if new_sl and new_sl != pos.sl:
                if (is_buy and new_sl > pos.sl) or (not is_buy and (pos.sl == 0 or new_sl < pos.sl)):
                    if self.connection.modify_sl_tp(ticket, symbol, new_sl, pos.tp):
                        logger.info(f"[Trailing] Trade {ticket} -> {new_sl:.2f}")

    @staticmethod
    def calculate_new_sl(is_buy: bool, entry: float, current_sl: float, best_price: float, 
                         atr: float, risk: float, last_candle: Optional[dict] = None) -> Optional[float]:
        """
        Calculates the new Stop Loss level based on the adaptive 3-phase logic.
        
        Args:
            is_buy (bool): Trade direction.
            entry (float): Entry price.
            current_sl (float): Current Stop Loss.
            best_price (float): Highest price reached (for BUY) or lowest (for SELL).
            atr (float): ATR value.
            risk (float): Initial risk amount (points).
            last_candle (Optional[dict]): Structured candle for Phase 3.
            
        Returns:
            Optional[float]: The new SL level if it should be moved, else None.
        """
        if risk <= 0: return None
        
        profit_points = (best_price - entry) if is_buy else (entry - best_price)
        rr = profit_points / risk
        
        new_sl = current_sl

        # Phase 1: Institutional Chandelier (Room to Breathe)
        # Wider buffer (4.0x ATR) early on, tighter (2.5x ATR) after 1.5R profit
        c_mult = 4.0 if rr < 1.5 else 2.5
        chandelier_sl = (best_price - (atr * c_mult)) if is_buy else (best_price + (atr * c_mult))
        
        if (is_buy and chandelier_sl > new_sl) or (not is_buy and (new_sl == 0 or chandelier_sl < new_sl)):
            new_sl = chandelier_sl

        # Phase 2: Delayed Break-Even (Active at 1.5R)
        # Gold often retraces to entry before surging; BE at 1.0R is too early.
        if rr >= 1.5:
            be_sl = entry + (risk * 0.1) if is_buy else entry - (risk * 0.1)
            if (is_buy and be_sl > new_sl) or (not is_buy and (new_sl == 0 or be_sl < new_sl)):
                new_sl = be_sl

        # Phase 3: Structural Lock-in (Active at 3.0R+)
        if rr >= 3.0 and last_candle:
            structural_sl = last_candle['low'] if is_buy else last_candle['high']
            if (is_buy and structural_sl > new_sl) or (not is_buy and (new_sl == 0 or structural_sl < new_sl)):
                new_sl = structural_sl
                
        return new_sl if new_sl != current_sl else None
