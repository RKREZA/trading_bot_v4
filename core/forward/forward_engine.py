import logging
import pandas as pd
from datetime import datetime
from typing import Dict, Any, List, Optional

from core.strategy.engine import BaseStrategy, TradeSignal
from core.data.mt5_service import mt5_service
from core.time.time_service import time_service

logger = logging.getLogger("trading_bot.forward_engine")

class ForwardTestEngine:
    """
    Simulates live execution on real-time data without sending real orders to MT5.
    (Paper Trading)
    """
    def __init__(self, strategy: BaseStrategy):
        self.strategy = strategy
        self.positions = []
        self.history = []
        self.equity = 10000.0
        
    def on_tick(self, symbol: str):
        """Processes a live tick."""
        tick = mt5_service.get_tick(symbol)
        if not tick:
            return
            
        ts = time_service.get_server_time()
        bid, ask = tick['bid'], tick['ask']
        
        # 1. Update existing positions
        for pos in self.positions:
            self._update_position(pos, bid, ask, ts)
            
        self.positions = [p for p in self.positions if not p.is_closed]
        
        # 2. Strategy Logic
        # (Usually on_candle is preferred for SMC, but ticks can trigger entries)
        pass

    def on_candle(self, df: pd.DataFrame):
        """Processes a new closed candle."""
        ts = time_service.get_server_time()
        signal = self.strategy.on_candle(df, ts)
        
        if signal:
            bid, ask = mt5_service.get_bid_ask(self.strategy.symbol)
            # Simulate entry at ASK for BUY, BID for SELL
            entry_price = ask if signal.direction == 'BUY' else bid
            
            logger.info(f"ForwardTest: Signal RECEIVED {signal.direction} at {entry_price}")
            
            pos = {
                "signal": signal,
                "entry_price": entry_price,
                "entry_time": ts,
                "is_closed": False,
                "pnl": 0.0
            }
            self.positions.append(pos)

    def _update_position(self, pos, bid, ask, ts):
        signal = pos['signal']
        if signal.direction == 'BUY':
            # Check SL/TP using BID
            if bid <= signal.stop_loss:
                self._close(pos, signal.stop_loss, ts)
            elif bid >= signal.take_profit:
                self._close(pos, signal.take_profit, ts)
        else:
            # SELL: Check SL/TP using ASK
            if ask >= signal.stop_loss:
                self._close(pos, signal.stop_loss, ts)
            elif ask <= signal.take_profit:
                self._close(pos, signal.take_profit, ts)

    def _close(self, pos, price, ts):
        pos['exit_price'] = price
        pos['exit_time'] = ts
        pos['is_closed'] = True
        
        # Simple PnL calculation
        mult = 1.0 # simplified
        if pos['signal'].direction == 'BUY':
            pos['pnl'] = (price - pos['entry_price']) * mult
        else:
            pos['pnl'] = (pos['entry_price'] - price) * mult
            
        self.history.append(pos)
        logger.info(f"ForwardTest: Position CLOSED. PnL: {pos['pnl']:.2f}")
