import logging
import asyncio
from typing import Dict, Any, List
from datetime import datetime

from core.strategy.engine import BaseStrategy, TradeSignal
from core.time.time_service import time_service
from core.data.mt5_service import mt5_service

logger = logging.getLogger("trading_bot.forward_engine")

class ShadowPosition:
    def __init__(self, signal: TradeSignal, exec_price: float, timestamp: datetime):
        self.symbol = signal.symbol
        self.direction = signal.direction
        self.entry_price = exec_price
        self.entry_time = timestamp
        self.sl = signal.stop_loss
        self.tp = signal.take_profit
        self.volume = 1.0 # simplified
        
        self.exit_price = 0.0
        self.exit_time = None
        self.pnl = 0.0
        self.is_closed = False

    def update_tick(self, bid: float, ask: float, timestamp: datetime):
        if self.is_closed:
            return

        if self.direction == 'BUY':
            # Exit at BID
            if bid <= self.sl:
                self.exit_price = self.sl
                self.exit_time = timestamp
                self.pnl = (self.exit_price - self.entry_price) * self.volume
                self.is_closed = True
            elif bid >= self.tp:
                self.exit_price = self.tp
                self.exit_time = timestamp
                self.pnl = (self.exit_price - self.entry_price) * self.volume
                self.is_closed = True
        elif self.direction == 'SELL':
            # Exit at ASK
            if ask >= self.sl:
                self.exit_price = self.sl
                self.exit_time = timestamp
                self.pnl = (self.entry_price - self.exit_price) * self.volume
                self.is_closed = True
            elif ask <= self.tp:
                self.exit_price = self.tp
                self.exit_time = timestamp
                self.pnl = (self.entry_price - self.exit_price) * self.volume
                self.is_closed = True

class ForwardTestEngine:
    """
    Shadow trading engine. Simulates execution using LIVE tick data, but places NO real trades.
    Relies on MT5 tick feed and TimeService for exact timestamping.
    """
    def __init__(self, strategy: BaseStrategy):
        self.strategy = strategy
        self.positions: List[ShadowPosition] = []
        self.history: List[ShadowPosition] = []
        self.is_running = False

    async def run_tick_loop(self):
        """Continuously pulls ticks and updates shadow positions/strategy."""
        self.is_running = True
        logger.info(f"ForwardTestEngine: Starting shadow trading for {self.strategy.symbol}")
        
        while self.is_running:
            tick = mt5_service.get_tick(self.strategy.symbol)
            if tick is None:
                await asyncio.sleep(1)
                continue
                
            # 'time_utc' was already injected by mt5_service using TimeService
            timestamp_utc = tick['time_utc']
            bid = tick['bid']
            ask = tick['ask']

            # 1. Update existing shadow positions
            for pos in self.positions:
                pos.update_tick(bid, ask, timestamp_utc)
                if pos.is_closed:
                    self.history.append(pos)
                    logger.info(f"Shadow Trade Closed: {pos.direction} {pos.symbol} PnL: {pos.pnl:.2f}")
            
            self.positions = [p for p in self.positions if not p.is_closed]

            # 2. Strategy on_tick (for early exits, trailing stops etc if implemented)
            self.strategy.on_tick(tick, timestamp_utc)

            # In a full implementation, we'd also pull candles and call strategy.on_candle()
            # on minute boundaries. For brevity, omitted here, but the timestamp_utc 
            # would be passed exactly the same way.

            await asyncio.sleep(0.5)

    def execute_signal(self, signal: TradeSignal, current_tick: dict):
        """Called by a coordinator when the strategy generates a signal."""
        bid = current_tick['bid']
        ask = current_tick['ask']
        timestamp = current_tick['time_utc']

        # Buy at Ask, Sell at Bid
        exec_price = ask if signal.direction == 'BUY' else bid
        
        pos = ShadowPosition(signal, exec_price, timestamp)
        self.positions.append(pos)
        logger.info(f"Shadow Trade Opened: {signal.direction} {signal.symbol} @ {exec_price}")

    def stop(self):
        self.is_running = False
        logger.info("ForwardTestEngine: Stopped.")
