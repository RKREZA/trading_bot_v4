import logging
import MetaTrader5 as mt5
from typing import Dict, Any, List, Optional
from datetime import datetime
import time

from core.strategy.engine import TradeSignal
from core.data.mt5_service import mt5_service
from core.time.time_service import time_service

logger = logging.getLogger("trading_bot.execution_engine")

class ExecutionEngine:
    """
    Handles live order execution via MT5 API.
    Uses precise timestamps to measure execution latency.
    """
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.max_retries = config.get("max_retries", 3)
        self.retry_delay_sec = config.get("retry_delay_sec", 1.0)
        self.deviation = config.get("slippage_deviation_points", 10)

    def execute_signal(self, signal: TradeSignal, volume: float) -> Optional[dict]:
        """
        Translates a TradeSignal into an MT5 order.
        Measures and logs execution latency.
        """
        if not mt5_service.connected:
            logger.error("ExecutionEngine: Cannot execute, MT5 not connected.")
            return None

        symbol = signal.symbol
        action = mt5.ORDER_TYPE_BUY if signal.direction == 'BUY' else mt5.ORDER_TYPE_SELL
        
        # Get the latest price right before execution
        tick = mt5.symbol_info_tick(symbol)
        if not tick:
            logger.error(f"ExecutionEngine: Could not get tick for {symbol}")
            return None

        price = tick.ask if signal.direction == 'BUY' else tick.bid
        
        # Normalize inputs
        volume = mt5_service.normalize_volume(symbol, volume)
        price = mt5_service.normalize_price(symbol, price)
        sl = mt5_service.normalize_price(symbol, signal.stop_loss)
        tp = mt5_service.normalize_price(symbol, signal.take_profit)

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": float(volume),
            "type": action,
            "price": float(price),
            "sl": float(sl),
            "tp": float(tp),
            "deviation": self.deviation,
            "magic": 234000,
            "comment": "Antigravity V5",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC, # Or FOK depending on broker
        }

        # Measure latency
        send_time = time_service.get_current_time()
        
        for attempt in range(self.max_retries):
            result = mt5.order_send(request)
            
            if result.retcode == mt5.TRADE_RETCODE_DONE:
                exec_time = time_service.get_current_time()
                latency_ms = (exec_time - signal.signal_time).total_seconds() * 1000
                logger.info(
                    f"Execution Success: {signal.direction} {symbol} "
                    f"Vol: {volume} @ {result.price} (Latency: {latency_ms:.0f}ms)"
                )
                return result._asdict()
            else:
                logger.warning(
                    f"Execution Failed (Attempt {attempt+1}/{self.max_retries}): "
                    f"Code: {result.retcode}, Comment: {result.comment}"
                )
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay_sec)
                else:
                    logger.error(f"Execution permanently failed for signal: {signal}")
                    
        return None

    def get_open_positions(self) -> List[dict]:
        """Returns currently open positions."""
        if not mt5_service.connected:
            return []
            
        positions = mt5.positions_get()
        if positions is None:
            return []
            
        return [p._asdict() for p in positions]

    def close_position(self, ticket: int) -> bool:
        """Closes an open position by ticket."""
        if not mt5_service.connected:
            return False
            
        pos = mt5.positions_get(ticket=ticket)
        if not pos or len(pos) == 0:
            logger.error(f"ExecutionEngine: Position {ticket} not found.")
            return False
            
        pos = pos[0]
        symbol = pos.symbol
        action = mt5.ORDER_TYPE_SELL if pos.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY
        
        tick = mt5.symbol_info_tick(symbol)
        price = tick.bid if action == mt5.ORDER_TYPE_SELL else tick.ask
        
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "position": ticket,
            "symbol": symbol,
            "volume": pos.volume,
            "type": action,
            "price": float(price),
            "deviation": self.deviation,
            "magic": 234000,
            "comment": "Close Position",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        
        result = mt5.order_send(request)
        if result.retcode == mt5.TRADE_RETCODE_DONE:
            logger.info(f"Position {ticket} closed successfully at {result.price}")
            return True
        else:
            logger.error(f"Failed to close position {ticket}. Code: {result.retcode}")
            return False
