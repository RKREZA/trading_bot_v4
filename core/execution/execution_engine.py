import MetaTrader5 as mt5
import logging
import uuid
from typing import Optional, Dict, Any
from datetime import datetime
from enum import Enum

from core.strategy.engine import TradeSignal
from core.data.mt5_service import mt5_service
from core.time.time_service import time_service

logger = logging.getLogger("trading_bot.execution_engine")

class OrderState(Enum):
    PENDING = "PENDING"
    SENT = "SENT"
    FILLED = "FILLED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"
    CANCELLED = "CANCELLED"

class ExecutionEngine:
    """
    Handles order execution via MT5.
    Implements idempotency and state tracking.
    """
    def __init__(self):
        self.orders: Dict[str, Dict[str, Any]] = {} # Track orders by execution_id

    def execute_signal(self, signal: TradeSignal, volume: float) -> Optional[int]:
        """
        Sends an order to MT5.
        signal: TradeSignal validated by RiskEngine.
        volume: Calculated lot size.
        Returns order ticket if successful.
        """
        # Idempotency check
        if signal.execution_id in self.orders:
            logger.warning(f"ExecutionEngine: Duplicate execution attempt for {signal.execution_id}")
            return self.orders[signal.execution_id].get("ticket")

        # Initialize order state
        self.orders[signal.execution_id] = {
            "state": OrderState.PENDING,
            "signal_id": signal.id,
            "timestamp": time_service.get_server_time(),
            "ticket": None
        }

        # Prepare request
        symbol_info = mt5_service.get_symbol_info(signal.symbol)
        if not symbol_info:
            self._update_state(signal.execution_id, OrderState.REJECTED, "Symbol info not found")
            return None

        # Determine trade type
        order_type = mt5.ORDER_TYPE_BUY if signal.direction == 'BUY' else mt5.ORDER_TYPE_SELL
        price = mt5_service.normalize_price(signal.symbol, signal.entry)
        sl = mt5_service.normalize_price(signal.symbol, signal.stop_loss)
        tp = mt5_service.normalize_price(signal.symbol, signal.take_profit)

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": signal.symbol,
            "volume": volume,
            "type": order_type,
            "price": price,
            "sl": sl,
            "tp": tp,
            "deviation": 20,
            "magic": 123456,
            "comment": f"Antigravity_{signal.execution_id[:8]}",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        # Send order
        start_time = datetime.now()
        self._update_state(signal.execution_id, OrderState.SENT)
        
        result = mt5.order_send(request)
        end_time = datetime.now()
        latency = (end_time - start_time).total_seconds() * 1000 # ms

        if result is None:
            self._update_state(signal.execution_id, OrderState.REJECTED, "MT5 order_send returned None")
            return None

        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error(f"Order failed: retcode={result.retcode}, error={mt5.last_error()}")
            self._update_state(signal.execution_id, OrderState.REJECTED, f"Retcode: {result.retcode}")
            return None

        # Success
        logger.info(f"Order FILLED: {result.order} for {signal.symbol} at {result.price}. Latency: {latency:.2f}ms")
        self._update_state(signal.execution_id, OrderState.FILLED, ticket=result.order, fill_price=result.price, latency=latency)
        
        return result.order

    def _update_state(self, execution_id: str, state: OrderState, reason: str = "", **kwargs):
        if execution_id in self.orders:
            self.orders[execution_id].update({
                "state": state,
                "reason": reason,
                "updated_at": time_service.get_server_time(),
                **kwargs
            })
            # In a real production app, we would persist this to PostgreSQL here
            logger.debug(f"ExecutionEngine: Order {execution_id} state -> {state.value} ({reason})")
