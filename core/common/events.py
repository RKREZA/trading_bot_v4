import asyncio
import logging
from typing import Any, Callable, Dict, List, Type
from datetime import datetime

from pydantic import BaseModel, Field, ConfigDict

logger = logging.getLogger("trading_bot.events")


class Event(BaseModel):
    model_config = ConfigDict(frozen=True)
    event_type: str = ""
    timestamp: datetime = Field(default_factory=datetime.now)
    correlation_id: str = ""


class SignalGeneratedEvent(Event):
    event_type: str = "SIGNAL_GENERATED"
    symbol: str = ""
    direction: str = ""
    strategy_id: str = ""
    confidence: float = 0.0
    price: float = 0.0


class TradeOpenedEvent(Event):
    event_type: str = "TRADE_OPENED"
    execution_id: str = ""
    symbol: str = ""
    direction: str = ""
    fill_price: float = 0.0
    volume: float = 0.0
    ticket: int = 0
    strategy_id: str = ""


class TradeClosedEvent(Event):
    event_type: str = "TRADE_CLOSED"
    execution_id: str = ""
    symbol: str = ""
    direction: str = ""
    entry_price: float = 0.0
    exit_price: float = 0.0
    pnl: float = 0.0
    ticket: int = 0
    strategy_id: str = ""


class RiskBreachEvent(Event):
    event_type: str = "RISK_BREACH"
    breach_type: str = ""
    severity: str = "WARNING"
    message: str = ""
    current_value: float = 0.0
    threshold: float = 0.0


class SystemStatusEvent(Event):
    event_type: str = "SYSTEM_STATUS"
    mt5_connected: bool = False
    is_trading: bool = False
    active_strategies: int = 0
    open_positions: int = 0
    equity: float = 0.0
    balance: float = 0.0
    drawdown_pct: float = 0.0


class MetricsEvent(Event):
    event_type: str = "METRICS"
    data: Dict[str, Any] = Field(default_factory=dict)


EventHandler = Callable[[Event], Any]


class EventBus:
    def __init__(self):
        self._handlers: Dict[str, List[EventHandler]] = {}
        self._queue: asyncio.Queue = asyncio.Queue()
        self._running = False

    def subscribe(self, event_type: str, handler: EventHandler) -> None:
        if event_type not in self._handlers:
            self._handlers[event_type] = []
        self._handlers[event_type].append(handler)

    def unsubscribe(self, event_type: str, handler: EventHandler) -> None:
        if event_type in self._handlers:
            self._handlers[event_type] = [
                h for h in self._handlers[event_type] if h is not handler
            ]

    async def publish(self, event: Event) -> None:
        await self._queue.put(event)

    def publish_sync(self, event: Event) -> None:
        try:
            loop = asyncio.get_running_loop()
            loop.call_soon_threadsafe(self._queue.put_nowait, event)
        except RuntimeError:
            pass

    async def start(self) -> None:
        self._running = True
        logger.info("EventBus started")
        while self._running:
            try:
                event = await asyncio.wait_for(self._queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue

            event_type = event.event_type
            handlers = self._handlers.get(event_type, []) + self._handlers.get("*", [])

            for handler in handlers:
                try:
                    result = handler(event)
                    if asyncio.iscoroutine(result):
                        await result
                except Exception:
                    logger.exception(f"EventBus handler error for {event_type}")

    async def stop(self) -> None:
        self._running = False
        logger.info("EventBus stopped")


event_bus = EventBus()
