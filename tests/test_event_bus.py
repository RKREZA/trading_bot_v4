"""Tests for the async EventBus."""

import asyncio
from core.common.events import EventBus, SignalGeneratedEvent, RiskBreachEvent


class TestEventBus:
    def test_publish_subscribe(self):
        async def _run():
            bus = EventBus()
            received = []

            async def handler(event):
                received.append(event)

            bus.subscribe("SIGNAL_GENERATED", handler)
            task = asyncio.create_task(bus.start())

            event = SignalGeneratedEvent(
                signal_id="test_001",
                symbol="XAUUSDm",
                direction="BUY",
                confidence=0.85,
                strategy_id="smc_v1",
            )
            await bus.publish(event)
            await asyncio.sleep(0.2)

            await bus.stop()
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

            assert len(received) == 1
            assert received[0].symbol == "XAUUSDm"

        asyncio.run(_run())

    def test_multiple_subscribers(self):
        async def _run():
            bus = EventBus()
            counts = {"a": 0, "b": 0}

            async def handler_a(event):
                counts["a"] += 1

            async def handler_b(event):
                counts["b"] += 1

            bus.subscribe("RISK_BREACH", handler_a)
            bus.subscribe("RISK_BREACH", handler_b)
            task = asyncio.create_task(bus.start())

            event = RiskBreachEvent(
                breach_type="MAX_DRAWDOWN",
                severity="CRITICAL",
                message="test",
                current_value=12.0,
                threshold=10.0,
            )
            await bus.publish(event)
            await asyncio.sleep(0.2)

            await bus.stop()
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

            assert counts["a"] == 1
            assert counts["b"] == 1

        asyncio.run(_run())

    def test_no_subscribers_no_error(self):
        async def _run():
            bus = EventBus()
            task = asyncio.create_task(bus.start())

            event = SignalGeneratedEvent(
                signal_id="test_002",
                symbol="EURUSD",
                direction="SELL",
                confidence=0.7,
                strategy_id="trend_v1",
            )
            await bus.publish(event)
            await asyncio.sleep(0.2)

            await bus.stop()
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        asyncio.run(_run())
