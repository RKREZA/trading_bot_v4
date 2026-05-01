"""Tests for the AuditTrail observability module."""

import asyncio
from core.observability.audit import AuditTrail


class TestAuditTrail:
    def test_start_stop(self):
        async def _run():
            trail = AuditTrail()
            await trail.start()
            assert trail._running is True
            await trail.stop()
            assert trail._running is False

        asyncio.run(_run())

    def test_log_queues_event(self):
        async def _run():
            trail = AuditTrail()
            await trail.start()
            trail.log("INFO", "TEST", "test message", data={"key": "value"})
            assert not trail._queue.empty()
            await trail.stop()

        asyncio.run(_run())

    def test_convenience_methods(self):
        async def _run():
            trail = AuditTrail()
            await trail.start()
            trail.log_signal({"direction": "BUY", "symbol": "XAUUSDm"})
            trail.log_execution({"ticket": 12345})
            trail.log_risk_decision("REJECT", {"reason": "max drawdown"})
            trail.log_system("System started")
            assert trail._queue.qsize() == 4
            await trail.stop()

        asyncio.run(_run())

    def test_log_before_start_falls_back(self):
        trail = AuditTrail()
        trail.log("INFO", "TEST", "before start")

    def test_double_start_idempotent(self):
        async def _run():
            trail = AuditTrail()
            await trail.start()
            await trail.start()
            assert trail._running is True
            await trail.stop()

        asyncio.run(_run())
