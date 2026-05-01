"""
Audit trail — persists every signal, execution, risk decision, and system event
to the audit_logs table for compliance and forensic analysis.
"""

import asyncio
import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Optional

logger = logging.getLogger("trading_bot.audit")


class AuditTrail:

    def __init__(self):
        self._queue: asyncio.Queue = None
        self._task: Optional[asyncio.Task] = None
        self._running = False

    async def start(self) -> None:
        if self._running:
            return
        self._queue = asyncio.Queue(maxsize=10000)
        self._running = True
        self._task = asyncio.create_task(self._flush_loop())
        logger.info("AuditTrail started")

    async def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        if self._queue and not self._queue.empty():
            await self._flush_batch()
        logger.info("AuditTrail stopped")

    def log(
        self,
        level: str,
        category: str,
        message: str,
        data: Optional[Dict[str, Any]] = None,
        correlation_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> None:
        event = {
            "event_id": uuid.uuid4().hex[:16],
            "correlation_id": correlation_id,
            "timestamp": datetime.now(timezone.utc),
            "level": level,
            "category": category,
            "message": message,
            "data": data,
            "user_id": user_id,
        }

        if self._queue and not self._queue.full():
            try:
                self._queue.put_nowait(event)
            except asyncio.QueueFull:
                logger.warning("AuditTrail queue full, dropping event")
        else:
            logger.debug(f"[AUDIT] {level} {category}: {message}")

    def log_signal(self, signal_data: Dict, correlation_id: str = None) -> None:
        self.log("INFO", "SIGNAL", f"Signal generated: {signal_data.get('direction', '?')} {signal_data.get('symbol', '?')}",
                 data=signal_data, correlation_id=correlation_id)

    def log_execution(self, exec_data: Dict, correlation_id: str = None) -> None:
        self.log("INFO", "EXECUTION", f"Trade executed: ticket={exec_data.get('ticket', '?')}",
                 data=exec_data, correlation_id=correlation_id)

    def log_risk_decision(self, decision: str, details: Dict, correlation_id: str = None) -> None:
        level = "WARNING" if "REJECT" in decision.upper() or "HALT" in decision.upper() else "INFO"
        self.log(level, "RISK", f"Risk decision: {decision}", data=details, correlation_id=correlation_id)

    def log_system(self, message: str, level: str = "INFO", data: Dict = None) -> None:
        self.log(level, "SYSTEM", message, data=data)

    async def _flush_loop(self) -> None:
        while self._running:
            try:
                await asyncio.sleep(2.0)
                if self._queue and not self._queue.empty():
                    await self._flush_batch()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"AuditTrail flush error: {e}")

    async def _flush_batch(self) -> None:
        batch = []
        while not self._queue.empty() and len(batch) < 100:
            try:
                batch.append(self._queue.get_nowait())
            except asyncio.QueueEmpty:
                break

        if not batch:
            return

        try:
            from core.common.database import async_session_factory
            from core.common.models import AuditLog

            async with async_session_factory() as session:
                for event in batch:
                    log = AuditLog(
                        event_id=event["event_id"],
                        correlation_id=event.get("correlation_id"),
                        timestamp=event["timestamp"],
                        level=event["level"],
                        category=event["category"],
                        message=event["message"],
                        data=event.get("data"),
                        user_id=event.get("user_id"),
                    )
                    session.add(log)
                await session.commit()
        except Exception as e:
            logger.error(f"AuditTrail DB flush failed ({len(batch)} events): {e}")


audit_trail = AuditTrail()
