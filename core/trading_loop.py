"""
Async trading loop — orchestrates the strategy cycle for the live platform.
Driven by the API's lifespan; runs as a background asyncio task.
"""
import asyncio
import logging
from datetime import datetime, timezone
from typing import Dict, Any

from core.data.mt5_service import mt5_service
from core.time.time_service import time_service
from core.common.events import event_bus, SystemStatusEvent

logger = logging.getLogger("trading_bot.loop")


class TradingLoop:

    def __init__(self, services):
        self.svc = services
        self._running = False
        self._task = None

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._run())
        logger.info("Trading loop started")

    async def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
            self._task = None
        logger.info("Trading loop stopped")

    async def _run(self) -> None:
        while self._running:
            try:
                if not self.svc.is_trading:
                    await asyncio.sleep(1)
                    continue

                if not mt5_service.connected:
                    await asyncio.sleep(5)
                    continue

                symbols = self.svc.config.get("symbols", ["XAUUSDm"])

                for symbol in symbols:
                    tick = await mt5_service.async_get_tick(symbol)
                    if not tick:
                        continue

                    for sid, strategy in self.svc.strategies.items():
                        if not strategy.enabled:
                            continue
                        if not strategy.is_symbol_allowed(symbol):
                            continue

                        gov_ok, gov_reason = self.svc.risk_engine.check_governance(
                            tick.get("bid", 0),
                            tick.get("bid", 0),
                        )
                        if not gov_ok:
                            logger.debug(f"Governance block for {sid}: {gov_reason}")
                            continue

                        strat_ok, strat_reason = self.svc.risk_engine.check_strategy_governance(sid)
                        if not strat_ok:
                            logger.debug(f"Strategy governance block: {strat_reason}")
                            continue

                await self._emit_status()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Trading loop error: {e}", exc_info=True)

            await asyncio.sleep(1)

    async def _emit_status(self) -> None:
        try:
            account = self.svc.recon_engine.get_account_summary()
            await event_bus.publish(SystemStatusEvent(
                mt5_connected=mt5_service.connected,
                is_trading=self.svc.is_trading,
                active_strategies=len(self.svc.strategies),
                equity=account.get("equity", 0),
                balance=account.get("balance", 0),
                drawdown_pct=account.get("drawdown", 0) * 100,
            ))
        except Exception:
            pass
