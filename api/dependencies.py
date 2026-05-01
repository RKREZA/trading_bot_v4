"""
Dependency injection for FastAPI routes.
Provides access to core services without global singletons in route modules.
"""
import os
from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession

from core.common.database import async_session_factory, init_db
from core.config.loader import load_config
from core.risk.risk_guardian import RiskGuardian
from core.execution.order_manager import OrderManager
from core.execution.reconciliation_engine import ReconciliationEngine
from core.data.manager import DataManager
from core.data.cache import redis_cache
from core.common.events import event_bus


class AppServices:
    """Holds all initialized service instances for the application lifecycle."""

    def __init__(self):
        self.config: dict = {}
        self.risk_engine: RiskGuardian = None
        self.order_manager: OrderManager = None
        self.recon_engine: ReconciliationEngine = None
        self.data_manager: DataManager = None
        self.is_trading: bool = False
        self.strategies: dict = {}

    async def initialize(self) -> None:
        self.config = load_config()

        self.risk_engine = RiskGuardian(self.config)
        self.order_manager = OrderManager(self.config)
        self.recon_engine = ReconciliationEngine(self.order_manager)
        self.data_manager = DataManager(self.config)

        self._register_strategies()

        await init_db()
        await redis_cache.connect()

    def _register_strategies(self) -> None:
        import logging
        logger = logging.getLogger("trading_bot.dependencies")
        try:
            from strategies import STRATEGY_REGISTRY, create_strategy
            strat_cfg = self.config.get("strategies", {})
            if strat_cfg:
                for sid, cfg in strat_cfg.items():
                    st_type = cfg.get("type", sid)
                    try:
                        strategy = create_strategy(sid, st_type, self.config)
                        self.strategies[sid] = strategy
                        logger.info(f"Registered strategy: {sid} ({strategy.__class__.__name__})")
                    except Exception as e:
                        logger.warning(f"Failed to create strategy {sid}: {e}")
            if not self.strategies and STRATEGY_REGISTRY:
                for st_type, st_class in STRATEGY_REGISTRY.items():
                    sid = st_type.lower() + "_v1"
                    try:
                        strategy = st_class(sid, self.config)
                        self.strategies[sid] = strategy
                        logger.info(f"Auto-registered strategy: {sid} ({st_class.__name__})")
                    except Exception as e:
                        logger.warning(f"Failed to auto-register {st_type}: {e}")
        except Exception as e:
            logger.error(f"Strategy registration failed: {e}")

    async def shutdown(self) -> None:
        from core.common.database import close_db
        await close_db()
        await redis_cache.close()
        await event_bus.stop()


services = AppServices()


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with async_session_factory() as session:
        yield session


def get_services() -> AppServices:
    return services


def get_config() -> dict:
    return services.config


def get_risk_engine() -> RiskGuardian:
    return services.risk_engine


def get_order_manager() -> OrderManager:
    return services.order_manager
