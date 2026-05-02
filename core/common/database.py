import os
import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from core.common.models import Base

logger = logging.getLogger("trading_bot.database")

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "sqlite+aiosqlite:///./trading_bot.db",
)

_engine = None
_session_factory = None


def _get_engine():
    global _engine
    if _engine is None:
        kwargs = {"echo": False}
        if DATABASE_URL.startswith("postgresql"):
            kwargs.update(pool_size=10, max_overflow=20, pool_pre_ping=True)
        _engine = create_async_engine(DATABASE_URL, **kwargs)
    return _engine


def _get_session_factory():
    global _session_factory
    if _session_factory is None:
        _session_factory = async_sessionmaker(_get_engine(), expire_on_commit=False)
    return _session_factory


def async_session_factory():
    return _get_session_factory()()


async def _migrate_mt_accounts(conn) -> None:
    """Add risk columns to mt_accounts if missing (SQLite won't add via create_all)."""
    import sqlalchemy as sa
    result = await conn.execute(sa.text("PRAGMA table_info(mt_accounts)"))
    existing = {row[1] for row in result}
    migrations = [
        ("max_drawdown_pct", "REAL DEFAULT 10.0"),
        ("risk_per_trade_pct", "REAL DEFAULT 1.0"),
        ("max_daily_drawdown_pct", "REAL DEFAULT 5.0"),
    ]
    for col, typedef in migrations:
        if col not in existing:
            await conn.execute(sa.text(f"ALTER TABLE mt_accounts ADD COLUMN {col} {typedef}"))
            logger.info(f"Added column mt_accounts.{col}")


async def init_db() -> None:
    engine = _get_engine()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        await _migrate_mt_accounts(conn)
    logger.info("Database tables initialized")


async def close_db() -> None:
    global _engine, _session_factory
    if _engine:
        await _engine.dispose()
        _engine = None
        _session_factory = None
    logger.info("Database connections closed")


@asynccontextmanager
async def get_session() -> AsyncGenerator[AsyncSession, None]:
    async with _get_session_factory()() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
