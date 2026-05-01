from sqlalchemy import String, Float, Integer, DateTime, JSON, Boolean, Text, ForeignKey
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from datetime import datetime, timezone
from typing import Optional


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class MTAccount(Base):
    __tablename__ = "mt_accounts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[Optional[str]] = mapped_column(String, nullable=True, index=True)
    login: Mapped[int] = mapped_column(Integer, nullable=False)
    password: Mapped[str] = mapped_column(String, default="")
    server: Mapped[str] = mapped_column(String, nullable=False)
    broker_utc_offset: Mapped[int] = mapped_column(Integer, default=0)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    label: Mapped[str] = mapped_column(String, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)

    trades: Mapped[list["Trade"]] = relationship(back_populates="account", lazy="selectin")
    equity_snapshots: Mapped[list["EquitySnapshot"]] = relationship(back_populates="account", lazy="selectin")


class Trade(Base):
    __tablename__ = "trades"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    user_id: Mapped[Optional[str]] = mapped_column(String, nullable=True, index=True)
    account_id: Mapped[Optional[int]] = mapped_column(Integer, ForeignKey("mt_accounts.id"), nullable=True)
    signal_id: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    execution_id: Mapped[Optional[str]] = mapped_column(String, nullable=True, unique=True)
    symbol: Mapped[str] = mapped_column(String, nullable=False, index=True)
    direction: Mapped[str] = mapped_column(String, nullable=False)
    entry_price: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    exit_price: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    volume: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    sl: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    tp: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    pnl: Mapped[float] = mapped_column(Float, default=0.0)
    status: Mapped[str] = mapped_column(String, default="OPEN")
    strategy_id: Mapped[Optional[str]] = mapped_column(String, nullable=True, index=True)
    ticket: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    execution_latency_ms: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    slippage_pips: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    spread_at_entry: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    intent_hash: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    entry_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    exit_time: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    metadata_json: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)

    account: Mapped[Optional["MTAccount"]] = relationship(back_populates="trades")


class AuditLog(Base):
    __tablename__ = "audit_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    event_id: Mapped[str] = mapped_column(String, unique=True, index=True)
    correlation_id: Mapped[Optional[str]] = mapped_column(String, nullable=True, index=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)
    level: Mapped[str] = mapped_column(String)
    category: Mapped[str] = mapped_column(String, index=True)
    message: Mapped[str] = mapped_column(Text)
    data: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    user_id: Mapped[Optional[str]] = mapped_column(String, nullable=True, index=True)


class BacktestRun(Base):
    __tablename__ = "backtest_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[Optional[str]] = mapped_column(String, nullable=True, index=True)
    status: Mapped[str] = mapped_column(String, default="PENDING")
    config_snapshot: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    results: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)


class EquitySnapshot(Base):
    __tablename__ = "equity_snapshots"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    account_id: Mapped[int] = mapped_column(Integer, ForeignKey("mt_accounts.id"), index=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)
    equity: Mapped[float] = mapped_column(Float)
    balance: Mapped[float] = mapped_column(Float)
    margin: Mapped[float] = mapped_column(Float, default=0.0)
    free_margin: Mapped[float] = mapped_column(Float, default=0.0)
    drawdown_pct: Mapped[float] = mapped_column(Float, default=0.0)

    account: Mapped["MTAccount"] = relationship(back_populates="equity_snapshots")


class SystemState(Base):
    __tablename__ = "system_state"

    key: Mapped[str] = mapped_column(String, primary_key=True)
    value: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)
