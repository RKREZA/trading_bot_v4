"""Initial schema

Revision ID: 001
Revises:
Create Date: 2026-05-02
"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa

revision: str = "001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "mt_accounts",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("user_id", sa.String(), nullable=True, index=True),
        sa.Column("login", sa.Integer(), nullable=False),
        sa.Column("server", sa.String(), nullable=False),
        sa.Column("broker_utc_offset", sa.Integer(), server_default="0"),
        sa.Column("is_active", sa.Boolean(), server_default=sa.text("true")),
        sa.Column("label", sa.String(), server_default=""),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.PrimaryKeyConstraint("id"),
    )

    op.create_table(
        "trades",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("user_id", sa.String(), nullable=True, index=True),
        sa.Column("account_id", sa.Integer(), sa.ForeignKey("mt_accounts.id"), nullable=True),
        sa.Column("signal_id", sa.String(), nullable=True),
        sa.Column("execution_id", sa.String(), nullable=True, unique=True),
        sa.Column("symbol", sa.String(), nullable=False, index=True),
        sa.Column("direction", sa.String(), nullable=False),
        sa.Column("entry_price", sa.Float(), nullable=True),
        sa.Column("exit_price", sa.Float(), nullable=True),
        sa.Column("volume", sa.Float(), nullable=True),
        sa.Column("sl", sa.Float(), nullable=True),
        sa.Column("tp", sa.Float(), nullable=True),
        sa.Column("pnl", sa.Float(), server_default="0.0"),
        sa.Column("status", sa.String(), server_default="OPEN"),
        sa.Column("strategy_id", sa.String(), nullable=True, index=True),
        sa.Column("ticket", sa.Integer(), nullable=True),
        sa.Column("execution_latency_ms", sa.Float(), nullable=True),
        sa.Column("slippage_pips", sa.Float(), nullable=True),
        sa.Column("spread_at_entry", sa.Float(), nullable=True),
        sa.Column("intent_hash", sa.String(), nullable=True),
        sa.Column("entry_time", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("exit_time", sa.DateTime(timezone=True), nullable=True),
        sa.Column("metadata_json", sa.JSON(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )

    op.create_table(
        "audit_logs",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("event_id", sa.String(), nullable=False, unique=True, index=True),
        sa.Column("correlation_id", sa.String(), nullable=True, index=True),
        sa.Column("timestamp", sa.DateTime(timezone=True), server_default=sa.func.now(), index=True),
        sa.Column("level", sa.String(), nullable=False),
        sa.Column("category", sa.String(), nullable=False, index=True),
        sa.Column("message", sa.Text(), nullable=False),
        sa.Column("data", sa.JSON(), nullable=True),
        sa.Column("user_id", sa.String(), nullable=True, index=True),
        sa.PrimaryKeyConstraint("id"),
    )

    op.create_table(
        "backtest_runs",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("user_id", sa.String(), nullable=True, index=True),
        sa.Column("status", sa.String(), server_default="PENDING"),
        sa.Column("config_snapshot", sa.JSON(), nullable=True),
        sa.Column("results", sa.JSON(), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )

    op.create_table(
        "equity_snapshots",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("account_id", sa.Integer(), sa.ForeignKey("mt_accounts.id"), nullable=False, index=True),
        sa.Column("timestamp", sa.DateTime(timezone=True), server_default=sa.func.now(), index=True),
        sa.Column("equity", sa.Float(), nullable=False),
        sa.Column("balance", sa.Float(), nullable=False),
        sa.Column("margin", sa.Float(), server_default="0.0"),
        sa.Column("free_margin", sa.Float(), server_default="0.0"),
        sa.Column("drawdown_pct", sa.Float(), server_default="0.0"),
        sa.PrimaryKeyConstraint("id"),
    )

    op.create_table(
        "system_state",
        sa.Column("key", sa.String(), nullable=False),
        sa.Column("value", sa.JSON(), nullable=True),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.PrimaryKeyConstraint("key"),
    )


def downgrade() -> None:
    op.drop_table("system_state")
    op.drop_table("equity_snapshots")
    op.drop_table("backtest_runs")
    op.drop_table("audit_logs")
    op.drop_table("trades")
    op.drop_table("mt_accounts")
