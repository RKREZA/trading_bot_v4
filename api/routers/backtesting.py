import asyncio
import logging
from datetime import datetime, timezone, timedelta

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
from sqlalchemy import select

from api.dependencies import get_services, AppServices
from core.common.models import BacktestRun
from core.common.database import async_session_factory

logger = logging.getLogger("trading_bot.api.backtesting")

router = APIRouter(prefix="/api/backtest", tags=["backtesting"])


class BacktestRequest(BaseModel):
    symbol: str = "XAUUSDm"
    strategy_id: str = "SMC"
    timeframe: str = "M5"
    start_date: str = ""
    initial_balance: float = 10000.0
    stress_test: bool = False
    monte_carlo_iterations: int = Field(default=500, ge=100, le=5000)
    parameters: Dict[str, Any] = Field(default_factory=dict)


async def _run_backtest_task(run_id: int, req: BacktestRequest, config: dict):
    """Background task that executes the full backtest pipeline."""
    from backtesting.backtester import PortfolioBacktester
    from backtesting.monte_carlo import MonteCarloSimulator
    from core.performance_tracker import PerformanceTracker
    from core.data.manager import DataManager
    from strategies import create_strategy

    async with async_session_factory() as session:
        run = await session.get(BacktestRun, run_id)
        run.status = "RUNNING"
        await session.commit()

    try:
        bt_config = {**config}
        bt_config.setdefault("backtest", {})
        bt_config["backtest"]["timeframe"] = req.timeframe
        bt_config["backtest"]["initial_balance_per_strategy"] = req.initial_balance

        if req.parameters:
            strat_key = req.strategy_id.split("_")[0] if "_" in req.strategy_id else req.strategy_id
            bt_config.setdefault("strategies", {}).setdefault(strat_key, {}).update(req.parameters)

        if req.start_date:
            start_dt = datetime.fromisoformat(req.start_date).replace(tzinfo=timezone.utc)
        else:
            start_dt = datetime.now(timezone.utc) - timedelta(days=90)

        data_mgr = DataManager(bt_config)
        timeframes_needed = ["M1", "M5", "M15", "H1", "D1"]
        data = {}
        for tf in timeframes_needed:
            try:
                data[tf] = await asyncio.to_thread(data_mgr.prepare_data, req.symbol, tf, start_dt)
            except Exception as e:
                logger.warning(f"Failed to load {tf} data: {e}")
                data[tf] = None

        primary_data = data.get(req.timeframe, data.get("M5"))
        if primary_data is None or len(primary_data) == 0:
            raise ValueError(f"No {req.timeframe} data available for {req.symbol}")

        strategy = await asyncio.to_thread(create_strategy, req.strategy_id, config=bt_config)
        strategies = [strategy]

        backtester = PortfolioBacktester(bt_config)
        history, equity_history = await asyncio.to_thread(
            backtester.run,
            req.symbol, strategies, primary_data,
            data.get("H1"), data.get("M15"), data.get("M5"), data.get("M1"),
            d1_data=data.get("D1"),
        )

        metrics = PerformanceTracker.calculate_metrics(history, req.initial_balance, equity_curve=equity_history)
        per_strategy = PerformanceTracker.calculate_per_strategy(history, req.initial_balance)

        mc_sim = MonteCarloSimulator(iterations=req.monte_carlo_iterations)
        mc_results = await asyncio.to_thread(mc_sim.run, history, req.initial_balance)

        stress_results = None
        if req.stress_test:
            try:
                from backtesting.stress_tester import StressTester
                tester = StressTester(bt_config)
                stress_results = await asyncio.to_thread(
                    tester.run_stress_test, req.symbol, strategies, data,
                )
            except Exception as e:
                logger.warning(f"Stress test failed: {e}")
                stress_results = {"error": str(e)}

        equity_curve = []
        if equity_history:
            seen_times = set()
            for pt in equity_history:
                t = pt.get("time", 0)
                if t not in seen_times:
                    seen_times.add(t)
                    total_eq = sum(p["equity"] for p in equity_history if p["time"] == t)
                    equity_curve.append({"time": t, "equity": round(total_eq, 2)})
            equity_curve.sort(key=lambda x: x["time"])
            if len(equity_curve) > 2000:
                step = len(equity_curve) // 2000
                equity_curve = equity_curve[::step]

        trade_list = []
        for t in history[:500]:
            trade_list.append({
                "direction": t.get("direction"),
                "fill_price": round(t.get("fill_price", 0), 5),
                "exit_price": round(t.get("exit_price", 0), 5),
                "pnl": round(t.get("pnl", 0), 2),
                "result": t.get("result"),
                "strategy_id": t.get("strategy_id"),
                "timestamp": t.get("timestamp"),
                "exit_time": t.get("exit_time"),
            })

        results = {
            "symbol": req.symbol,
            "timeframe": req.timeframe,
            "strategy_id": req.strategy_id,
            "initial_balance": req.initial_balance,
            "metrics": metrics,
            "per_strategy": per_strategy,
            "monte_carlo": mc_results,
            "stress_test": stress_results,
            "equity_curve": equity_curve,
            "trades": trade_list,
            "total_trades": len(history),
            "volatility_summary": backtester.get_volatility_summary(),
        }

        async with async_session_factory() as session:
            run = await session.get(BacktestRun, run_id)
            run.status = "COMPLETED"
            run.results = results
            run.completed_at = datetime.now(timezone.utc)
            await session.commit()

        logger.info(f"Backtest run {run_id} completed: {len(history)} trades, PnL={metrics.get('net_profit', 0)}")

    except Exception as e:
        logger.exception(f"Backtest run {run_id} failed: {e}")
        async with async_session_factory() as session:
            run = await session.get(BacktestRun, run_id)
            run.status = "FAILED"
            run.error_message = str(e)
            run.completed_at = datetime.now(timezone.utc)
            await session.commit()


@router.post("/run")
async def run_backtest(req: BacktestRequest, background_tasks: BackgroundTasks,
                       svc: AppServices = Depends(get_services)):
    async with async_session_factory() as session:
        run = BacktestRun(
            status="PENDING",
            config_snapshot={
                "symbol": req.symbol,
                "strategy_id": req.strategy_id,
                "timeframe": req.timeframe,
                "initial_balance": req.initial_balance,
                "monte_carlo_iterations": req.monte_carlo_iterations,
                "stress_test": req.stress_test,
                "parameters": req.parameters,
            },
        )
        session.add(run)
        await session.commit()
        await session.refresh(run)
        run_id = run.id

    background_tasks.add_task(_run_backtest_task, run_id, req, svc.config)

    return {"run_id": run_id, "status": "PENDING"}


@router.get("/runs")
async def list_backtest_runs(limit: int = 20, offset: int = 0):
    async with async_session_factory() as session:
        result = await session.execute(
            select(BacktestRun)
            .order_by(BacktestRun.created_at.desc())
            .offset(offset)
            .limit(limit)
        )
        runs = result.scalars().all()
        return [
            {
                "id": r.id,
                "status": r.status,
                "config": r.config_snapshot,
                "created_at": r.created_at.isoformat() if r.created_at else None,
                "completed_at": r.completed_at.isoformat() if r.completed_at else None,
                "has_results": r.results is not None,
            }
            for r in runs
        ]


@router.get("/{run_id}/results")
async def get_backtest_results(run_id: int):
    async with async_session_factory() as session:
        result = await session.execute(
            select(BacktestRun).where(BacktestRun.id == run_id)
        )
        run = result.scalar_one_or_none()
        if not run:
            raise HTTPException(status_code=404, detail="Backtest run not found")
        return {
            "id": run.id,
            "status": run.status,
            "config": run.config_snapshot,
            "results": run.results,
            "error_message": run.error_message,
            "created_at": run.created_at.isoformat() if run.created_at else None,
            "completed_at": run.completed_at.isoformat() if run.completed_at else None,
        }


@router.get("/{run_id}/status")
async def get_backtest_status(run_id: int):
    async with async_session_factory() as session:
        result = await session.execute(
            select(BacktestRun.status, BacktestRun.error_message)
            .where(BacktestRun.id == run_id)
        )
        row = result.one_or_none()
        if not row:
            raise HTTPException(status_code=404, detail="Backtest run not found")
        return {"run_id": run_id, "status": row[0], "error_message": row[1]}
