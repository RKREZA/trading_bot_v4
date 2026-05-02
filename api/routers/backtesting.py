import asyncio
import logging
import time
import io
import csv
from datetime import datetime, timezone, timedelta
from collections import defaultdict

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
from sqlalchemy import select
import numpy as np

from api.dependencies import get_services, AppServices
from core.common.models import BacktestRun
from core.common.database import async_session_factory

logger = logging.getLogger("trading_bot.api.backtesting")

router = APIRouter(prefix="/api/backtest", tags=["backtesting"])

_progress_store: Dict[int, Dict[str, Any]] = {}


def _update_progress(run_id: int, pct: float, stage: str, detail: str = ""):
    _progress_store[run_id] = {
        "progress_pct": round(pct, 1),
        "stage": stage,
        "detail": detail,
        "updated_at": time.time(),
    }


class BacktestRequest(BaseModel):
    symbol: str = "XAUUSDm"
    strategy_id: str = "SMC"
    timeframe: str = "M5"
    start_date: str = ""
    end_date: str = ""
    initial_balance: float = 10000.0
    risk_per_trade_pct: float = 1.0
    commission_per_lot: float = 0.0
    spread_pips: float = 0.0
    slippage_points: float = 0.0
    max_open_trades: int = 1
    session_filter: str = ""
    stress_test: bool = False
    walk_forward: bool = False
    walk_forward_windows: int = Field(default=5, ge=2, le=20)
    walk_forward_oos_pct: float = Field(default=0.3, ge=0.1, le=0.5)
    monte_carlo_iterations: int = Field(default=500, ge=100, le=5000)
    random_seed: int = 42
    use_m1_execution: bool = True
    parameters: Dict[str, Any] = Field(default_factory=dict)


def _compute_analytics(history: list, equity_history: list, initial_balance: float) -> dict:
    if not history:
        return {}

    day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    pnls = [t.get("pnl", 0) for t in history]

    # Drawdown curve from equity history
    drawdown_curve = []
    if equity_history:
        seen: Dict[float, float] = {}
        for pt in equity_history:
            t = pt.get("time", 0)
            seen[t] = seen.get(t, 0) + pt.get("equity", 0)
        peak = 0.0
        for t in sorted(seen):
            eq = seen[t]
            peak = max(peak, eq)
            dd = ((peak - eq) / peak * 100) if peak > 0 else 0
            drawdown_curve.append({"time": t, "drawdown": round(dd, 2), "equity": round(eq, 2)})
        if len(drawdown_curve) > 2000:
            step = len(drawdown_curve) // 2000
            drawdown_curve = drawdown_curve[::step]

    # Monthly returns
    monthly_returns: Dict[str, dict] = {}
    for t in history:
        ts = t.get("timestamp", 0)
        if not ts:
            continue
        dt = datetime.fromtimestamp(ts, tz=timezone.utc)
        key = dt.strftime("%Y-%m")
        if key not in monthly_returns:
            monthly_returns[key] = {"pnl": 0, "trades": 0, "wins": 0}
        monthly_returns[key]["pnl"] += t.get("pnl", 0)
        monthly_returns[key]["trades"] += 1
        if t.get("pnl", 0) > 0:
            monthly_returns[key]["wins"] += 1
    for k in monthly_returns:
        m = monthly_returns[k]
        m["pnl"] = round(m["pnl"], 2)
        m["win_rate"] = round(m["wins"] / m["trades"] * 100, 1) if m["trades"] > 0 else 0

    # Hourly distribution
    hourly = defaultdict(lambda: {"trades": 0, "pnl": 0.0, "wins": 0})
    for t in history:
        ts = t.get("timestamp", 0)
        if ts:
            h = datetime.fromtimestamp(ts, tz=timezone.utc).hour
            hourly[h]["trades"] += 1
            hourly[h]["pnl"] += t.get("pnl", 0)
            if t.get("pnl", 0) > 0:
                hourly[h]["wins"] += 1
    hourly_dist = {}
    for h in range(24):
        d = hourly.get(h, {"trades": 0, "pnl": 0, "wins": 0})
        hourly_dist[str(h)] = {
            "trades": d["trades"],
            "pnl": round(d["pnl"], 2),
            "win_rate": round(d["wins"] / d["trades"] * 100, 1) if d["trades"] > 0 else 0,
        }

    # Day of week distribution
    daily = defaultdict(lambda: {"trades": 0, "pnl": 0.0, "wins": 0})
    for t in history:
        ts = t.get("timestamp", 0)
        if ts:
            d = datetime.fromtimestamp(ts, tz=timezone.utc).weekday()
            daily[d]["trades"] += 1
            daily[d]["pnl"] += t.get("pnl", 0)
            if t.get("pnl", 0) > 0:
                daily[d]["wins"] += 1
    day_dist = {}
    for d in range(7):
        dd = daily.get(d, {"trades": 0, "pnl": 0, "wins": 0})
        day_dist[day_names[d]] = {
            "trades": dd["trades"],
            "pnl": round(dd["pnl"], 2),
            "win_rate": round(dd["wins"] / dd["trades"] * 100, 1) if dd["trades"] > 0 else 0,
        }

    # Session performance
    sess = defaultdict(lambda: {"trades": 0, "pnl": 0.0, "wins": 0})
    for t in history:
        s = t.get("session", "UNKNOWN")
        sess[s]["trades"] += 1
        sess[s]["pnl"] += t.get("pnl", 0)
        if t.get("pnl", 0) > 0:
            sess[s]["wins"] += 1
    session_perf = {}
    for s, d in sess.items():
        session_perf[s] = {
            "trades": d["trades"],
            "pnl": round(d["pnl"], 2),
            "win_rate": round(d["wins"] / d["trades"] * 100, 1) if d["trades"] > 0 else 0,
        }

    # Streak analysis
    max_win = max_loss = cur_win = cur_loss = 0
    for t in history:
        if t.get("pnl", 0) > 0:
            cur_win += 1
            cur_loss = 0
            max_win = max(max_win, cur_win)
        else:
            cur_loss += 1
            cur_win = 0
            max_loss = max(max_loss, cur_loss)

    # Duration analysis
    durations = []
    for t in history:
        entry_ts = t.get("timestamp", 0)
        exit_ts = t.get("exit_time", 0)
        if entry_ts and exit_ts and isinstance(exit_ts, (int, float)):
            dur = exit_ts - entry_ts
            if dur > 0:
                durations.append(dur)

    # Advanced metrics
    cumulative = np.cumsum(pnls)
    equity_arr = initial_balance + cumulative
    peak_arr = np.maximum.accumulate(equity_arr)
    dd_arr = np.where(peak_arr > 0, (peak_arr - equity_arr) / peak_arr * 100, 0)
    max_dd = float(np.max(dd_arr)) if len(dd_arr) > 0 else 0
    net_profit = sum(pnls)
    recovery_factor = abs(net_profit / (max_dd * initial_balance / 100)) if max_dd > 0 else 0
    ulcer_index = float(np.sqrt(np.mean(dd_arr ** 2))) if len(dd_arr) > 0 else 0

    gains = [p for p in pnls if p > 0]
    losses_list = [abs(p) for p in pnls if p < 0]
    tail_ratio = 0.0
    if gains and losses_list:
        p95g = float(np.percentile(gains, 95))
        p95l = float(np.percentile(losses_list, 95))
        tail_ratio = round(p95g / p95l, 2) if p95l > 0 else 0

    avg_win = round(float(np.mean(gains)), 2) if gains else 0
    avg_loss = round(float(np.mean(losses_list)), 2) if losses_list else 0
    payoff = round(avg_win / avg_loss, 2) if avg_loss > 0 else 0
    w = len(gains) / len(pnls) if pnls else 0
    kelly = round((w - (1 - w) / payoff) * 100, 2) if payoff > 0 and w > 0 else 0

    # Rolling Sharpe (30-trade window)
    rolling_sharpe = []
    window = 30
    if len(pnls) >= window:
        pnl_arr = np.array(pnls)
        for i in range(window, len(pnl_arr) + 1):
            chunk = pnl_arr[i - window : i]
            std_r = float(np.std(chunk))
            sharpe = (float(np.mean(chunk)) / std_r * np.sqrt(260)) if std_r > 0 else 0
            ts_val = history[i - 1].get("timestamp", 0)
            rolling_sharpe.append({"time": ts_val, "sharpe": round(float(sharpe), 2)})
        if len(rolling_sharpe) > 500:
            step = len(rolling_sharpe) // 500
            rolling_sharpe = rolling_sharpe[::step]

    # PnL distribution buckets
    bucket_size = max(1, int((max(pnls) - min(pnls)) / 30)) if len(pnls) > 1 else 50
    pnl_buckets: Dict[int, int] = defaultdict(int)
    for p in pnls:
        b = int(p // bucket_size) * bucket_size
        pnl_buckets[b] += 1
    pnl_distribution = [{"bucket": k, "count": v} for k, v in sorted(pnl_buckets.items())]

    # Cumulative PnL
    cumulative_pnl = []
    running = 0.0
    for t in history:
        running += t.get("pnl", 0)
        cumulative_pnl.append({"time": t.get("timestamp", 0), "pnl": round(running, 2)})
    if len(cumulative_pnl) > 2000:
        step = len(cumulative_pnl) // 2000
        cumulative_pnl = cumulative_pnl[::step]

    return {
        "drawdown_curve": drawdown_curve,
        "monthly_returns": dict(sorted(monthly_returns.items())),
        "hourly_distribution": hourly_dist,
        "day_distribution": day_dist,
        "session_performance": session_perf,
        "streak_analysis": {"max_win_streak": max_win, "max_loss_streak": max_loss},
        "duration_analysis": {
            "avg_minutes": round(float(np.mean(durations)) / 60, 1) if durations else 0,
            "median_minutes": round(float(np.median(durations)) / 60, 1) if durations else 0,
            "min_minutes": round(min(durations) / 60, 1) if durations else 0,
            "max_minutes": round(max(durations) / 60, 1) if durations else 0,
        },
        "advanced_metrics": {
            "recovery_factor": round(float(recovery_factor), 2),
            "ulcer_index": round(float(ulcer_index), 2),
            "tail_ratio": tail_ratio,
            "payoff_ratio": payoff,
            "kelly_criterion": kelly,
            "best_trade": round(float(max(pnls)), 2) if pnls else 0,
            "worst_trade": round(float(min(pnls)), 2) if pnls else 0,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
        },
        "rolling_sharpe": rolling_sharpe,
        "pnl_distribution": pnl_distribution,
        "cumulative_pnl": cumulative_pnl,
    }


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
        _update_progress(run_id, 5, "CONFIGURING", "Building config")

        bt_config = {**config}
        bt_config.setdefault("backtest", {})
        bt_config["backtest"]["timeframe"] = req.timeframe
        bt_config["backtest"]["initial_balance_per_strategy"] = req.initial_balance
        bt_config["backtest"]["random_seed"] = req.random_seed

        if req.risk_per_trade_pct > 0:
            bt_config.setdefault("risk_governance", {})["risk_per_trade_pct"] = req.risk_per_trade_pct

        if req.commission_per_lot > 0:
            sym_cfg = bt_config.setdefault("symbols_config", {}).setdefault(req.symbol, {})
            sym_cfg["commission_per_lot"] = req.commission_per_lot

        if req.spread_pips > 0:
            sym_cfg = bt_config.setdefault("symbols_config", {}).setdefault(req.symbol, {})
            sym_cfg["spread_pips"] = req.spread_pips

        if req.slippage_points > 0:
            exec_cfg = bt_config.setdefault("execution", {})
            exec_cfg["entry_slippage_points"] = req.slippage_points

        if req.parameters:
            strat_key = req.strategy_id.split("_")[0] if "_" in req.strategy_id else req.strategy_id
            bt_config.setdefault("strategies", {}).setdefault(strat_key, {}).update(req.parameters)

        if req.start_date:
            start_dt = datetime.fromisoformat(req.start_date).replace(tzinfo=timezone.utc)
        else:
            start_dt = datetime.now(timezone.utc) - timedelta(days=90)

        if req.end_date:
            end_dt = datetime.fromisoformat(req.end_date).replace(tzinfo=timezone.utc)
        else:
            end_dt = datetime.now(timezone.utc)

        _update_progress(run_id, 10, "SYNCING_DATA", "Syncing market data")
        data_mgr = DataManager(bt_config)

        timeframes_needed = ["M1", "M5", "M15", "H1", "H4", "D1"]
        for idx, tf in enumerate(timeframes_needed):
            pct = 10 + (idx / len(timeframes_needed)) * 15
            _update_progress(run_id, pct, "SYNCING_DATA", f"Syncing {tf}")
            try:
                await asyncio.to_thread(data_mgr.sync.update_incremental, req.symbol, tf)
            except Exception as e:
                logger.debug(f"Pre-sync {tf} skipped: {e}")

        _update_progress(run_id, 25, "LOADING_DATA", "Loading candle data")
        data = {}
        for tf in timeframes_needed:
            try:
                data[tf] = await asyncio.to_thread(data_mgr.prepare_data, req.symbol, tf, start_dt)
            except Exception as e:
                logger.warning(f"Failed to load {tf} data: {e}")
                data[tf] = None

        primary_data = data.get(req.timeframe, data.get("M5"))

        if primary_data is not None and len(primary_data) > 0:
            end_ts = int(end_dt.timestamp())
            mask = primary_data.time <= end_ts
            if not np.all(mask):
                idx_cut = int(np.sum(mask))
                if idx_cut > 0:
                    primary_data = primary_data[:idx_cut]
        if primary_data is None or len(primary_data) == 0:
            raise ValueError(f"No {req.timeframe} data available for {req.symbol}")

        _update_progress(run_id, 30, "RUNNING_BACKTEST", f"Executing on {len(primary_data)} bars")

        strategy = await asyncio.to_thread(create_strategy, req.strategy_id, config=bt_config)
        strategies = [strategy]

        backtester = PortfolioBacktester(bt_config)
        history, equity_history = await asyncio.to_thread(
            backtester.run,
            req.symbol, strategies, primary_data,
            data.get("H1"), data.get("M15"), data.get("M5"), data.get("M1"),
            d1_data=data.get("D1"), h4_data=data.get("H4"),
        )

        _update_progress(run_id, 70, "COMPUTING_METRICS", "Calculating performance")
        metrics = PerformanceTracker.calculate_metrics(history, req.initial_balance, equity_curve=equity_history)
        per_strategy = PerformanceTracker.calculate_per_strategy(history, req.initial_balance)

        _update_progress(run_id, 75, "MONTE_CARLO", f"Running {req.monte_carlo_iterations} simulations")
        mc_sim = MonteCarloSimulator(iterations=req.monte_carlo_iterations, seed=req.random_seed)
        mc_results = await asyncio.to_thread(mc_sim.run, history, req.initial_balance)

        stress_results = None
        if req.stress_test:
            _update_progress(run_id, 80, "STRESS_TEST", "Running stress scenarios")
            try:
                from backtesting.stress_tester import StressTester
                tester = StressTester(bt_config)
                stress_results = await asyncio.to_thread(
                    tester.run_stress_test, req.symbol, strategies, data,
                )
            except Exception as e:
                logger.warning(f"Stress test failed: {e}")
                stress_results = {"error": str(e)}

        wf_results = None
        if req.walk_forward:
            _update_progress(run_id, 85, "WALK_FORWARD", "Running walk-forward validation")
            try:
                from backtesting.walk_forward import WalkForwardValidator
                wf = WalkForwardValidator(bt_config)
                wf_results = await asyncio.to_thread(
                    wf.run_validation, req.symbol, strategies, data,
                    window_weeks=8, test_weeks=2, run_mc=True,
                )
            except Exception as e:
                logger.warning(f"Walk-forward failed: {e}")
                wf_results = {"error": str(e)}

        _update_progress(run_id, 90, "ANALYTICS", "Computing advanced analytics")
        analytics = _compute_analytics(history, equity_history, req.initial_balance)

        equity_curve = []
        if equity_history:
            seen_times: Dict[float, float] = {}
            for pt in equity_history:
                t = pt.get("time", 0)
                seen_times[t] = seen_times.get(t, 0) + pt.get("equity", 0)
            for t in sorted(seen_times):
                equity_curve.append({"time": t, "equity": round(seen_times[t], 2)})
            if len(equity_curve) > 2000:
                step = len(equity_curve) // 2000
                equity_curve = equity_curve[::step]

        trade_list = []
        for t in history[:1000]:
            trade_list.append({
                "direction": t.get("direction"),
                "fill_price": round(t.get("fill_price", 0), 5),
                "exit_price": round(t.get("exit_price", 0), 5),
                "pnl": round(t.get("pnl", 0), 2),
                "result": t.get("result"),
                "strategy_id": t.get("strategy_id"),
                "session": t.get("session", ""),
                "timestamp": t.get("timestamp"),
                "exit_time": t.get("exit_time"),
                "lots": t.get("lots", 0),
                "sl": round(t.get("sl", 0), 5),
                "tp": round(t.get("tp", 0), 5),
            })

        _update_progress(run_id, 95, "SAVING", "Saving results")

        results = {
            "symbol": req.symbol,
            "timeframe": req.timeframe,
            "strategy_id": req.strategy_id,
            "initial_balance": req.initial_balance,
            "start_date": req.start_date or start_dt.isoformat(),
            "end_date": req.end_date or end_dt.isoformat(),
            "duration_days": (end_dt - start_dt).days,
            "metrics": metrics,
            "per_strategy": per_strategy,
            "monte_carlo": mc_results,
            "stress_test": stress_results,
            "walk_forward": wf_results,
            "analytics": analytics,
            "equity_curve": equity_curve,
            "trades": trade_list,
            "total_trades": len(history),
            "total_bars": len(primary_data),
            "volatility_summary": backtester.get_volatility_summary(),
        }

        async with async_session_factory() as session:
            run = await session.get(BacktestRun, run_id)
            run.status = "COMPLETED"
            run.results = results
            run.completed_at = datetime.now(timezone.utc)
            await session.commit()

        _update_progress(run_id, 100, "COMPLETED", f"{len(history)} trades")
        logger.info(f"Backtest run {run_id} completed: {len(history)} trades, PnL={metrics.get('net_profit', 0)}")

    except Exception as e:
        logger.exception(f"Backtest run {run_id} failed: {e}")
        _update_progress(run_id, 0, "FAILED", str(e))
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
            config_snapshot=req.model_dump(),
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

        progress = _progress_store.get(run_id, {})
        return {
            "run_id": run_id,
            "status": row[0],
            "error_message": row[1],
            "progress_pct": progress.get("progress_pct", 0),
            "stage": progress.get("stage", ""),
            "detail": progress.get("detail", ""),
        }


@router.get("/{run_id}/export")
async def export_backtest_csv(run_id: int):
    async with async_session_factory() as session:
        result = await session.execute(
            select(BacktestRun).where(BacktestRun.id == run_id)
        )
        run = result.scalar_one_or_none()
        if not run or not run.results:
            raise HTTPException(status_code=404, detail="No results to export")

    trades = run.results.get("trades", [])
    buf = io.StringIO()
    writer = csv.DictWriter(
        buf,
        fieldnames=["direction", "fill_price", "exit_price", "pnl", "result",
                     "strategy_id", "session", "timestamp", "exit_time", "lots", "sl", "tp"],
    )
    writer.writeheader()
    for t in trades:
        writer.writerow(t)
    buf.seek(0)

    return StreamingResponse(
        iter([buf.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=backtest_{run_id}.csv"},
    )


@router.get("/strategies/available")
async def list_available_strategies():
    from strategies import STRATEGY_REGISTRY
    return [
        {"id": key, "class": cls.__name__}
        for key, cls in STRATEGY_REGISTRY.items()
    ]


@router.delete("/{run_id}")
async def delete_backtest_run(run_id: int):
    async with async_session_factory() as session:
        run = await session.get(BacktestRun, run_id)
        if not run:
            raise HTTPException(status_code=404, detail="Backtest run not found")
        await session.delete(run)
        await session.commit()
    _progress_store.pop(run_id, None)
    return {"status": "deleted", "run_id": run_id}


@router.delete("/runs/all")
async def delete_all_backtest_runs():
    from sqlalchemy import delete as sa_delete
    async with async_session_factory() as session:
        await session.execute(sa_delete(BacktestRun))
        await session.commit()
    _progress_store.clear()
    return {"status": "all_deleted"}
