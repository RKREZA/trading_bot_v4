import os

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, List, Optional

from core.data.mt5_service import mt5_service
from core.observability.audit import audit_trail
from api.dependencies import get_services, AppServices

router = APIRouter(prefix="/api", tags=["trading"])


class ModifySLTP(BaseModel):
    sl: float
    tp: float
    symbol: str


class TradingConfig(BaseModel):
    assignments: Dict[str, List[str]]
    primarySymbol: str = ""
    pairOptions: Dict[str, Dict[str, Any]] = {}


class StartRequest(BaseModel):
    assignments: Optional[Dict[str, List[str]]] = None


async def _get_mt5_credentials(svc: AppServices) -> Dict[str, Any]:
    """Resolve MT5 credentials: active DB account -> config -> env vars."""
    from sqlalchemy import select
    from core.common.models import MTAccount
    from core.common.database import async_session_factory

    async with async_session_factory() as session:
        result = await session.execute(
            select(MTAccount).where(MTAccount.is_active == True).order_by(MTAccount.id).limit(1)
        )
        account = result.scalar_one_or_none()

    if account and account.login and account.server:
        creds = {"login": account.login, "server": account.server}
        if account.password:
            creds["password"] = account.password
        return creds

    mt5_cfg = svc.config.get("mt5", {})
    login = mt5_cfg.get("login") or os.getenv("MT5_LOGIN")
    password = mt5_cfg.get("password") or os.getenv("MT5_PASSWORD")
    server = mt5_cfg.get("server") or os.getenv("MT5_SERVER")

    creds = {}
    if login:
        creds["login"] = int(login)
    if password:
        creds["password"] = password
    if server:
        creds["server"] = server
    return creds


async def _load_trading_config() -> dict:
    from sqlalchemy import select
    from core.common.models import SystemState
    from core.common.database import async_session_factory

    async with async_session_factory() as session:
        result = await session.execute(select(SystemState).where(SystemState.key == "trading_config"))
        row = result.scalar_one_or_none()
    if row and row.value:
        return row.value
    return {"assignments": {}, "primarySymbol": ""}


async def _save_trading_config(config: dict) -> None:
    from sqlalchemy import select
    from core.common.models import SystemState
    from core.common.database import async_session_factory

    async with async_session_factory() as session:
        result = await session.execute(select(SystemState).where(SystemState.key == "trading_config"))
        row = result.scalar_one_or_none()
        if row:
            row.value = config
        else:
            session.add(SystemState(key="trading_config", value=config))
        await session.commit()


def _apply_assignments(svc: AppServices, assignments: Dict[str, List[str]], pair_options: Dict[str, Any] = None) -> None:
    svc.active_assignments = assignments
    svc.pair_options = pair_options or {}
    svc.config["pair_options"] = svc.pair_options
    for sid, strategy in svc.strategies.items():
        assigned_symbols = [sym for sym, strats in assignments.items() if sid in strats]
        strategy.config["symbols"] = assigned_symbols
        strategy.enabled = len(assigned_symbols) > 0


@router.get("/trading-config")
async def get_trading_config():
    return await _load_trading_config()


@router.put("/trading-config")
async def put_trading_config(body: TradingConfig):
    config = {
        "assignments": body.assignments,
        "primarySymbol": body.primarySymbol,
        "pairOptions": body.pairOptions,
    }
    await _save_trading_config(config)
    return config


@router.post("/control/start")
async def start_trading(body: Optional[StartRequest] = None, svc: AppServices = Depends(get_services)):
    if not mt5_service.connected:
        creds = await _get_mt5_credentials(svc)
        connected = await mt5_service.async_connect(**creds)
        if not connected:
            raise HTTPException(status_code=503, detail="Failed to connect to MT5")

    assignments = (body.assignments if body and body.assignments else None)
    pair_options = {}
    if not assignments:
        saved = await _load_trading_config()
        assignments = saved.get("assignments", {})
        pair_options = saved.get("pairOptions", {})
    else:
        saved = await _load_trading_config()
        pair_options = saved.get("pairOptions", {})

    valid = {sym: strats for sym, strats in assignments.items() if strats}
    if not valid:
        raise HTTPException(status_code=400, detail="No symbol-strategy assignments configured")

    _apply_assignments(svc, valid, pair_options)
    svc.is_trading = True

    audit_trail.log_system("Trading started", data={"assignments": valid})
    return {"status": "started", "assignments": valid}


@router.post("/control/stop")
async def stop_trading(svc: AppServices = Depends(get_services)):
    svc.is_trading = False
    svc.active_assignments = {}
    audit_trail.log_system("Trading stopped")
    return {"status": "stopped"}


@router.post("/control/kill")
async def kill_switch(svc: AppServices = Depends(get_services)):
    svc.is_trading = False
    if svc.risk_engine:
        svc.risk_engine.kill_switch_active = True
    audit_trail.log_system("KILL SWITCH activated", level="CRITICAL")
    return {"status": "killed", "message": "Kill switch activated. All trading halted."}


@router.post("/control/kill/reset")
async def reset_kill_switch(svc: AppServices = Depends(get_services)):
    if svc.risk_engine:
        svc.risk_engine.kill_switch_active = False
    audit_trail.log_system("Kill switch deactivated", level="WARNING")
    return {"status": "reset", "message": "Kill switch deactivated. Trading can resume."}


@router.get("/positions")
async def get_positions(svc: AppServices = Depends(get_services)) -> List[Dict[str, Any]]:
    if not mt5_service.connected:
        return []
    try:
        import MetaTrader5 as mt5
        positions = mt5.positions_get()
        if positions is None:
            return []
        return [p._asdict() for p in positions]
    except Exception:
        return []


@router.get("/account")
async def get_account_info(svc: AppServices = Depends(get_services)) -> Dict[str, Any]:
    if not mt5_service.connected:
        return {"connected": False, "balance": 0, "equity": 0, "margin": 0, "free_margin": 0, "profit": 0}
    try:
        import MetaTrader5 as mt5
        info = mt5.account_info()
        if info is None:
            return {"connected": True, "balance": 0, "equity": 0, "margin": 0, "free_margin": 0, "profit": 0}
        d = info._asdict()
        return {
            "connected": True,
            "login": d.get("login"),
            "server": d.get("server", ""),
            "name": d.get("name", ""),
            "currency": d.get("currency", "USD"),
            "balance": d.get("balance", 0),
            "equity": d.get("equity", 0),
            "margin": d.get("margin", 0),
            "free_margin": d.get("margin_free", 0),
            "margin_level": d.get("margin_level", 0),
            "profit": d.get("profit", 0),
            "leverage": d.get("leverage", 0),
        }
    except Exception:
        return {"connected": False, "balance": 0, "equity": 0, "margin": 0, "free_margin": 0, "profit": 0}


@router.get("/trades")
async def get_trades(
    limit: int = 50,
    strategy_id: str = None,
    svc: AppServices = Depends(get_services),
):
    from sqlalchemy import select, desc
    from core.common.models import Trade
    from core.common.database import async_session_factory

    async with async_session_factory() as session:
        query = select(Trade).order_by(desc(Trade.entry_time)).limit(limit)
        if strategy_id:
            query = query.where(Trade.strategy_id == strategy_id)
        result = await session.execute(query)
        trades = result.scalars().all()
        return [
            {
                "id": t.id,
                "symbol": t.symbol,
                "direction": t.direction,
                "entry_price": t.entry_price,
                "exit_price": t.exit_price,
                "volume": t.volume,
                "pnl": t.pnl,
                "status": t.status,
                "strategy_id": t.strategy_id,
                "entry_time": t.entry_time.isoformat() if t.entry_time else None,
                "exit_time": t.exit_time.isoformat() if t.exit_time else None,
            }
            for t in trades
        ]


@router.post("/mt5/connect")
async def connect_mt5(svc: AppServices = Depends(get_services)):
    if mt5_service.connected:
        return {"status": "already_connected"}
    creds = await _get_mt5_credentials(svc)
    connected = await mt5_service.async_connect(**creds)
    if not connected:
        raise HTTPException(status_code=503, detail="Failed to connect to MT5")
    audit_trail.log_system("MT5 connected via dashboard")
    return {"status": "connected"}


@router.post("/mt5/disconnect")
async def disconnect_mt5(svc: AppServices = Depends(get_services)):
    svc.is_trading = False
    await mt5_service.async_disconnect()
    audit_trail.log_system("MT5 disconnected via dashboard")
    return {"status": "disconnected"}


@router.post("/positions/{ticket}/close")
async def close_position(ticket: int):
    if not mt5_service.connected:
        raise HTTPException(status_code=503, detail="MT5 not connected")
    result = await mt5_service.async_close_position(ticket)
    if not result["success"]:
        raise HTTPException(status_code=400, detail=result.get("error", "Close failed"))
    return result


@router.put("/positions/{ticket}/modify")
async def modify_position(ticket: int, body: ModifySLTP):
    if not mt5_service.connected:
        raise HTTPException(status_code=503, detail="MT5 not connected")
    result = await mt5_service.async_modify_sl_tp(ticket, body.symbol, body.sl, body.tp)
    if not result["success"]:
        raise HTTPException(status_code=400, detail=result.get("error", "Modify failed"))
    return result
