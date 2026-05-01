from fastapi import APIRouter, Depends
from datetime import datetime, timezone

from core.time.time_service import time_service
from core.data.mt5_service import mt5_service
from core.observability.metrics import metrics
from api.dependencies import get_services, AppServices

router = APIRouter(prefix="/api", tags=["status"])


@router.get("/health")
async def health():
    return {"status": "ok", "timestamp": datetime.now(timezone.utc).isoformat()}


@router.get("/status")
async def get_status(svc: AppServices = Depends(get_services)):
    server_time = time_service.get_server_time()
    local_time = datetime.now(timezone.utc)
    drift = (local_time - server_time).total_seconds()

    account = svc.recon_engine.get_account_summary()

    return {
        "is_trading": svc.is_trading,
        "mt5_connected": mt5_service.connected,
        "server_time": server_time.isoformat(),
        "local_time": local_time.isoformat(),
        "drift_sec": round(drift, 3),
        "active_strategies": list(svc.strategies.keys()),
        "kill_switch": svc.risk_engine.kill_switch_active if svc.risk_engine else False,
        "account": account,
    }


@router.get("/metrics")
async def get_metrics():
    return metrics.get_snapshot()
