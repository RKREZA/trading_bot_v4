from fastapi import APIRouter, Depends
from typing import Dict, Any

from api.dependencies import get_services, AppServices

router = APIRouter(prefix="/api/config", tags=["config"])


@router.get("/")
async def get_config(svc: AppServices = Depends(get_services)) -> Dict[str, Any]:
    safe_config = dict(svc.config)
    for key in ("mt5_password", "password", "secret"):
        safe_config.pop(key, None)
    return safe_config


@router.post("/")
async def update_config(
    updates: Dict[str, Any],
    svc: AppServices = Depends(get_services),
):
    svc.config.update(updates)
    return {"status": "updated", "keys": list(updates.keys())}
