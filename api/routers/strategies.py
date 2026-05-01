from fastapi import APIRouter, Depends, HTTPException
from typing import Dict, Any, List

from api.dependencies import get_services, AppServices

router = APIRouter(prefix="/api/strategies", tags=["strategies"])


@router.get("/")
async def list_strategies(svc: AppServices = Depends(get_services)) -> List[Dict[str, Any]]:
    results = []
    for sid, strategy in svc.strategies.items():
        results.append({
            "strategy_id": sid,
            "enabled": strategy.enabled,
            "class": strategy.__class__.__name__,
            "thresholds": strategy.get_thresholds() if hasattr(strategy, "get_thresholds") else {},
        })
    return results


@router.post("/{strategy_id}/toggle")
async def toggle_strategy(strategy_id: str, svc: AppServices = Depends(get_services)):
    strategy = svc.strategies.get(strategy_id)
    if not strategy:
        raise HTTPException(status_code=404, detail=f"Strategy {strategy_id} not found")
    strategy.enabled = not strategy.enabled
    return {"strategy_id": strategy_id, "enabled": strategy.enabled}


@router.get("/{strategy_id}/metrics")
async def get_strategy_metrics(strategy_id: str, svc: AppServices = Depends(get_services)):
    strategy = svc.strategies.get(strategy_id)
    if not strategy:
        raise HTTPException(status_code=404, detail=f"Strategy {strategy_id} not found")
    return strategy.get_thresholds()


@router.put("/{strategy_id}/params")
async def update_strategy_params(
    strategy_id: str,
    params: Dict[str, Any],
    svc: AppServices = Depends(get_services),
):
    strategy = svc.strategies.get(strategy_id)
    if not strategy:
        raise HTTPException(status_code=404, detail=f"Strategy {strategy_id} not found")

    for key, value in params.items():
        if hasattr(strategy, key):
            setattr(strategy, key, value)

    return {"strategy_id": strategy_id, "updated": list(params.keys())}
