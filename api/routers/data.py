import os
import asyncio
import logging

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from typing import Dict, Any, List

from api.dependencies import get_services, AppServices
from core.data.mt5_service import mt5_service

logger = logging.getLogger("trading_bot.api.data")

router = APIRouter(prefix="/api/data", tags=["data"])


@router.get("/symbols")
async def get_symbols(svc: AppServices = Depends(get_services)) -> List[str]:
    symbols = set()

    symbols_dir = "config/symbols"
    if os.path.isdir(symbols_dir):
        for f in os.listdir(symbols_dir):
            if f.endswith(".json"):
                symbols.add(f.replace(".json", ""))

    cfg_symbols = svc.config.get("symbols", [])
    if isinstance(cfg_symbols, list):
        symbols.update(cfg_symbols)

    symbols_config = svc.config.get("symbols_config", {})
    if isinstance(symbols_config, dict):
        symbols.update(symbols_config.keys())

    if mt5_service.connected:
        try:
            import MetaTrader5 as mt5
            mt5_symbols = await asyncio.to_thread(mt5.symbols_get)
            if mt5_symbols:
                for s in mt5_symbols:
                    if s.visible:
                        symbols.add(s.name)
        except Exception:
            pass

    return sorted(symbols)


@router.get("/{symbol}/{timeframe}")
async def get_candle_data(
    symbol: str,
    timeframe: str,
    count: int = 200,
    svc: AppServices = Depends(get_services),
):
    source = svc.data_manager.source
    candles = await source.async_fetch_candles(symbol, timeframe, count)

    if candles is None or len(candles) == 0:
        raise HTTPException(status_code=404, detail=f"No data for {symbol} {timeframe}")

    limit = min(count, len(candles))
    return {
        "symbol": symbol,
        "timeframe": timeframe,
        "count": limit,
        "candles": {
            "time": candles.time[-limit:].tolist(),
            "open": candles.open[-limit:].tolist(),
            "high": candles.high[-limit:].tolist(),
            "low": candles.low[-limit:].tolist(),
            "close": candles.close[-limit:].tolist(),
            "volume": candles.tick_volume[-limit:].tolist(),
        },
    }


@router.post("/sync")
async def trigger_sync(
    body: Dict[str, Any],
    svc: AppServices = Depends(get_services),
):
    symbol = body.get("symbol")
    timeframe = body.get("timeframe", "M5")

    if not symbol:
        raise HTTPException(status_code=400, detail="symbol is required")

    try:
        svc.data_manager.sync.update_incremental(symbol, timeframe)
        return {"status": "ok", "symbol": symbol, "timeframe": timeframe}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/sync/all")
async def sync_all_timeframes(
    body: Dict[str, Any],
    background_tasks: BackgroundTasks,
    svc: AppServices = Depends(get_services),
):
    symbol = body.get("symbol")
    if not symbol:
        raise HTTPException(status_code=400, detail="symbol is required")

    timeframes = ["M1", "M5", "M15", "H1", "H4", "D1"]

    def _sync_all():
        for tf in timeframes:
            try:
                svc.data_manager.sync.update_incremental(symbol, tf)
                logger.info(f"Synced {symbol} {tf}")
            except Exception as e:
                logger.warning(f"Sync failed {symbol} {tf}: {e}")

    background_tasks.add_task(asyncio.to_thread, _sync_all)
    return {"status": "syncing", "symbol": symbol, "timeframes": timeframes}


@router.get("/sync/status")
async def sync_status(svc: AppServices = Depends(get_services)):
    store = svc.data_manager.store
    symbols = svc.config.get("symbols", ["XAUUSDm"])
    timeframes = ["M1", "M5", "M15", "H1", "D1"]

    status = {}
    for sym in symbols:
        status[sym] = {}
        for tf in timeframes:
            last_ts = store.get_last_timestamp(sym, tf)
            status[sym][tf] = {
                "last_timestamp": last_ts,
                "has_data": last_ts > 0,
            }

    return status
