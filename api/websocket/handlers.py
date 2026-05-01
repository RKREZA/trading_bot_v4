import asyncio
import json
import logging
from datetime import datetime, timezone

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from api.websocket.manager import ws_manager
from api.dependencies import get_services

logger = logging.getLogger("trading_bot.ws")

router = APIRouter()


@router.websocket("/ws/events")
async def websocket_events(websocket: WebSocket):
    client_id = await ws_manager.connect(websocket)
    svc = get_services()

    try:
        receive_task = asyncio.create_task(_handle_incoming(client_id, websocket))
        push_task = asyncio.create_task(_push_metrics(client_id, websocket, svc))

        done, pending = await asyncio.wait(
            {receive_task, push_task},
            return_when=asyncio.FIRST_COMPLETED,
        )

        for task in pending:
            task.cancel()

    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.error(f"WebSocket error ({client_id}): {e}")
    finally:
        ws_manager.disconnect(client_id)


async def _handle_incoming(client_id: str, websocket: WebSocket):
    try:
        while True:
            raw = await websocket.receive_text()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                continue

            action = msg.get("action")
            if action == "subscribe":
                event_types = msg.get("events", [])
                ws_manager.subscribe(client_id, event_types)
                await ws_manager.send_to(client_id, {
                    "type": "SUBSCRIBED",
                    "events": list(ws_manager._subscriptions.get(client_id, set())),
                })
            elif action == "unsubscribe":
                event_types = msg.get("events", [])
                ws_manager.unsubscribe(client_id, event_types)

    except WebSocketDisconnect:
        raise
    except Exception:
        pass


async def _push_metrics(client_id: str, websocket: WebSocket, svc):
    while True:
        try:
            account = svc.recon_engine.get_account_summary() if svc.recon_engine else {}
            await ws_manager.broadcast("METRICS", {
                "account": account,
                "is_trading": svc.is_trading,
                "active_strategies": len(svc.strategies),
                "ws_clients": ws_manager.active_count,
            })
        except Exception as e:
            logger.debug(f"Metrics push error: {e}")
            break

        await asyncio.sleep(2)
