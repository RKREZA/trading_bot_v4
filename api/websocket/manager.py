import asyncio
import logging
import json
from typing import Dict, List, Set
from datetime import datetime, timezone

from fastapi import WebSocket, WebSocketDisconnect

logger = logging.getLogger("trading_bot.ws")


class ConnectionManager:
    """Multi-client WebSocket manager with typed event subscriptions."""

    def __init__(self):
        self._connections: Dict[str, WebSocket] = {}
        self._subscriptions: Dict[str, Set[str]] = {}
        self._counter = 0

    async def connect(self, websocket: WebSocket) -> str:
        await websocket.accept()
        self._counter += 1
        client_id = f"ws_{self._counter}"
        self._connections[client_id] = websocket
        self._subscriptions[client_id] = {"METRICS", "SYSTEM_STATUS"}
        logger.info(f"WebSocket connected: {client_id}")
        return client_id

    def disconnect(self, client_id: str) -> None:
        self._connections.pop(client_id, None)
        self._subscriptions.pop(client_id, None)
        logger.info(f"WebSocket disconnected: {client_id}")

    def subscribe(self, client_id: str, event_types: List[str]) -> None:
        if client_id in self._subscriptions:
            self._subscriptions[client_id].update(event_types)

    def unsubscribe(self, client_id: str, event_types: List[str]) -> None:
        if client_id in self._subscriptions:
            self._subscriptions[client_id] -= set(event_types)

    async def broadcast(self, event_type: str, data: dict) -> None:
        payload = json.dumps({
            "type": event_type,
            "data": data,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }, default=str)

        dead = []
        for client_id, ws in self._connections.items():
            subs = self._subscriptions.get(client_id, set())
            if event_type not in subs and "*" not in subs:
                continue
            try:
                await ws.send_text(payload)
            except Exception:
                dead.append(client_id)

        for cid in dead:
            self.disconnect(cid)

    async def send_to(self, client_id: str, data: dict) -> None:
        ws = self._connections.get(client_id)
        if ws:
            try:
                await ws.send_json(data)
            except Exception:
                self.disconnect(client_id)

    @property
    def active_count(self) -> int:
        return len(self._connections)


ws_manager = ConnectionManager()
