import logging
import os
import json
import numpy as np
from typing import Optional, Dict, Any

logger = logging.getLogger("trading_bot.cache")

try:
    import redis.asyncio as aioredis
except ImportError:
    aioredis = None


class RedisCache:
    """
    Async Redis cache for latest candle data per symbol+timeframe.
    Falls back to no-op if Redis is unavailable.
    """

    TTL_MAP = {
        "M1": 60,
        "M5": 120,
        "M15": 300,
        "H1": 900,
        "D1": 3600,
    }

    def __init__(self, redis_url: str = None):
        self._url = redis_url or os.getenv("REDIS_URL", "redis://localhost:6379/0")
        self._client: Optional[Any] = None
        self._available = False

    async def connect(self) -> bool:
        if aioredis is None:
            logger.warning("redis.asyncio not installed, cache disabled")
            return False
        try:
            self._client = aioredis.from_url(self._url, decode_responses=True)
            await self._client.ping()
            self._available = True
            logger.info(f"Redis cache connected: {self._url}")
            return True
        except Exception as e:
            logger.warning(f"Redis unavailable ({e}), cache disabled")
            self._available = False
            return False

    async def close(self) -> None:
        if self._client:
            await self._client.aclose()
            self._available = False

    def _key(self, symbol: str, timeframe: str) -> str:
        return f"candles:{symbol}:{timeframe}"

    async def get_candles(self, symbol: str, timeframe: str) -> Optional[Dict[str, Any]]:
        if not self._available:
            return None
        try:
            raw = await self._client.get(self._key(symbol, timeframe))
            if raw is None:
                return None
            return json.loads(raw)
        except Exception:
            return None

    async def set_candles(self, symbol: str, timeframe: str,
                          data: Dict[str, Any]) -> None:
        if not self._available:
            return
        try:
            ttl = self.TTL_MAP.get(timeframe, 300)
            payload = json.dumps(data, default=_json_default)
            await self._client.setex(self._key(symbol, timeframe), ttl, payload)
        except Exception as e:
            logger.debug(f"Cache write failed: {e}")

    async def get_tick(self, symbol: str) -> Optional[Dict[str, float]]:
        if not self._available:
            return None
        try:
            raw = await self._client.get(f"tick:{symbol}")
            return json.loads(raw) if raw else None
        except Exception:
            return None

    async def set_tick(self, symbol: str, tick_data: Dict[str, float]) -> None:
        if not self._available:
            return
        try:
            await self._client.setex(f"tick:{symbol}", 5, json.dumps(tick_data))
        except Exception:
            pass

    async def publish_event(self, channel: str, data: Dict[str, Any]) -> None:
        if not self._available:
            return
        try:
            await self._client.publish(channel, json.dumps(data, default=_json_default))
        except Exception:
            pass

    @property
    def available(self) -> bool:
        return self._available


def _json_default(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Not JSON serializable: {type(obj)}")


redis_cache = RedisCache()
