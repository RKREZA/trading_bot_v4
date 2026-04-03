"""
TRADING BOT V3 — Broker Clock
===============================
Authoritative time source derived from the MT5 broker server.

Instead of relying on the local machine clock (which may be wrong or in
a different timezone), this module reads the latest tick timestamp from
the MT5 terminal and computes a UTC offset.

All time-sensitive decisions in the bot (session detection, daily reset,
news filtering, circuit breakers) MUST use this clock.

Usage:
    clock = BrokerClock()
    clock.sync("XAUUSDm")          # Call once per cycle
    now   = clock.now()             # → datetime (UTC)
    today = clock.today()           # → date (UTC)
    hour  = clock.hour()            # → int (0-23 UTC)
    ts    = clock.timestamp()       # → float (Unix epoch)
"""

import logging
import threading
import time
from datetime import datetime, date, timezone, timedelta
from typing import Optional

logger = logging.getLogger("trading_bot.clock")

try:
    import MetaTrader5 as mt5
except ImportError:
    mt5 = None


class BrokerClock:
    """
    Derives authoritative UTC time from the MT5 broker's tick stream.

    Design:
    -------
    1. On each `sync()` call, fetch the latest tick for the active symbol.
    2. Compute `_offset = tick_server_time - time.time()`.
    3. All subsequent `now()`/`today()`/`hour()` calls apply this offset
       to the local monotonic clock — zero MT5 calls per query.
    4. Thread-safe: offset updates are guarded by a lock.

    Fallback:
    ---------
    If MT5 is unavailable (e.g. during backtesting or disconnect), falls
    back to `datetime.now(timezone.utc)`. A warning is logged once.
    """

    def __init__(self):
        self._offset: float = 0.0          # seconds: broker_utc - local_time
        self._synced: bool = False
        self._lock = threading.Lock()
        self._last_sync: float = 0.0
        self._fallback_warned: bool = False
        self._sync_count: int = 0

    def sync(self, symbol: str, mt5_lock: Optional[threading.Lock] = None) -> bool:
        """
        Refresh the internal time offset from the latest MT5 tick.

        Should be called once at the top of each trading cycle (every 5s).
        Uses the tick's `time_msc` field (ms precision) which is the broker
        server's UTC epoch.

        STALENESS GUARD: If the tick is older than 5 minutes, the market is
        likely closed. In that case we fall back to local UTC to avoid
        computing a wildly wrong offset from a stale last-trade timestamp.

        Args:
            symbol: The active trading symbol (e.g. "XAUUSDm")
            mt5_lock: The global MT5_LOCK for thread-safe terminal access.

        Returns:
            True if sync succeeded, False if using fallback.
        """
        if mt5 is None:
            return self._use_fallback("MT5 not installed")

        try:
            if mt5_lock:
                with mt5_lock:
                    tick = mt5.symbol_info_tick(symbol)
            else:
                tick = mt5.symbol_info_tick(symbol)

            if tick is None:
                return self._use_fallback("No tick data")

            # Use time_msc (millisecond precision) if available, else time
            if hasattr(tick, 'time_msc') and tick.time_msc > 0:
                broker_utc_epoch = float(tick.time_msc) / 1000.0
            else:
                broker_utc_epoch = float(tick.time)

            local_epoch = time.time()

            # STALENESS CHECK: if tick is older than 5 minutes, market is 
            # likely closed. Don't use a stale tick for offset computation.
            tick_age_seconds = local_epoch - broker_utc_epoch
            if tick_age_seconds > 300:  # 5 minutes
                return self._use_fallback(
                    f"Stale tick ({tick_age_seconds/3600:.1f}h old) — market likely closed"
                )

            new_offset = broker_utc_epoch - local_epoch

            with self._lock:
                self._offset = new_offset
                self._synced = True
                self._last_sync = local_epoch
                self._sync_count += 1
                # Reset fallback warning so we re-log if we go stale again
                self._fallback_warned = False

            # Log first successful sync and periodically after
            if self._sync_count == 1 or self._sync_count % 1000 == 0:
                broker_dt = datetime.fromtimestamp(broker_utc_epoch, tz=timezone.utc)
                local_dt = datetime.now(timezone.utc)
                logger.info(
                    "BrokerClock synced: broker=%s local=%s offset=%.2fs",
                    broker_dt.strftime("%I:%M:%S %p"),
                    local_dt.strftime("%I:%M:%S %p"),
                    self._offset,
                )
            return True

        except Exception as e:
            return self._use_fallback(f"Sync error: {e}")

    def _use_fallback(self, reason: str) -> bool:
        """Fall back to local UTC clock with a one-time warning."""
        if not self._fallback_warned:
            logger.warning(
                "BrokerClock fallback to local UTC (%s). "
                "Time-sensitive decisions may be inaccurate.",
                reason,
            )
            self._fallback_warned = True
        # Keep offset at 0 → effectively local UTC
        with self._lock:
            self._offset = 0.0
        return False

    def now(self) -> datetime:
        """Current broker time as a timezone-aware UTC datetime."""
        with self._lock:
            offset = self._offset
        epoch = time.time() + offset
        return datetime.fromtimestamp(epoch, tz=timezone.utc)

    def today(self) -> date:
        """Current broker date in UTC."""
        return self.now().date()

    def hour(self) -> int:
        """Current broker hour (0-23, UTC)."""
        return self.now().hour

    def minute(self) -> int:
        """Current broker minute (0-59, UTC)."""
        return self.now().minute

    def timestamp(self) -> float:
        """Current broker time as Unix epoch (float)."""
        with self._lock:
            return time.time() + self._offset

    @property
    def is_synced(self) -> bool:
        """True if at least one successful sync has occurred."""
        with self._lock:
            return self._synced

    @property
    def offset_seconds(self) -> float:
        """Current offset in seconds (broker - local)."""
        with self._lock:
            return self._offset

    def format_time(self, fmt: str = "%H:%M:%S") -> str:
        """Formatted broker time string."""
        return self.now().strftime(fmt)

    def format_datetime(self, fmt: str = "%Y-%m-%d %H:%M:%S") -> str:
        """Formatted broker datetime string."""
        return self.now().strftime(fmt)

    def __repr__(self) -> str:
        return (
            f"<BrokerClock synced={self.is_synced} "
            f"offset={self.offset_seconds:+.1f}s "
            f"now={self.format_time()}>"
        )
