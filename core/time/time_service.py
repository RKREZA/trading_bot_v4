from datetime import datetime, timezone, timedelta
import logging
from typing import Dict

logger = logging.getLogger("trading_bot.time_service")

class TimeService:
    """
    Unified Time Alignment Service.
    MT5 server time is the SOURCE OF TRUTH.
    Local system time MUST NOT be trusted for trading logic.
    """
    
    def __init__(self, broker_utc_offset_hours: int = 0):
        self.broker_utc_offset = timedelta(hours=broker_utc_offset_hours)
        self._last_server_time_utc: datetime = None
        self._last_local_time: datetime = None
        self.drift_warning_threshold_sec = 2.0
        self.drift_fatal_threshold_sec = 10.0
        
        # Cache for aligned times to avoid redundant calculations in same tick
        self._alignment_cache: Dict[str, datetime] = {}

    def update_server_time(self, server_timestamp: int):
        """
        Update the internal server time reference from an MT5 tick.
        server_timestamp: integer timestamp from MT5 (broker local time).
        """
        # Convert broker local timestamp to UTC aware datetime
        dt_broker = datetime.fromtimestamp(server_timestamp, tz=timezone.utc)
        new_server_time_utc = dt_broker - self.broker_utc_offset
        
        # Calculate drift before updating
        if self._last_server_time_utc:
            local_now = datetime.now(timezone.utc)
            # How much time passed locally since last update
            local_elapsed = (local_now - self._last_local_time).total_seconds()
            # How much time passed on server since last update
            server_elapsed = (new_server_time_utc - self._last_server_time_utc).total_seconds()
            
            # If server_elapsed is negative or zero (duplicate tick or late tick), we don't update last_local_time
            # to maintain a monotonic extrapolation.
            if server_elapsed > 0:
                self._last_server_time_utc = new_server_time_utc
                self._last_local_time = local_now
        else:
            self._last_server_time_utc = new_server_time_utc
            self._last_local_time = datetime.now(timezone.utc)
            
        self._alignment_cache.clear() # Clear cache on new time update
        self.check_drift()

    def get_server_time(self) -> datetime:
        """
        Returns the latest known MT5 server time, normalized to UTC.
        Extrapolates using system clock delta if necessary.
        """
        if not self._last_server_time_utc:
            logger.warning("TimeService: Server time not initialized. Using local time (UTC).")
            return datetime.now(timezone.utc)
            
        elapsed_since_update = datetime.now(timezone.utc) - self._last_local_time
        return self._last_server_time_utc + elapsed_since_update

    def to_utc(self, dt: datetime) -> datetime:
        """Ensures a datetime object is timezone-aware and set to UTC."""
        if dt.tzinfo is None:
            # If naive, we assume it's already UTC or broker-time based on context.
            # In this platform, naive datetimes from MT5 are treated as broker time.
            return dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)

    def align_to_timeframe(self, timestamp: datetime, timeframe_minutes: int) -> datetime:
        """
        Aligns a timestamp to the exact start of a timeframe candle.
        Supported timeframes: 1, 2, 3, 4, 5, 6, 10, 12, 15, 20, 30, 60 (1H), 240 (4H), 1440 (1D)
        """
        cache_key = f"{timestamp.isoformat()}_{timeframe_minutes}"
        if cache_key in self._alignment_cache:
            return self._alignment_cache[cache_key]

        ts = self.to_utc(timestamp)
        
        if timeframe_minutes < 1440:
            # Minutes based alignment
            total_minutes = ts.hour * 60 + ts.minute
            aligned_total_minutes = (total_minutes // timeframe_minutes) * timeframe_minutes
            
            new_hour = aligned_total_minutes // 60
            new_minute = aligned_total_minutes % 60
            
            aligned = ts.replace(hour=new_hour, minute=new_minute, second=0, microsecond=0)
        else:
            # Day based alignment
            aligned = ts.replace(hour=0, minute=0, second=0, microsecond=0)
            
        self._alignment_cache[cache_key] = aligned
        return aligned
        
    def check_drift(self):
        """Logs drift between system clock and broker clock."""
        if not self._last_server_time_utc:
            return
            
        local_utc = datetime.now(timezone.utc)
        drift = (local_utc - self._last_server_time_utc).total_seconds()
        abs_drift = abs(drift)
        
        if abs_drift > self.drift_fatal_threshold_sec:
            logger.critical(f"CLOCK DRIFT FATAL: {drift:.3f}s. System is out of sync.")
        elif abs_drift > self.drift_warning_threshold_sec:
            logger.warning(f"CLOCK DRIFT WARNING: {drift:.3f}s detected.")
            
    def set_broker_offset(self, hours: int):
        self.broker_utc_offset = timedelta(hours=hours)
        logger.info(f"TimeService: Broker UTC offset set to {hours} hours.")

# Singleton instance
time_service = TimeService()
