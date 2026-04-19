import logging
import queue
import requests
import os
import time
import threading
from typing import Optional, List, Dict, Any
from datetime import datetime, timezone
from enum import Enum

logger = logging.getLogger("trading_bot.notifications")

class AlertPriority(Enum):
    DEBUG = 1
    INFO = 2
    WARNING = 3
    CRITICAL = 4
    EMERGENCY = 5

class AlertCategory(Enum):
    SYSTEM = "SYSTEM"
    RISK = "RISK"
    TRADE = "TRADE"
    REGIME = "REGIME"
    PERFORMANCE = "PERFORMANCE"
    HEALTH = "HEALTH"

class TelegramAlerter:
    """
    Institutional Telemetry & Alerting Broker.
    Sends real-time high-priority alerts to a Telegram chat with async queuing.
    Safe-fails if no token is configured to prevent blocking the main thread.
    Supports rich formatting, threaded delivery, priority filtering, and HTTP retry.

    Architecture:
    - All `send_*` calls are NON-BLOCKING: they enqueue a message and return immediately.
    - A single background daemon thread drains the queue and POSTs to Telegram.
    - Failed requests are retried up to 3 times with exponential back-off.
    - A second daemon thread fires a daily summary at 23:00 UTC if registered.
    """

    def __init__(self, token: Optional[str] = None, chat_id: Optional[str] = None):
        self.token = token or os.environ.get("TELEGRAM_BOT_TOKEN")
        self.chat_id = chat_id or os.environ.get("TELEGRAM_CHAT_ID")

        # Corrected from 'bas_url' typo
        self.base_url = f"https://api.telegram.org/bot{self.token}" if self.token else None

        if not self.token or not self.chat_id:
            logger.warning("[TelegramAlerter] Bot Token or Chat ID missing. Alerting acts as no-op.")
            self.enabled = False
        else:
            self.enabled = True
            logger.info("[TelegramAlerter] Initialized successfully.")

        # Thread-safe FIFO queue replacing the old dead list
        self._alert_queue: queue.Queue = queue.Queue(maxsize=500)
        self._queue_lock = threading.Lock()
        self._min_priority = AlertPriority(int(os.environ.get("TELEGRAM_MIN_PRIORITY", "1")))
        self._last_alert_time: Dict[str, float] = {}

        # Daily summary callback (registered by calling start_daily_summary_scheduler)
        self._daily_summary_callback = None

        # Start async consumer daemon
        if self.enabled:
            self._consumer_thread = threading.Thread(
                target=self._queue_consumer, daemon=True, name="TelegramAlerterConsumer"
            )
            self._consumer_thread.start()
            logger.info("[TelegramAlerter] Async consumer thread started.")

    # ──────────────────────────────────────────────────────────────────────────
    # ASYNC CONSUMER
    # ──────────────────────────────────────────────────────────────────────────

    def _queue_consumer(self):
        """Background daemon: drains the alert queue and delivers to Telegram."""
        while True:
            try:
                payload = self._alert_queue.get(timeout=1.0)
                self._deliver_with_retry(payload)
                self._alert_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"[TelegramAlerter] Consumer error: {e}")
                time.sleep(1.0)

    def _deliver_with_retry(self, payload: dict, max_retries: int = 3):
        """POST to Telegram with up to 3 attempts and exponential back-off."""
        if not self.enabled or not self.base_url:
            return
        url = f"{self.base_url}/sendMessage"
        backoff = 1.0
        for attempt in range(1, max_retries + 1):
            try:
                resp = requests.post(url, json=payload, timeout=5.0)
                if resp.status_code == 200:
                    return
                logger.warning(
                    f"[TelegramAlerter] Attempt {attempt}/{max_retries} failed "
                    f"(HTTP {resp.status_code}): {resp.text[:200]}"
                )
            except Exception as e:
                logger.warning(f"[TelegramAlerter] Attempt {attempt}/{max_retries} exception: {e}")
            if attempt < max_retries:
                time.sleep(backoff)
                backoff *= 2.0
        logger.error("[TelegramAlerter] All retry attempts exhausted. Message dropped.")

    # ──────────────────────────────────────────────────────────────────────────
    # PUBLIC API
    # ──────────────────────────────────────────────────────────────────────────

    def send_alert(self, message: str, parse_mode: str = "HTML",
                   priority: AlertPriority = AlertPriority.INFO,
                   category: AlertCategory = AlertCategory.SYSTEM,
                   suppress_duplicate_seconds: int = 60) -> bool:
        """
        Enqueues a message for async delivery to the configured Telegram chat.
        Returns True immediately if successfully queued; False if filtered/disabled.
        The main thread is NEVER blocked by network I/O.
        """
        if not self.enabled:
            return False

        if priority.value < self._min_priority.value:
            return False

        # Duplicate suppression
        msg_key = f"{category.value}:{message[:50]}"
        current_time = time.time()
        with self._queue_lock:
            if msg_key in self._last_alert_time:
                if current_time - self._last_alert_time[msg_key] < suppress_duplicate_seconds:
                    return False
            self._last_alert_time[msg_key] = current_time

        priority_emoji = {
            AlertPriority.DEBUG: "🔍",
            AlertPriority.INFO: "ℹ️",
            AlertPriority.WARNING: "⚠️",
            AlertPriority.CRITICAL: "🚨",
            AlertPriority.EMERGENCY: "🛑"
        }
        formatted = f"{priority_emoji.get(priority, 'ℹ️')} <b>[{category.value}]</b>\n{message}"

        payload = {
            "chat_id": self.chat_id,
            "text": formatted,
            "parse_mode": parse_mode
        }

        try:
            self._alert_queue.put_nowait(payload)
            return True
        except queue.Full:
            logger.warning("[TelegramAlerter] Alert queue full. Message dropped.")
            return False

    def send_trade_alert(self, direction: str, symbol: str, volume: float, price: float,
                     sl: float, tp: float, strategy: str) -> bool:
        """Send formatted trade execution alert."""
        emoji = "🟢" if direction.upper() == "BUY" else "🔴"
        msg = (f"{emoji} <b>TRADE EXECUTED</b>\n"
               f"Symbol: {symbol}\n"
               f"Direction: {direction}\n"
               f"Volume: {volume:.2f}\n"
               f"Entry: {price:.2f}\n"
               f"SL: {sl:.2f} | TP: {tp:.2f}\n"
               f"Strategy: {strategy}")
        return self.send_alert(msg, priority=AlertPriority.INFO, category=AlertCategory.TRADE)

    def send_risk_alert(self, reason: str, details: str) -> bool:
        """Send risk-related alert with critical priority."""
        msg = (f"🚨 <b>RISK ALERT</b>\n"
               f"Reason: {reason}\n"
               f"Details: {details}")
        return self.send_alert(msg, priority=AlertPriority.CRITICAL, category=AlertCategory.RISK)

    def send_emergency_alert(self, reason: str) -> bool:
        """Send emergency flatten alert."""
        msg = (f"🛑 <b>EMERGENCY FLATTEN</b>\n"
               f"Reason: {reason}\n"
               f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        return self.send_alert(
            msg, priority=AlertPriority.EMERGENCY,
            category=AlertCategory.RISK, suppress_duplicate_seconds=300
        )

    def send_performance_summary(self, equity: float, daily_pnl: float,
                                  dd: float, trades: int) -> bool:
        """Send daily performance summary."""
        pnl_emoji = "🟢" if daily_pnl >= 0 else "🔴"
        msg = (f"📊 <b>DAILY SUMMARY</b>\n"
               f"Equity: ${equity:,.2f}\n"
               f"{pnl_emoji} PnL: ${daily_pnl:,.2f}\n"
               f"Drawdown: {dd:.1f}%\n"
               f"Trades: {trades}")
        return self.send_alert(
            msg, priority=AlertPriority.INFO,
            category=AlertCategory.PERFORMANCE, suppress_duplicate_seconds=300
        )

    def send_regime_change(self, old_regime: str, new_regime: str) -> bool:
        """Send regime change notification."""
        msg = f"🔄 <b>REGIME CHANGE</b>\n{old_regime} → {new_regime}"
        return self.send_alert(
            msg, priority=AlertPriority.WARNING,
            category=AlertCategory.REGIME, suppress_duplicate_seconds=1800
        )

    # ──────────────────────────────────────────────────────────────────────────
    # DAILY SUMMARY SCHEDULER
    # ──────────────────────────────────────────────────────────────────────────

    def start_daily_summary_scheduler(self, summary_callback):
        """
        Registers a callback that is invoked automatically at 23:00 UTC each day
        to send a daily performance summary.

        The callback must return a dict with keys: equity, daily_pnl, dd, trades.
        Example:
            alerter.start_daily_summary_scheduler(lambda: {
                "equity": 10000, "daily_pnl": 150, "dd": 1.5, "trades": 5
            })
        """
        self._daily_summary_callback = summary_callback
        t = threading.Thread(target=self._daily_summary_loop, daemon=True,
                             name="TelegramDailySummary")
        t.start()
        logger.info("[TelegramAlerter] Daily summary scheduler started (fires at 23:00 UTC).")

    def _daily_summary_loop(self):
        """Daemon thread: waits until 23:00 UTC, fires summary, then sleeps 24h."""
        while True:
            try:
                now = datetime.now(timezone.utc)
                target_hour = 23
                seconds_until_target = ((target_hour - now.hour) % 24) * 3600 - now.minute * 60 - now.second
                if seconds_until_target <= 0:
                    seconds_until_target += 86400  # Fire tomorrow
                time.sleep(seconds_until_target)

                if self._daily_summary_callback:
                    data = self._daily_summary_callback()
                    if isinstance(data, dict):
                        self.send_performance_summary(
                            equity=data.get("equity", 0),
                            daily_pnl=data.get("daily_pnl", 0),
                            dd=data.get("dd", 0),
                            trades=data.get("trades", 0),
                        )
            except Exception as e:
                logger.error(f"[TelegramAlerter] Daily summary error: {e}")
                time.sleep(3600)  # Retry in 1 hour on failure


if __name__ == "__main__":
    import dotenv
    dotenv.load_dotenv()

    alerter = TelegramAlerter()
    if alerter.enabled:
        alerter.send_alert("<b>V5-INSIGNIA System Alert</b>\nTest connection from live engine.")
        time.sleep(3)  # Allow async queue to drain
    else:
        print("Missing variables in .env")
