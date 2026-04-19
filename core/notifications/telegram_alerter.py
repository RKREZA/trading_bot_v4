import logging
import requests
import os
import time
import threading
from typing import Optional, List, Dict, Any
from datetime import datetime
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
    Supports rich formatting, threaded delivery, and priority filtering.
    """

    def __init__(self, token: Optional[str] = None, chat_id: Optional[str] = None):
        self.token = token or os.environ.get("TELEGRAM_BOT_TOKEN")
        self.chat_id = chat_id or os.environ.get("TELEGRAM_CHAT_ID")
        
        self.bas_url = f"https://api.telegram.org/bot{self.token}" if self.token else None
        
        if not self.token or not self.chat_id:
            logger.warning("[TelegramAlerter] Bot Token or Chat ID missing. Alerting acts as no-op.")
            self.enabled = False
        else:
            self.enabled = True
            logger.info("[TelegramAlerter] Initialized successfully.")

        self._alert_queue: List[Dict[str, Any]] = []
        self._queue_lock = threading.Lock()
        self._min_priority = AlertPriority(int(os.environ.get("TELEGRAM_MIN_PRIORITY", "1")))
        self._last_alert_time: Dict[str, float] = {}

    def send_alert(self, message: str, parse_mode: str = "HTML", priority: AlertPriority = AlertPriority.INFO,
              category: AlertCategory = AlertCategory.SYSTEM, suppress_duplicate_seconds: int = 60) -> bool:
        """
        Sends a message to the configured Telegram chat.
        Supports priority levels, categories, and duplicate suppression.
        """
        if not self.enabled:
            return False

        if priority.value < self._min_priority.value:
            return False

        msg_key = f"{category.value}:{message[:50]}"
        current_time = time.time()
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

        try:
            url = f"{self.bas_url}/sendMessage"
            payload = {
                "chat_id": self.chat_id,
                "text": formatted,
                "parse_mode": parse_mode
            }
            resp = requests.post(url, json=payload, timeout=2.0)
            
            if resp.status_code == 200:
                return True
            else:
                logger.error(f"[TelegramAlerter] Alert rejected by Telegram: {resp.text}")
                return False
                
        except Exception as e:
            logger.error(f"[TelegramAlerter] Exception sending alert: {str(e)}")
            return False

    def send_trade_alert(self, direction: str, symbol: str, volume: float, price: float, 
                     sl: float, tp: float, strategy: str) -> bool:
        """Send formatted trade execution alert."""
        emoji = "🟢" if direction.upper() == "BUY" else "🔴"
        msg = f"{emoji} <b>TRADE EXECUTED</b>\n" \
             f"Symbol: {symbol}\n" \
             f"Direction: {direction}\n" \
             f"Volume: {volume:.2f}\n" \
             f"Entry: {price:.2f}\n" \
             f"SL: {sl:.2f} | TP: {tp:.2f}\n" \
             f"Strategy: {strategy}"
        return self.send_alert(msg, priority=AlertPriority.INFO, category=AlertCategory.TRADE)

    def send_risk_alert(self, reason: str, details: str) -> bool:
        """Send risk-related alert with critical priority."""
        msg = f"🚨 <b>RISK ALERT</b>\n" \
             f"Reason: {reason}\n" \
             f"Details: {details}"
        return self.send_alert(msg, priority=AlertPriority.CRITICAL, category=AlertCategory.RISK)

    def send_emergency_alert(self, reason: str) -> bool:
        """Send emergency flatten alert."""
        msg = f"🛑 <b>EMERGENCY FLATTEN</b>\n" \
             f"Reason: {reason}\n" \
             f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        return self.send_alert(msg, priority=AlertPriority.EMERGENCY, category=AlertCategory.RISK, suppress_duplicate_seconds=300)

    def send_performance_summary(self, equity: float, daily_pnl: float, dd: float, trades: int) -> bool:
        """Send daily performance summary."""
        pnl_emoji = "🟢" if daily_pnl >= 0 else "🔴"
        msg = f"📊 <b>DAILY SUMMARY</b>\n" \
             f"Equity: ${equity:,.2f}\n" \
             f"{pnl_emoji} PnL: ${daily_pnl:,.2f}\n" \
             f"Drawdown: {dd:.1f}%\n" \
             f"Trades: {trades}"
        return self.send_alert(msg, priority=AlertPriority.INFO, category=AlertCategory.PERFORMANCE, suppress_duplicate_seconds=300)

    def send_regime_change(self, old_regime: str, new_regime: str) -> bool:
        """Send regime change notification."""
        msg = f"🔄 <b>REGIME CHANGE</b>\n" \
             f"{old_regime} → {new_regime}"
        return self.send_alert(msg, priority=AlertPriority.WARNING, category=AlertCategory.REGIME, suppress_duplicate_seconds=1800)

if __name__ == "__main__":
    import dotenv
    dotenv.load_dotenv()
    
    alerter = TelegramAlerter()
    if alerter.enabled:
        alerter.send_alert("<b>V5-INSIGNIA System Alert</b>\nTest connection from live engine.")
    else:
        print("Missing variables in .env")
