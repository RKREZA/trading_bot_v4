import logging
import requests
import os
from typing import Optional

logger = logging.getLogger("trading_bot.notifications")

class NotificationManager:
    def __init__(self, config: dict):
        self.config = config.get("notifications", {})
        self.enabled = self.config.get("enabled", False)
        self.telegram_token = os.getenv("TELEGRAM_BOT_TOKEN")
        self.chat_id = os.getenv("TELEGRAM_CHAT_ID")
        
    def send_telegram(self, message: str):
        """Send a message to the configured Telegram chat."""
        if not self.enabled or not self.telegram_token or not self.chat_id:
            return

        try:
            url = f"https://api.telegram.org/bot{self.telegram_token}/sendMessage"
            payload = {
                "chat_id": self.chat_id,
                "text": f"🤖 *TRADING BOT V3+*\n\n{message}",
                "parse_mode": "Markdown"
            }
            response = requests.post(url, json=payload, timeout=10)
            response.raise_for_status()
        except Exception as e:
            logger.error(f"Telegram Notification Error: {e}")

    def notify_critical(self, event: str, details: str):
        """For events that need immediate human attention."""
        msg = f"🚨 *CRITICAL ALERT*\n{event}\n\n{details}"
        self.send_telegram(msg)

    def notify_trade_open(self, symbol: str, direction: str, entry: float, lot: float, sl: float, tp: float):
        msg = (
            f"🚀 *TRADE OPENED*\n"
            f"Instrument: `{symbol}`\n"
            f"Direction: *{direction}*\n"
            f"Entry: `{entry:.5f}`\n"
            f"Lot: `{lot:.2f}`\n"
            f"SL: `{sl:.5f}`\n"
            f"TP: `{tp:.5f}`"
        )
        self.send_telegram(msg)

    def notify_trade_close(self, symbol: str, direction: str, exit_price: float, pnl: float, result: str):
        icon = "✅" if pnl > 0 else "❌"
        msg = (
            f"{icon} *TRADE CLOSED*\n"
            f"Instrument: `{symbol}`\n"
            f"Result: *{result}*\n"
            f"Exit: `{exit_price:.5f}`\n"
            f"PnL: `${pnl:.2f}`"
        )
        self.send_telegram(msg)
