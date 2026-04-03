import logging
import requests
import os
import time
from typing import Optional

logger = logging.getLogger("trading_bot.notifications")

class NotificationManager:
    """
    Handles outbound notifications to Telegram and other channels.
    Provides structured methods for trade alerts, critical system errors, and performance updates.
    """
    def __init__(self, config: dict):
        """
        Initializes the notification manager.
        
        Args:
            config (dict): Global configuration (expects 'notifications' key).
        """
        self.config = config.get("notifications", {})
        self.enabled = self.config.get("enabled", False)
        self.telegram_token = os.getenv("TELEGRAM_BOT_TOKEN")
        self.chat_id = os.getenv("TELEGRAM_CHAT_ID")
        
    def send_telegram(self, message: str):
        """
        Sends a generic markdown-formatted message to the Telegram chat.
        
        Args:
            message (str): Text content of the message.
        """
        if not self.enabled or not self.telegram_token or not self.chat_id:
            return

        max_retries = 3
        backoff_factor = 2
        
        for attempt in range(max_retries):
            try:
                url = f"https://api.telegram.org/bot{self.telegram_token}/sendMessage"
                payload = {
                    "chat_id": self.chat_id,
                    "text": f"🤖 *TRADING BOT V3+*\n\n{message}",
                    "parse_mode": "Markdown"
                }
                response = requests.post(url, json=payload, timeout=10)
                response.raise_for_status()
                return # Success
            except Exception as e:
                if attempt < max_retries - 1:
                    sleep_time = backoff_factor ** attempt
                    logger.warning(f"Telegram attempt {attempt + 1} failed: {e}. Retrying in {sleep_time}s...")
                    time.sleep(sleep_time)
                else:
                    logger.error(f"Telegram Notification Error after {max_retries} attempts: {e}")

    def notify_critical(self, event: str, details: str):
        """
        Sends a high-priority alert for critical system events (e.g., connection loss, circuit breaker).
        
        Args:
            event (str): Short name of the event.
            details (str): Detailed explanation or error trace.
        """
        msg = f"🚨 *CRITICAL ALERT*\n{event}\n\n{details}"
        self.send_telegram(msg)

    def notify_trade_open(self, symbol: str, direction: str, entry: float, lot: float, sl: float, tp: float):
        """
        Formulates and sends an alert when a new trade is opened.
        
        Args:
            symbol (str): Trading instrument.
            direction (str): BUY or SELL.
            entry (float): Execution price.
            lot (float): Position size.
            sl (float): Stop Loss.
            tp (float): Take Profit.
        """
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
        """
        Formulates and sends an alert when a trade is closed.
        
        Args:
            symbol (str): Trading instrument.
            direction (str): BUY or SELL.
            exit_price (float): Closing price.
            pnl (float): Profit or Loss in base currency.
            result (str): Outcome description (e.g., 'TP' or 'SL').
        """
        icon = "✅" if pnl > 0 else "❌"
        msg = (
            f"{icon} *TRADE CLOSED*\n"
            f"Instrument: `{symbol}`\n"
            f"Result: *{result}*\n"
            f"Exit: `{exit_price:.5f}`\n"
            f"PnL: `${pnl:.2f}`"
        )
        self.send_telegram(msg)
