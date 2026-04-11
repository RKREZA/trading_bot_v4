import logging
import requests
import os
from typing import Optional

logger = logging.getLogger("trading_bot.notifications")

class TelegramAlerter:
    """
    Institutional Telemetry & Alerting Broker.
    Sends real-time high-priority alerts to a Telegram chat.
    Safe-fails if no token is configured to prevent blocking the main thread.
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

    def send_alert(self, message: str, parse_mode: str = "HTML") -> bool:
        """
        Sends a message to the configured Telegram chat.
        Synchronous, but with a fast timeout (2s) to prevent blocking MT5 logic.
        """
        if not self.enabled:
            return False
            
        try:
            url = f"{self.bas_url}/sendMessage"
            payload = {
                "chat_id": self.chat_id,
                "text": message,
                "parse_mode": parse_mode
            }
            # Short timeout is critical for institutional execution paths.
            resp = requests.post(url, json=payload, timeout=2.0)
            
            if resp.status_code == 200:
                return True
            else:
                logger.error(f"[TelegramAlerter] Alert rejected by Telegram: {resp.text}")
                return False
                
        except Exception as e:
            logger.error(f"[TelegramAlerter] Exception sending alert: {str(e)}")
            return False

# Quick test footprint
if __name__ == "__main__":
    import dotenv
    dotenv.load_dotenv()
    
    alerter = TelegramAlerter()
    if alerter.enabled:
        alerter.send_alert("<b>V5-INSIGNIA System Alert</b>\nTest connection from live engine.")
    else:
        print("Missing variables in .env")
