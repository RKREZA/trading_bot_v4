"""
TRADING BOT V3 - MT5 Connection Manager
Handles connection lifecycle, health checks, and auto-reconnect.
"""

import logging
import os
import time
from datetime import datetime
from typing import Optional

try:
    import MetaTrader5 as mt5
except ImportError:
    mt5 = None

logger = logging.getLogger("trading_bot.connection")


class MT5Connection:
    """Manages the MT5 terminal connection with health checks and auto-reconnect."""

    def __init__(self, max_retries: int = 5, health_check_interval: int = 30):
        self.max_retries = max_retries
        self.health_check_interval = health_check_interval
        self.connected = False
        self.account_info: dict = {}
        self._last_health_check = 0.0

    def _get_credentials(self) -> dict:
        """Load MT5 credentials from environment variables."""
        login = os.environ.get("MT5_LOGIN")
        password = os.environ.get("MT5_PASSWORD")
        server = os.environ.get("MT5_SERVER")

        if not all([login, password, server]):
            logger.error(
                "Missing MT5 credentials. Set MT5_LOGIN, MT5_PASSWORD, MT5_SERVER "
                "environment variables or create a .env file."
            )
            return {}

        return {"login": int(login), "password": password, "server": server}

    def connect(self) -> bool:
        """
        Connect to MT5 terminal with retry logic.
        Returns True if connected successfully.
        """
        if mt5 is None:
            logger.error("MetaTrader5 package not installed. Run: pip install MetaTrader5")
            return False

        creds = self._get_credentials()
        if not creds:
            return False

        logger.info("=" * 50)
        logger.info("MT5 CONNECTION")
        logger.info("=" * 50)
        logger.info("Server: %s", creds["server"])
        logger.info("Login: %s", creds["login"])

        for attempt in range(self.max_retries):
            try:
                mt5.shutdown()
                time.sleep(2)
                logger.info("Attempt %d/%d...", attempt + 1, self.max_retries)

                if not mt5.initialize(
                    login=creds["login"],
                    password=creds["password"],
                    server=creds["server"],
                    timeout=30000,
                    portable=False,
                ):
                    error = mt5.last_error()
                    logger.warning("Failed: %s", error)
                    if error[0] == -10005:
                        logger.warning("MT5 terminal is not running. Open MT5 first!")
                    elif error[0] == -10011 or "Invalid" in str(error):
                        logger.warning("Invalid credentials. Check login/password/server!")
                    time.sleep(3)
                    continue

                info = mt5.account_info()
                if info is None:
                    logger.warning("Account info failed: %s", mt5.last_error())
                    mt5.shutdown()
                    time.sleep(3)
                    continue

                self.connected = True
                self._update_account_info(info)
                self._last_health_check = time.time()
                logger.info("Connected! Balance: $%s", f"{info.balance:,.2f}")
                return True

            except Exception as e:
                logger.exception("Connection exception: %s", e)
                time.sleep(3)

        logger.error("=" * 50)
        logger.error("CONNECTION FAILED after %d attempts", self.max_retries)
        logger.error("=" * 50)
        logger.error("TROUBLESHOOTING:")
        logger.error("1. Make sure MT5 terminal is OPEN and logged in")
        logger.error("2. Check if Algo Trading is enabled in MT5")
        logger.error("3. Run: python test_connection.py")
        return False

    def disconnect(self):
        """Cleanly disconnect from MT5."""
        if mt5 is not None:
            mt5.shutdown()
        self.connected = False
        logger.info("MT5 disconnected")

    def is_alive(self) -> bool:
        """
        Check if the connection is still alive.
        Only performs the actual MT5 check if enough time has passed since the last check.
        """
        if not self.connected:
            return False

        now = time.time()
        if now - self._last_health_check < self.health_check_interval:
            return True  # Assume alive between check intervals

        try:
            info = mt5.account_info()
            if info is None:
                logger.warning("Health check failed — connection lost")
                self.connected = False
                return False
            self._update_account_info(info)
            self._last_health_check = now
            return True
        except Exception as e:
            logger.warning("Health check exception: %s", e)
            self.connected = False
            return False

    def reconnect(self) -> bool:
        """Attempt to reconnect after a disconnection."""
        logger.info("Attempting reconnection...")
        self.connected = False
        return self.connect()

    def ensure_connected(self) -> bool:
        """Check health and reconnect if needed. Returns True if connected."""
        if self.is_alive():
            return True
        logger.warning("Connection lost — attempting reconnect")
        return self.reconnect()

    def _update_account_info(self, info):
        """Update cached account information."""
        self.account_info = {
            "login": info.login,
            "server": info.server,
            "balance": info.balance,
            "equity": info.equity,
            "profit": info.profit,
            "margin": info.margin,
            "free_margin": info.margin_free,
            "margin_level": info.margin_level if info.margin > 0 else 0,
            "positions": 0,
            "connected": True,
            "server_time": datetime.now().strftime("%H:%M:%S"),
        }

    def place_order(self, symbol: str, signal, lot_size: float) -> Optional[dict]:
        """
        Place an order in MT5 based on the provided signal.
        """
        if not self.ensure_connected():
            return None

        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            logger.error("%s not found, can not call order_check()", symbol)
            return None

        if not symbol_info.visible:
            logger.info("%s, is not visible, trying to switch on", symbol)
            if not mt5.symbol_select(symbol, True):
                logger.error("symbol_select({}}) failed, exit", symbol)
                return None

        # Determine order type
        if signal.direction == "BUY":
            order_type = mt5.ORDER_TYPE_BUY
            price = mt5.symbol_info_tick(symbol).ask
        elif signal.direction == "SELL":
            order_type = mt5.ORDER_TYPE_SELL
            price = mt5.symbol_info_tick(symbol).bid
        else:
            logger.error("Invalid signal direction: %s", signal.direction)
            return None

        # Prepare the request
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": float(lot_size),
            "type": order_type,
            "price": price,
            "sl": float(signal.stop_loss),
            "tp": float(signal.take_profit),
            "deviation": 20,
            "magic": 234000,
            "comment": "Bot V3",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        # Send order
        result = mt5.order_send(request)
        
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error("Order failed: %s, %s", result.retcode, result.comment)
            return None
            
        logger.info("Order placed successfully. Ticket: %s", result.order)
        return {
            "ticket": result.order,
            "volume": result.volume,
            "price": result.price
        }
