"""
TRADING BOT V3 - MT5 Connection Manager
Handles connection lifecycle, health checks, and auto-reconnect.
"""

import logging
import os
import time
from datetime import datetime
from typing import Optional, List

try:
    import MetaTrader5 as mt5
except ImportError:
    mt5 = None

logger = logging.getLogger("trading_bot.connection")

# Magic number identifying orders placed by this bot in MT5.
# Override in config.json under "magic_number" for multi-instance setups.
BOT_MAGIC_NUMBER = 234000


class MT5Connection:
    """Manages the MT5 terminal connection with health checks and auto-reconnect."""

    def __init__(self, max_retries: int = 5, health_check_interval: int = 30):
        self.max_retries = max_retries
        self.health_check_interval = health_check_interval
        self.connected = False
        self.account_info: dict = {}
        self._last_health_check = 0.0
        self.config = {}  # Will be set from main

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
        """Connect to MT5 terminal with retry logic."""
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
        return False

    def disconnect(self):
        """Cleanly disconnect from MT5."""
        if mt5 is not None:
            mt5.shutdown()
        self.connected = False
        logger.info("MT5 disconnected")

    def is_alive(self) -> bool:
        """Check if the connection is still alive."""
        if not self.connected:
            return False

        now = time.time()
        if now - self._last_health_check < self.health_check_interval:
            return True

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
        # Use MT5 server time if available, fall back to local time
        try:
            terminal = mt5.terminal_info()
            server_time = datetime.fromtimestamp(terminal.data_center if terminal else 0).strftime("%H:%M:%S")
        except Exception:
            server_time = datetime.now().strftime("%H:%M:%S")

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
            "server_time": server_time,
        }

    def place_order(self, symbol: str, signal, lot_size: float, max_retries: int = 3, delay: float = 1.0) -> Optional[dict]:
        """
        Place an order in MT5 based on the provided signal, with retry logic.
        """
        if not self.ensure_connected():
            return None

        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            logger.error("%s not found, can not call order_check()", symbol)
            return None

        if not symbol_info.visible:
            logger.info("%s is not visible, trying to switch on", symbol)
            if not mt5.symbol_select(symbol, True):
                logger.error("symbol_select(%s) failed, exit", symbol)
                return None

        # Determine order type and price
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            logger.error("Cannot get tick for %s", symbol)
            return None

        if signal.direction == "BUY":
            order_type = mt5.ORDER_TYPE_BUY
            price = tick.ask
        else:
            order_type = mt5.ORDER_TYPE_SELL
            price = tick.bid

        # Get deviation from config
        deviation = self.config.get("symbols_config", {}).get(symbol, {}).get("deviation", 20)
        magic = self.config.get("magic_number", BOT_MAGIC_NUMBER)

        for attempt in range(max_retries):
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": float(lot_size),
                "type": order_type,
                "price": price,
                "sl": float(signal.stop_loss),
                "tp": float(signal.take_profit),
                "deviation": deviation,
                "magic": magic,
                "comment": "Bot V3",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_RETURN,
            }

            result = mt5.order_send(request)
            if result.retcode == mt5.TRADE_RETCODE_DONE:
                logger.info("Order placed successfully. Ticket: %s", result.order)
                return {"ticket": result.order, "volume": result.volume, "price": result.price}

            logger.warning("Order attempt %d/%d failed: retcode=%s, comment=%s",
                           attempt+1, max_retries, result.retcode, result.comment)
            if attempt < max_retries - 1:
                time.sleep(delay)

        return None

    def get_pending_orders(self, symbol: str = None) -> List:
        """Return list of pending orders, optionally filtered by symbol."""
        if not self.ensure_connected():
            return []
        try:
            orders = mt5.orders_get(symbol=symbol)
            return orders if orders else []
        except Exception as e:
            logger.error("Error fetching pending orders: %s", e)
            return []


class PositionManager:
    """Manages open positions and risk-based lot sizing."""

    def __init__(self, connection: MT5Connection):
        self.connection = connection

    def get_open_positions(self, symbol: str = None) -> List:
        """Return list of open positions, optionally filtered by symbol."""
        if not self.connection.ensure_connected():
            return []
        try:
            positions = mt5.positions_get(symbol=symbol)
            return positions if positions else []
        except Exception as e:
            logger.error("Error fetching positions: %s", e)
            return []

    def count_open_positions(self, symbol: str = None) -> int:
        return len(self.get_open_positions(symbol))

    def calculate_lot_size(self, symbol: str, signal, risk_percent: float, account_balance: float = None) -> float:
        """Calculate lot size based on risk percentage and stop-loss distance."""
        if account_balance is None:
            account_balance = self.connection.account_info.get('balance', 1000)

        if risk_percent <= 0 or account_balance <= 0:
            return 0.01  # fallback

        risk_amount = account_balance * (risk_percent / 100.0)

        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            logger.error("Cannot get symbol info for %s", symbol)
            return 0.01

        point = symbol_info.point
        contract_size = symbol_info.trade_contract_size

        risk_points = abs(signal.entry_price - signal.stop_loss) / point
        if risk_points <= 0:
            logger.warning("Zero risk distance, cannot calculate lot size")
            return 0.01

        # NOTE: point_value assumes quote currency == account currency (USD).
        # For cross-currency pairs (e.g. EURJPY with a USD account), multiply
        # by the relevant FX conversion rate before going live with new symbols.
        point_value = contract_size * point
        lot = risk_amount / (risk_points * point_value)

        # Round to allowed step
        step = symbol_info.volume_step
        lot = round(lot / step) * step
        # Clamp to min/max
        lot = max(symbol_info.volume_min, min(symbol_info.volume_max, lot))
        return lot