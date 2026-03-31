"""
TRADING BOT V3 - MT5 Connection Manager
Handles connection lifecycle, health checks, and auto-reconnect.
"""

import logging
import os
import time
import math
import threading
from core.lot_calculator import LotCalculator
from datetime import datetime, timezone
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
    
    # Global lock for all mt5.* library calls across the entire application.
    # This prevents race conditions and segmentation faults in the non-thread-safe C-wrapper.
    MT5_LOCK = threading.Lock()

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
                with self.MT5_LOCK:
                    mt5.shutdown()
                time.sleep(2)
                logger.info("Attempt %d/%d...", attempt + 1, self.max_retries)

                with self.MT5_LOCK:
                    success = mt5.initialize(
                        login=creds["login"],
                        password=creds["password"],
                        server=creds["server"],
                        timeout=30000,
                        portable=False,
                    )
                
                if not success:
                    with self.MT5_LOCK:
                        error = mt5.last_error()
                    logger.warning("Failed: %s", error)
                    if error[0] == -10005:
                        logger.warning("MT5 terminal is not running. Open MT5 first!")
                    elif error[0] == -10011 or "Invalid" in str(error):
                        logger.warning("Invalid credentials. Check login/password/server!")
                    time.sleep(3)
                    continue

                with self.MT5_LOCK:
                    info = mt5.account_info()
                
                if info is None:
                    with self.MT5_LOCK:
                        err = mt5.last_error()
                    logger.warning("Account info failed: %s", err)
                    with self.MT5_LOCK:
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
            with self.MT5_LOCK:
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
            with self.MT5_LOCK:
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

    def get_account_snapshot(self) -> dict:
        """Safe access to account info snapshot under MT5_LOCK."""
        if not self.ensure_connected():
            return {"connected": False, "balance": 0, "equity": 0, "profit": 0}
        with self.MT5_LOCK:
            return dict(self.account_info)

    def _update_account_info(self, info):
        """Update cached account information."""
        # Use MT5 server time if available, fall back to local time
        try:
            # We use the time of the last received tick as a proxy for server time
            # Since TerminalInfo doesn't have it in Python, we fetch any liquid symbol
            with self.MT5_LOCK:
                tick = mt5.symbol_info_tick(info.login_symbol if hasattr(info, 'login_symbol') else "BTCUSDm")
            if tick:
                server_time = datetime.fromtimestamp(tick.time, tz=timezone.utc).strftime("%H:%M:%S")
            else:
                server_time = datetime.now().strftime("%H:%M:%S")
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

    def get_market_status(self, symbol: str) -> bool:
        """
        Check if the market is currently open for a symbol.
        Uses candle-freshness (M1) as a broker-agnostic heuristic.
        """
        if mt5 is None: return False
        
        # 1. Basic Trade Mode Check
        with self.MT5_LOCK:
            symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None: return False
        
        # If explicitly disabled, it's definitely closed
        if symbol_info.trade_mode == mt5.SYMBOL_TRADE_MODE_DISABLED:
            return False
            
        # 2. Candle Freshness Check
        # Many brokers (like Exness) keep trade_mode=4 (Full) on weekends,
        # but stop producing candles. We check if the last M1 candle is 'fresh'.
        try:
            with self.MT5_LOCK:
                rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M1, 0, 1)
            if rates is None or len(rates) == 0:
                return False
                
            last_candle_time = rates[0][0] # time field
            
            # Use current UTC time as reference (most brokers are UTC+2/3)
            # 1 hour (3600s) threshold is safe for session/weekend detection
            import time
            if time.time() - last_candle_time > 36000: # 10 hours (Very safe margin for weekends)
                return False
        except Exception:
            return False
                
        return True

    def get_filling_mode(self, symbol: str) -> int:
        """
        Dynamically determine the supported filling mode for a symbol.
        Common modes: FOK (Fill or Kill), IOC (Immediate or Cancel), RETURN.
        """
        if mt5 is None:
            return 0

        with self.MT5_LOCK:
            symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            return mt5.ORDER_FILLING_RETURN

        # filling_mode is a bitmask of supported modes:
        # SYMBOL_FILLING_FOK = 1
        # SYMBOL_FILLING_IOC = 2
        # SYMBOL_FILLING_RETURN = 0 (sometimes implied or required for certain accounts)
        
        filling = symbol_info.filling_mode
        
        if filling & 1: # mt5.SYMBOL_FILLING_FOK
            return mt5.ORDER_FILLING_FOK
        elif filling & 2: # mt5.SYMBOL_FILLING_IOC
            return mt5.ORDER_FILLING_IOC
        else:
            return mt5.ORDER_FILLING_RETURN

    def place_order(self, symbol: str, signal, lot_size: float, max_retries: int = 3, delay: float = 1.0) -> Optional[dict]:
        """
        Place an order in MT5 based on the provided signal, with retry logic.
        """
        if not self.ensure_connected():
            return None

        with self.MT5_LOCK:
            symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            logger.error("%s not found, can not call order_check()", symbol)
            return None

        if not symbol_info.visible:
            logger.info("%s is not visible, trying to switch on", symbol)
            with self.MT5_LOCK:
                select_res = mt5.symbol_select(symbol, True)
            if not select_res:
                logger.error("symbol_select(%s) failed, exit", symbol)
                return None

        # Determine order type and price
        with self.MT5_LOCK:
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

        # Stop Level Check
        stops_level = symbol_info.trade_stops_level
        point = symbol_info.point
        
        for attempt in range(max_retries):
            # Recalculate stops if needed to respect stops_level
            min_dist = stops_level * point
            sl = float(signal.stop_loss)
            tp = float(signal.take_profit)
            
            # Clamp SL/TP to minimum stops_level distance
            if abs(price - sl) < min_dist:
                sl = price - min_dist if signal.direction == "BUY" else price + min_dist
                logger.info("Adjusting SL to respect stops_level: %s", sl)
            
            if tp != 0 and abs(price - tp) < min_dist:
                tp = price + min_dist if signal.direction == "BUY" else price - min_dist
                logger.info("Adjusting TP to respect stops_level: %s", tp)

            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": float(lot_size),
                "type": order_type,
                "price": price,
                "sl": float(sl),
                "tp": float(tp),
                "deviation": deviation,
                "magic": magic,
                "comment": "Bot V3",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": self.get_filling_mode(symbol),
            }

            with self.MT5_LOCK:
                result = mt5.order_send(request)
            
            if result.retcode in [mt5.TRADE_RETCODE_DONE, mt5.TRADE_RETCODE_DONE_PARTIAL, mt5.TRADE_RETCODE_PLACED]:
                logger.info("Order placed successfully. Ticket: %s", result.order)
                return {"ticket": result.order, "volume": result.volume, "price": result.price}

            logger.warning("Order attempt %d/%d failed: retcode=%s, comment=%s",
                           attempt+1, max_retries, result.retcode, result.comment)
            
            if attempt < max_retries - 1:
                # 10004 REQUOTE, 10006 REJECTED, 10020 PRICE_CHANGED
                if result.retcode in [10004, 10006, 10020]: 
                    logger.warning("Retrying with new tick...")
                    time.sleep(0.2)
                    with self.MT5_LOCK:
                        tick = mt5.symbol_info_tick(symbol)
                    if tick:
                        price = tick.ask if signal.direction == "BUY" else tick.bid
                        deviation += 5
                    continue
                
                elif result.retcode == 10021: # INVALID_STOPS
                    logger.warning("Execution rejected (Invalid Stops). Enforcing broker minimum stops...")
                    time.sleep(0.150)
                    with self.MT5_LOCK:
                        tick = mt5.symbol_info_tick(symbol)
                        sym_info = mt5.symbol_info(symbol)
                    
                    if tick and sym_info:
                        price = tick.ask if signal.direction == "BUY" else tick.bid
                        # Absolute minimum distance allowed by broker
                        min_stop_distance = sym_info.trade_stops_level * sym_info.point
                        
                        # Calculate original intended risk
                        calculated_risk = abs(signal.entry_price - signal.stop_loss)
                        # Force SL to be at least the broker's minimum distance
                        safe_risk_distance = max(calculated_risk, min_stop_distance)
                        
                        if signal.direction == "BUY":
                            signal.stop_loss = price - safe_risk_distance
                        else:
                            signal.stop_loss = price + safe_risk_distance
                        
                        signal.entry_price = price
                    continue
                    
                time.sleep(delay)

        return None

    def get_pending_orders(self, symbol: str = None) -> List:
        """Return list of pending orders, optionally filtered by symbol."""
        if not self.ensure_connected():
            return []
        try:
            with self.MT5_LOCK:
                orders = mt5.orders_get(symbol=symbol)
            return list(orders) if orders else []
        except Exception as e:
            logger.error("Error fetching pending orders: %s", e)
            return []

    def get_positions(self, symbol: str = None) -> List:
        """Proxy to fetch open positions from MT5."""
        if not self.ensure_connected():
            return []
        try:
            with self.MT5_LOCK:
                positions = mt5.positions_get(symbol=symbol)
            return list(positions) if positions else []
        except Exception as e:
            logger.error("Error fetching positions: %s", e)
            return []

    def get_symbol_info(self, symbol: str) -> Optional[dict]:
        """Safe wrapper to get core symbol parameters in a dictionary."""
        if not self.ensure_connected():
            return None
        with self.MT5_LOCK:
            info = mt5.symbol_info(symbol)
        if not info:
            return None
        return {
            "point": info.point,
            "trade_tick_value": info.trade_tick_value,
            "volume_min": info.volume_min,
            "volume_max": info.volume_max,
            "volume_step": info.volume_step,
            "trade_stops_level": info.trade_stops_level
        }

    def modify_sl_tp(self, ticket: int, symbol: str, sl: float, tp: float):
        """Request to modify SL/TP for an existing position."""
        if not self.ensure_connected():
            return False
        
        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "position": ticket,
            "symbol": symbol,
            "sl": float(sl),
            "tp": float(tp),
        }
        
        with self.MT5_LOCK:
            result = mt5.order_send(request)
            
        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
            return True
        else:
            logger.warning("SL/TP modification failed for %s: %s", ticket, result.comment if result else "No Result")
            return False


class PositionManager:
    """Manages open positions and risk-based lot sizing."""

    def __init__(self, connection: MT5Connection):
        self.connection = connection

    def get_open_positions(self, symbol: str = None) -> List:
        """Return list of open positions, optionally filtered by symbol and magic number."""
        if not self.connection.ensure_connected():
            return []
        try:
            # Reusing the connection's lock for PositionManager
            with self.connection.MT5_LOCK:
                positions = mt5.positions_get(symbol=symbol)
            
            if not positions:
                return []
            
            # Filter by Magic Number to avoid "Rogue Positions" (manual trades)
            magic = self.connection.config.get("magic_number", BOT_MAGIC_NUMBER)
            return [p for p in positions if p.magic == magic]
            
        except Exception as e:
            logger.error("Error fetching positions: %s", e)
            return []

    def count_open_positions(self, symbol: str = None) -> int:
        return len(self.get_open_positions(symbol))

    def calculate_lot_size(self, symbol: str, signal, risk_percent: float, account_balance: float = None) -> float:
        """Calculate lot size based on risk percentage and stop-loss distance using unified LotCalculator."""
        if account_balance is None:
            account_balance = self.connection.account_info.get('balance', 1000)

        if risk_percent <= 0 or account_balance <= 0:
            return 0.01  # fallback

        risk_amount = account_balance * (risk_percent / 100.0)

        with self.connection.MT5_LOCK:
            symbol_info = mt5.symbol_info(symbol)
        
        if symbol_info is None:
            logger.error("Cannot get symbol info for %s", symbol)
            return 0.01

        sl_distance = abs(signal.entry_price - signal.stop_loss)
        
        # Professional Cross-Currency Lot Sizing logic (delegated to LotCalculator)
        lot = LotCalculator.calculate(
            risk_amount=risk_amount,
            sl_distance=sl_distance,
            tick_size=symbol_info.trade_tick_size,
            tick_value=symbol_info.trade_tick_value,
            volume_min=symbol_info.volume_min,
            volume_max=symbol_info.volume_max,
            volume_step=symbol_info.volume_step
        )

        # Scale down lot proportionally if SL is too tight (to avoid risk inflation)
        point = symbol_info.point
        risk_points = sl_distance / point
        min_sl_points = self.connection.config.get("strategy_defaults", {}).get("min_sl_points", 150)
        if risk_points < min_sl_points:
            lot *= (risk_points / min_sl_points)
            # Re-clamp and snap after scaling
            lot = round(lot / symbol_info.volume_step) * symbol_info.volume_step
            lot = max(symbol_info.volume_min, min(symbol_info.volume_max, lot))

        # Precision handling
        decimals = max(0, -int(math.floor(math.log10(symbol_info.volume_step))))
        return round(float(lot), decimals)