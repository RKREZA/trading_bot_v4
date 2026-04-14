"""
TRADING BOT V3 - MT5 Connection Manager
Handles connection lifecycle, health checks, and auto-reconnect.
"""

import logging
import os
import time
import math
import threading
from datetime import datetime, timezone, timedelta
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
    """
    Manages the MT5 terminal connection with health checks and auto-reconnect.
    
    This class is the primary interface to the MetaTrader 5 terminal.
    It uses a global MT5_LOCK to ensure thread safety across all mt5 library calls,
    preventing race conditions in the underlying C wrapper.
    """
    
    # Global lock for all mt5.* library calls across the entire application.
    # This prevents race conditions and segmentation faults in the non-thread-safe C-wrapper.
    MT5_LOCK = threading.RLock()

    def __init__(self, max_retries: int = 5, health_check_interval: int = 30):
        """
        Initializes the connection manager.
        
        Args:
            max_retries (int): Number of attempts to reconnect before giving up.
            health_check_interval (int): Seconds between connection pings.
        """
        self.max_retries = max_retries
        self.health_check_interval = health_check_interval
        self.connected = False
        self.account_info: dict = {}
        self._last_health_check = 0.0
        self.config = {}  # Will be set from main
        self.server_utc_offset: int = 0 # Offset in hours to reach Broker Time from UTC

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
        Connects to the MT5 terminal using credentials from environment variables.
        Shuts down any existing connection before starting a new one for a clean slate.
        
        Returns:
            bool: True if connection and account initialization were successful.
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
                
                # Auto-Detect UTC Offset (Step 25)
                self.server_utc_offset = self._calculate_utc_offset()
                logger.info("Universal Time Sync: Broker Offset = %+d hours", self.server_utc_offset)

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
        # Derive server time from MT5's actual server timestamp
        try:
            tz = self._get_broker_tz()
            loc = self._get_broker_location()
            
            server_ts = getattr(info, 'server_time', None)
            if server_ts:
                dt = datetime.fromtimestamp(server_ts, tz=tz)
            else:
                dt = datetime.now(tz)
            
            # Compute ordinal suffix for day (1st, 2nd, 3rd, etc.)
            day = dt.day
            if 11 <= day <= 13:
                suffix = "th"
            else:
                suffix = {1: "st", 2: "nd", 3: "rd"}.get(day % 10, "th")
            
            server_time = dt.strftime(f"{day}{suffix} %B %Y - %I:%M%p ({tz.tzname(None)}) {loc}")
            
        except Exception:
            tz = self._get_broker_tz()
            loc = self._get_broker_location()
            server_time = datetime.now(tz).strftime(f"%d %B %Y - %I:%M%p ({tz.tzname(None)}) {loc}")

        self.account_info = {
            "login": info.login,
            "server": info.server,
            "balance": info.balance,
            "equity": info.equity,
            "profit": info.profit,
            "margin": info.margin,
            "free_margin": info.margin_free,
            "margin_level": info.margin_level if info.margin > 0 else 0,
            "leverage": info.leverage,
            "positions": 0,
            "connected": True,
            "server_time": server_time,
        }

    def _get_broker_tz(self) -> timezone:
        """Returns a timezone object based on the detected server offset."""
        sign = "+" if self.server_utc_offset >= 0 else ""
        return timezone(timedelta(hours=self.server_utc_offset), name=f"UTC {sign}{self.server_utc_offset}")

    def _get_broker_location(self) -> str:
        """Heuristic to guess broker location based on UTC offset."""
        offset = self.server_utc_offset
        # Common Broker Timezone Mappings
        mapping = {
            3: "Cyprus/EE (Market Standard)",
            2: "Cyprus/EE (B)",
            1: "London/WE",
            0: "London/WE",
            -4: "New York/EDT",
            -5: "New York/Chicago (DST)",
            -6: "Chicago/Central (S)",
            -7: "Mountain",
            -8: "Pacific",
            9: "Tokyo",
            8: "Singapore/HK"
        }
        return mapping.get(offset, "Unknown")

    def format_broker_time(self, dt: datetime) -> str:
        """Standardized broker time formatting with location mapping."""
        if not dt: return "N/A"
        tz = self._get_broker_tz()
        loc = self._get_broker_location()
        return dt.strftime(f"%d-%b-%Y %I:%M:%S %p ({tz.tzname(None)}) {loc}")

    def _calculate_utc_offset(self) -> int:
        """Calculates the integer hour offset between Broker Time and UTC."""
        # 1. Check Config Override (Institutional Priority)
        override = self.config.get("server_utc_offset_override")
        if override is not None:
            return int(override)
            
        try:
            with self.MT5_LOCK:
                # Use tick time as the primary source for current server time
                tick = mt5.symbol_info_tick("XAUUSDm")
                
                if tick and tick.time > 0:
                    # Weekend Guard: If tick is older than 2 minutes, market is likely closed.
                    # Do NOT use stale Friday ticks to calculate Saturday offsets.
                    gap_seconds = time.time() - tick.time
                    if gap_seconds > 120:
                        # Return current offset instead of hallucinating a new one
                        if self.server_utc_offset == 0:
                            # If we haven't detected it yet and market is closed, default to +2 (Common broker time)
                            return 2
                        return self.server_utc_offset
                    broker_ts = tick.time
                else:
                    info = mt5.account_info()
                    broker_ts = getattr(info, 'server_time', 0)
                    if broker_ts == 0:
                        return self.server_utc_offset if self.server_utc_offset != 0 else 2
            
            utc_ts = datetime.now(timezone.utc).timestamp()
            offset_hours = round((broker_ts - utc_ts) / 3600.0)
            return offset_hours
        except Exception as e:
            logger.error("Failed to calculate UTC offset: %s. Defaulting to 0.", e)
            return 0

    def get_symbol_snapshot(self, symbol: str) -> dict:
        """Fetch current Bid, Ask, Spread, and Point for a specific symbol."""
        if not self.ensure_connected():
            return {"price": 0, "spread": 0, "point": 0}
            
        with self.MT5_LOCK:
            tick = mt5.symbol_info_tick(symbol)
            if tick:
                # Basic spread calculation: (Ask - Bid) / Point
                point = mt5.symbol_info(symbol).point
                spread_pts = (tick.ask - tick.bid) / point if point > 0 else 0
                return {
                    "price": tick.bid,
                    "spread": spread_pts,
                    "point": point,
                    "bid": tick.bid,
                    "ask": tick.ask
                }
        return {"price": 0, "spread": 0, "point": 0}

    def get_broker_time(self, symbol: str = None) -> datetime:
        """Fetch the current broker server time as a timezone-aware datetime."""
        # Check connection status BEFORE taking the lock to prevent unnecessary contention
        if not self.ensure_connected():
            return None
            
        with self.MT5_LOCK:
            # Most accurate broker time is the last tick of a major symbol
            tick_symbol = symbol if symbol else "XAUUSDm"
            tick = mt5.symbol_info_tick(tick_symbol)
            
            if tick and tick.time > 0:
                return datetime.fromtimestamp(tick.time, tz=self._get_broker_tz())
            
            # Secondary Fallback: Account Snapshot (if your broker supports it)
            info = mt5.account_info()
            server_ts = getattr(info, 'server_time', None)
            if server_ts:
                return datetime.fromtimestamp(server_ts, tz=self._get_broker_tz())
                
            # Institutional Guard: Do NOT fallback to local clock if broker sync fails.
            # This prevents "Premature Candle Evaluation" in MTF strategies.
            return None

    def get_market_status(self, symbol: str) -> bool:
        """
        Determines if the market is open for a specific symbol.
        Uses a broker-agnostic heuristic combining trade_mode and candle freshness.
        
        Args:
            symbol (str): The trading symbol to check.
            
        Returns:
            bool: True if symbol is tradeable and has recent candle activity.
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
            
            # Ensure we compare broker time with TRUE LOCAL time to catch stale ticks
            # On weekends, current_tick.time freezes, resulting in false 'Market Open' reports.
            current_system_time = time.time()
            
            if current_system_time - last_candle_time > 36000: # 10 hours (Very safe margin for weekends)
                return False
        except Exception:
            return False
                
        return True

    def get_filling_mode(self, symbol: str) -> int:
        """
        Queries the broker for the supported order filling mode (FOK, IOC, or RETURN).
        
        Args:
            symbol (str): Symbol to query.
            
        Returns:
            int: The MT5 filling mode constant.
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

    def place_order(self, symbol: str, signal, lot_size: float, max_retries: int = 3, delay: float = 1.0, comment: str = "Bot V3", magic: int = None) -> Optional[dict]:
        """
        Sends a trade request to the MT5 server.
        Includes automatic retries for requotes and price changes,
        and dynamically adjusts stops to respect broker minimum distances.
        
        Args:
            symbol (str): Symbol to trade.
            signal (TradeSignal): Signal object containing direction, SL, and TP.
            lot_size (float): The volume to trade.
            max_retries (int): Number of retries on transient errors.
            delay (float): Wait time between retries.
            comment (str): Order comment for trade attribution (default: "Bot V3").
            magic (Optional[int]): Specific magic number for this trade (default: from config).
            
        Returns:
            Optional[dict]: Dict containing 'ticket', 'volume', and 'price' if successful.
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
        if magic is None:
            magic = self.config.get("magic_number", BOT_MAGIC_NUMBER)

        # Stop Level Check
        stops_level = symbol_info.trade_stops_level
        point = symbol_info.point
        
        # FIX #2: Work with LOCAL copies of SL/TP — never mutate the input signal
        local_sl = float(signal.stop_loss)
        local_tp = float(signal.take_profit)
        local_entry = float(signal.price)
        
        for attempt in range(max_retries):
            # Recalculate stops if needed to respect stops_level
            min_dist = stops_level * point
            sl = local_sl
            tp = local_tp
            
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
                "comment": comment,
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": self.get_filling_mode(symbol),
            }

            with self.MT5_LOCK:
                result = mt5.order_send(request)
            
            # FIX #1: Guard against None return from mt5.order_send (Terminal unresponsive)
            if result is None:
                with self.MT5_LOCK:
                    err = mt5.last_error()
                logger.error("order_send returned None for %s. Terminal may be unresponsive. Error: %s", symbol, err)
                time.sleep(delay)
                continue

            if result.retcode in [mt5.TRADE_RETCODE_DONE, mt5.TRADE_RETCODE_DONE_PARTIAL, mt5.TRADE_RETCODE_PLACED]:
                logger.info("Order placed successfully. Ticket: %s", result.order)
                return {"ticket": result.order, "volume": result.volume, "price": result.price}

            # SAFE LOGGING (Audit Fix): Guard against NoneType return in final warning
            comment = result.comment if result else "TERMINAL_TIMEOUT"
            retcode = result.retcode if result else "NO_RESPONSE"
            logger.warning(f"Order attempt {attempt+1}/{max_retries} failed: retcode={retcode}, comment={comment}")
            
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
                        min_stop_distance = sym_info.trade_stops_level * sym_info.point
                        calculated_risk = abs(local_entry - local_sl)
                        safe_risk_distance = max(calculated_risk, min_stop_distance)
                        
                        # Update LOCAL copies only — signal object is untouched
                        if signal.direction == "BUY":
                            local_sl = price - safe_risk_distance
                            if local_tp != 0:
                                tp_dist = abs(signal.price - signal.take_profit)
                                local_tp = price + max(tp_dist, min_stop_distance)
                        else:
                            local_sl = price + safe_risk_distance
                            if local_tp != 0:
                                tp_dist = abs(signal.price - signal.take_profit)
                                local_tp = price - max(tp_dist, min_stop_distance)
                        
                        local_entry = price
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
            "tick_value": info.trade_tick_value,  # Alias for RiskGuardian
            "volume_min": info.volume_min,
            "min_lot": info.volume_min,           # Alias for RiskGuardian
            "volume_max": info.volume_max,
            "max_lot": info.volume_max,           # Alias for RiskGuardian
            "volume_step": info.volume_step,
            "lot_step": info.volume_step,         # Alias for RiskGuardian
            "trade_stops_level": info.trade_stops_level
        }

    def close_position(self, ticket: int, symbol: str) -> bool:
        """
        Institutional Grade: Fully closes an open position.
        Required for news circuit breakers and proactive risk reduction.
        """
        if not self.ensure_connected():
            return False

        with self.MT5_LOCK:
            position = mt5.positions_get(ticket=ticket)
        if not position:
            logger.error("Close failed: Position %s not found.", ticket)
            return False

        pos = position[0]
        order_type = mt5.ORDER_TYPE_SELL if pos.type == mt5.POSITION_TYPE_BUY else mt5.ORDER_TYPE_BUY
        
        with self.MT5_LOCK:
            tick = mt5.symbol_info_tick(symbol)
        if not tick:
            return False
            
        price = tick.bid if pos.type == mt5.POSITION_TYPE_BUY else tick.ask

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": float(pos.volume),
            "type": order_type,
            "position": ticket,
            "price": price,
            "deviation": 20,
            "magic": pos.magic,
            "comment": "Bot V3 - Flatten",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": self.get_filling_mode(symbol),
        }

        with self.MT5_LOCK:
            result = mt5.order_send(request)
        
        if result and result.retcode in [mt5.TRADE_RETCODE_DONE, mt5.TRADE_RETCODE_DONE_PARTIAL]:
            logger.info("Position %s closed successfully.", ticket)
            return True
        elif result and result.retcode == 10006: # TRADE_RETCODE_REJECTED (Position not found)
            logger.warning("Close attempt: Position %s already closed or missing. Treating as Success.", ticket)
            return True
        else:
            logger.warning("Close failed for %s: %s (Retcode: %s)", ticket, result.comment if result else "No Result", result.retcode if result else "N/A")
            return False

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

    def close_position_partial(self, ticket: int, volume: float) -> bool:
        """Scales out of a position by closing a specific volume."""
        if not self.ensure_connected():
            return False

        with self.MT5_LOCK:
            position = mt5.positions_get(ticket=ticket)
        if not position:
            logger.error("Partial close failed: Position %s not found.", ticket)
            return False

        pos = position[0]
        symbol = pos.symbol
        order_type = mt5.ORDER_TYPE_SELL if pos.type == mt5.POSITION_TYPE_BUY else mt5.ORDER_TYPE_BUY
        
        with self.MT5_LOCK:
            tick = mt5.symbol_info_tick(symbol)
        if not tick:
            return False
            
        price = tick.bid if pos.type == mt5.POSITION_TYPE_BUY else tick.ask

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": float(volume),
            "type": order_type,
            "position": ticket,
            "price": price,
            "deviation": 20,
            "magic": pos.magic,
            "comment": "Partial Close",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": self.get_filling_mode(symbol),
        }

        with self.MT5_LOCK:
            result = mt5.order_send(request)
        
        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
            logger.info("Partial close successful for ticket %s: %s lots closed.", ticket, volume)
            return True
        else:
            logger.warning("Partial close failed for %s: %s", ticket, result.comment if result else "No Result")
            return False


class PositionManager:
    """
    Higher-level manager for tracking open positions and calculating precise lot sizes.
    Filters positions by the bot's magic number to avoid interference with manual trades.
    """

    def __init__(self, connection: MT5Connection):
        """
        Initializes the PositionManager.
        
        Args:
            connection (MT5Connection): The active MT5 terminal connection.
        """
        self.connection = connection

    def get_open_positions(self, symbol: str = None) -> List:
        """
        Fetches positions from MT5 and filters them by the instance magic number.
        
        Args:
            symbol (Optional[str]): Symbol filter.
            
        Returns:
            List: List of filtered MT5 position structures.
        """
        if not self.connection.ensure_connected():
            return []
        try:
            # Reusing the connection's lock for PositionManager
            with self.connection.MT5_LOCK:
                positions = mt5.positions_get(symbol=symbol)
            
            if not positions:
                return []
            
            # Institutional Restoration: Range-based Magic Filtering (Audit Bug #1 Fix)
            # All bot strategies use unique magics in the [BOT_MAGIC_NUMBER, BOT_MAGIC_NUMBER + 999] range.
            base_magic = self.connection.config.get("magic_number", BOT_MAGIC_NUMBER)
            return [p for p in positions if base_magic <= p.magic < base_magic + 1000]
            
        except Exception as e:
            logger.error("Error fetching positions: %s", e)
            return []

    def get_positions_by_magic(self, magic: int, symbol: str = None) -> List:
        """
        Institutional: Retrieves positions filtered by a specific magic number.
        Essential for Anti-Grid logic and strategy isolation.
        """
        if not self.connection.ensure_connected():
            return []
        try:
            with self.connection.MT5_LOCK:
                positions = mt5.positions_get(symbol=symbol)
            if not positions:
                return []
            return [p for p in positions if p.magic == magic]
        except Exception as e:
            logger.error("Error in get_positions_by_magic: %s", e)
            return []

    def count_open_positions(self, symbol: str = None) -> int:
        return len(self.get_open_positions(symbol))
