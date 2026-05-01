import asyncio
import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime, timezone
import logging
from typing import Optional, Dict, Any, Tuple

from core.time.time_service import time_service

logger = logging.getLogger("trading_bot.mt5_service")

class MT5Service:
    def __init__(self):
        self.connected = False
        self.symbol_cache: Dict[str, dict] = {}

    def connect(self, **kwargs) -> bool:
        """Connects to MT5 terminal."""
        if not mt5.initialize(**kwargs):
            logger.error(f"MT5 initialize failed, error code = {mt5.last_error()}")
            return False
            
        self.connected = True
        logger.info(f"MT5 Connected: {mt5.terminal_info()}")
        return True

    def disconnect(self):
        """Disconnects from MT5 terminal."""
        if self.connected:
            mt5.shutdown()
            self.connected = False
            logger.info("MT5 Disconnected.")

    def get_symbol_info(self, symbol: str) -> Optional[dict]:
        """Fetches and caches symbol info."""
        if not self.connected:
            return None
            
        if symbol in self.symbol_cache:
            return self.symbol_cache[symbol]
            
        info = mt5.symbol_info(symbol)
        if info is None:
            logger.error(f"Symbol {symbol} not found.")
            return None
            
        # Ensure symbol is selected in Market Watch
        if not info.visible:
            if not mt5.symbol_select(symbol, True):
                logger.error(f"Failed to select symbol {symbol} in Market Watch.")
                return None
                
        info_dict = info._asdict()
        self.symbol_cache[symbol] = info_dict
        return info_dict

    def get_tick(self, symbol: str) -> Optional[dict]:
        """Fetches latest tick and updates TimeService."""
        if not self.connected:
            return None
            
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            logger.error(f"Failed to get tick for {symbol}")
            return None
            
        # Update TimeService with the broker time from the tick
        time_service.update_server_time(tick.time)
        
        tick_dict = tick._asdict()
        # Convert timestamp to UTC aware datetime
        tick_dict['time_utc'] = time_service.to_utc(datetime.fromtimestamp(tick.time, tz=timezone.utc))
        return tick_dict

    def get_bid_ask(self, symbol: str) -> Tuple[float, float]:
        """Returns (bid, ask) for a symbol."""
        tick = self.get_tick(symbol)
        if not tick:
            return 0.0, 0.0
        return tick['bid'], tick['ask']

    def get_spread(self, symbol: str) -> float:
        """Returns current spread in points."""
        tick = self.get_tick(symbol)
        if not tick:
            return 0.0
        # Spread is ASK - BID
        bid, ask = tick['bid'], tick['ask']
        info = self.get_symbol_info(symbol)
        if not info:
            return ask - bid
        return (ask - bid) / info['point']

    def copy_rates_range(self, symbol: str, timeframe: int, date_from: datetime, date_to: datetime) -> Optional[pd.DataFrame]:
        """Fetches historical OHLC data aligned to TimeService UTC."""
        if not self.connected:
            return None
            
        rates = mt5.copy_rates_range(symbol, timeframe, date_from, date_to)
        if rates is None or len(rates) == 0:
            logger.warning(f"No rates found for {symbol} from {date_from} to {date_to}")
            return None
            
        df = pd.DataFrame(rates)
        # Convert raw 'time' (broker local int) to UTC datetime
        df['time'] = pd.to_datetime(df['time'], unit='s', utc=True) - time_service.broker_utc_offset
        df.set_index('time', inplace=True)
        return df

    def normalize_price(self, symbol: str, price: float) -> float:
        """Rounds price to symbol's digits."""
        info = self.get_symbol_info(symbol)
        if not info:
            return price
        return round(price, info['digits'])

    def normalize_volume(self, symbol: str, volume: float) -> float:
        """Rounds volume to nearest lot_step, respecting min/max lots."""
        info = self.get_symbol_info(symbol)
        if not info:
            return volume
            
        step = info['volume_step']
        min_vol = info['volume_min']
        max_vol = info['volume_max']
        
        normalized = round(volume / step) * step
        return max(min_vol, min(normalized, max_vol))

    def get_execution_prices(self, symbol: str, direction: str) -> Tuple[float, float]:
        """
        Returns (entry_price, exit_price_type)
        BUY: Enter at ASK, Exit at BID
        SELL: Enter at BID, Exit at ASK
        """
        bid, ask = self.get_bid_ask(symbol)
        if direction.upper() == 'BUY':
            return ask, bid
        else:
            return bid, ask

    async def async_get_tick(self, symbol: str) -> Optional[dict]:
        return await asyncio.to_thread(self.get_tick, symbol)

    async def async_get_symbol_info(self, symbol: str) -> Optional[dict]:
        return await asyncio.to_thread(self.get_symbol_info, symbol)

    async def async_copy_rates_range(self, symbol: str, timeframe: int,
                                      date_from: datetime, date_to: datetime) -> Optional[pd.DataFrame]:
        return await asyncio.to_thread(self.copy_rates_range, symbol, timeframe, date_from, date_to)

    async def async_connect(self, **kwargs) -> bool:
        return await asyncio.to_thread(self.connect, **kwargs)

    async def async_disconnect(self):
        return await asyncio.to_thread(self.disconnect)

    def close_position(self, ticket: int) -> dict:
        if not self.connected:
            return {"success": False, "error": "MT5 not connected"}
        position = mt5.positions_get(ticket=ticket)
        if not position:
            return {"success": False, "error": f"Position {ticket} not found"}
        pos = position[0]
        symbol = pos.symbol
        order_type = mt5.ORDER_TYPE_SELL if pos.type == mt5.POSITION_TYPE_BUY else mt5.ORDER_TYPE_BUY
        tick = mt5.symbol_info_tick(symbol)
        if not tick:
            return {"success": False, "error": f"No tick data for {symbol}"}
        price = tick.bid if pos.type == mt5.POSITION_TYPE_BUY else tick.ask
        info = mt5.symbol_info(symbol)
        filling = mt5.ORDER_FILLING_IOC
        if info and info.filling_mode & mt5.SYMBOL_FILLING_FOK:
            filling = mt5.ORDER_FILLING_FOK
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": float(pos.volume),
            "type": order_type,
            "position": ticket,
            "price": price,
            "deviation": 20,
            "magic": pos.magic,
            "comment": "Dashboard close",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": filling,
        }
        result = mt5.order_send(request)
        if result and result.retcode in [mt5.TRADE_RETCODE_DONE, mt5.TRADE_RETCODE_DONE_PARTIAL]:
            return {"success": True, "ticket": ticket}
        return {"success": False, "error": result.comment if result else "order_send failed", "retcode": result.retcode if result else None}

    async def async_close_position(self, ticket: int) -> dict:
        return await asyncio.to_thread(self.close_position, ticket)

    def modify_sl_tp(self, ticket: int, symbol: str, sl: float, tp: float) -> dict:
        if not self.connected:
            return {"success": False, "error": "MT5 not connected"}
        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "position": ticket,
            "symbol": symbol,
            "sl": float(sl),
            "tp": float(tp),
        }
        result = mt5.order_send(request)
        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
            return {"success": True, "ticket": ticket}
        return {"success": False, "error": result.comment if result else "order_send failed", "retcode": result.retcode if result else None}

    async def async_modify_sl_tp(self, ticket: int, symbol: str, sl: float, tp: float) -> dict:
        return await asyncio.to_thread(self.modify_sl_tp, ticket, symbol, sl, tp)


mt5_service = MT5Service()
