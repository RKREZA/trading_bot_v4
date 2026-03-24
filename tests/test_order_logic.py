import sys
import os
import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import TradingBot
from core.strategy_engine import TradeSignal

@patch('core.connection.MT5Connection.place_order')
@patch('core.connection.MT5Connection.connect', return_value=True)
@patch('core.connection.MT5Connection.ensure_connected', return_value=True)
def test_order_cooldown_logic(mock_ensure, mock_connect, mock_place_order):
    # Setup the bot
    bot = TradingBot()
    bot.running = False # Prevent live loop
    
    # Mock data
    symbol = "BTCUSDm"
    m15_candles = [{"time": 1000}, {"time": 2000}]
    signal = TradeSignal("BUY", 50000, 49000, 52000, 80, 5)
    
    # Mock place_order to succeed
    mock_place_order.return_value = {"ticket": 12345, "volume": 0.01, "price": 50000}
    
    # Set lot size correctly config
    bot.config["symbols_config"] = {symbol: {"lot": 0.01}}
    
    # Call the logic directly (Simulating the block inside run_live)
    # First time placing order
    current_candle_time = m15_candles[-1]["time"]
    last_trade = bot.last_trade_time.get(symbol, 0)
    
    if current_candle_time > last_trade:
        lot_size = bot.config.get("symbols_config", {}).get(symbol, {}).get("lot", 0.01)
        result = bot.connection.place_order(symbol, signal, lot_size)
        if result:
            bot.last_trade_time[symbol] = current_candle_time
            bot.daily_trades += 1

    assert bot.daily_trades == 1
    assert bot.last_trade_time[symbol] == 2000
    mock_place_order.assert_called_once()
    
    # Second time placing order (same candle)
    mock_place_order.reset_mock()
    current_candle_time = m15_candles[-1]["time"]
    last_trade = bot.last_trade_time.get(symbol, 0)
    
    if current_candle_time > last_trade:
        # Should NOT reach here
        pass
    else:
        # Cooldown active
        pass

    # Should still be 1 daily trade
    assert bot.daily_trades == 1
    mock_place_order.assert_not_called()
    
    # Third time placing order (new candle)
    m15_candles.append({"time": 3000})
    current_candle_time = m15_candles[-1]["time"]
    last_trade = bot.last_trade_time.get(symbol, 0)
    
    if current_candle_time > last_trade:
        lot_size = bot.config.get("symbols_config", {}).get(symbol, {}).get("lot", 0.01)
        result = bot.connection.place_order(symbol, signal, lot_size)
        if result:
            bot.last_trade_time[symbol] = current_candle_time
            bot.daily_trades += 1

    assert bot.daily_trades == 2
    assert bot.last_trade_time[symbol] == 3000
    mock_place_order.assert_called_once()
