import os
import sys
from datetime import datetime

# Add current directory to path
sys.path.append(r"c:\xampp\htdocs\trading_bot_v3")

from dashboard import TradingDashboard
from rich.console import Console

def test_dashboard():
    dashboard = TradingDashboard()
    
    # Mock state
    state = {
        "connection": {"connected": True},
        "account": {
            "login": 12345678,
            "server": "Demo-Server",
            "balance": 10000.0,
            "equity": 10500.0,
            "margin_free": 9000.0,
            "leverage": 100,
            "profit": 500.0,
        },
        "symbol": "XAUUSDm",
        "price": 4713.05,
        "ask": 4713.35,
        "bid": 4713.05,
        "spread": 300.0,
        "pips": 3.0,
        "digits": 2,
        "tick_lag": 1,
        "session": "London",
        "regime_type": "TRENDING",
        "volatility": "HIGH",
        "server_time": datetime.now().strftime("%d-%b-%Y %H:%M:%S"),
        "login": 12345678,
        "account_name": "Test Account",
        "terminal_path": "C:\\Program Files\\MT5",
        "logs": ["[ANALYSIS] Trend up confirmed", "[TRADE] BUY XAUUSDm @ 2345.67"],
        "setups": {
            "breakout_v4": {
                "signal": "NONE",
                "metrics": {
                    "H1 Body": 0.45,
                    "M5 Body": 0.72,
                    "Volume": 1.25,
                    "Range": "Inside Range"
                },
                "thresholds": {
                    "H1 Body": "> 0.50",
                    "M5 Body": "> 0.65",
                    "Volume": "> 1.0x",
                    "Range": "Breakout"
                }
            }
        },
        "positions": [{"symbol": "XAUUSDm", "type_text": "BUY", "volume": 0.1, "profit": 500.0}]
    }
    
    layout = dashboard.update(state)
    console = Console(width=120)
    console.print(layout)

if __name__ == "__main__":
    test_dashboard()
