import MetaTrader5 as mt5
from datetime import datetime, timedelta, timezone
import pandas as pd
import json
import os

def get_live_trades_brief():
    if not mt5.initialize():
        print("Failed to initialize MT5")
        return

    # Fetch deals for the last 48 hours
    now = datetime.now()
    start = now - timedelta(days=2)
    
    deals = mt5.history_deals_get(start, now)
    if deals is None or len(deals) == 0:
        print("No trades found in the last 2 days.")
        mt5.shutdown()
        return

    trades = []
    for d in deals:
        # Only interested in the closing part of the deal to see profit/loss
        if d.entry == mt5.DEAL_ENTRY_OUT:
            trades.append({
                "ticket": d.ticket,
                "position_id": d.position_id,
                "symbol": d.symbol,
                "volume": d.volume,
                "direction": "SELL" if d.type == mt5.DEAL_TYPE_SELL else "BUY",
                "profit": d.profit,
                "time": datetime.fromtimestamp(d.time).strftime('%Y-%m-%d %H:%M:%S'),
                "reason": d.reason
            })
    
    if trades:
        df = pd.DataFrame(trades)
        print(df.to_string())
    else:
        print("No OUT deals found.")
    
    mt5.shutdown()

if __name__ == "__main__":
    get_live_trades_brief()
