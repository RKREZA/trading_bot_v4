
import MetaTrader5 as mt5
import os

def check():
    if not mt5.initialize():
        print("MT5 Init Failed")
        return
        
    positions = mt5.positions_get()
    if positions:
        print(f"FOUND {len(positions)} OPEN POSITIONS:")
        for p in positions:
            print(f"  ID: {p.ticket} | Symbol: {p.symbol} | Type: {'BUY' if p.type == 0 else 'SELL'} | Profit: {p.profit}")
    else:
        print("No open positions found.")
    
    mt5.shutdown()

if __name__ == "__main__":
    check()
