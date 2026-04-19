import sys
import os
sys.path.insert(0, os.getcwd())

try:
    import MetaTrader5 as mt5
except:
    print("MT5 not available - checking mock data")
    import sys
    sys.exit()

if mt5.initialize():
    symbol = "BTCUSDm"
    tick = mt5.symbol_info_tick(symbol)
    si = mt5.symbol_info(symbol)
    
    if tick and si:
        spread_raw = tick.ask - tick.bid
        spread_pts = spread_raw / si.point
        spread_pips_00001 = spread_raw / 0.00001
        spread_pips_0001 = spread_raw / 0.0001
        
        print(f"Symbol: {symbol}")
        print(f"Bid: {tick.bid:,.2f}")
        print(f"Ask: {tick.ask:,.2f}")
        print(f"Raw spread: {spread_raw}")
        print(f"Point value: {si.point}")
        print(f"Digits: {si.digits}")
        print(f"---")
        print(f"Spread / 0.01 (pts): {spread_pts:,.0f}")
        print(f"Spread / 0.00001 (pips): {spread_pips_00001:,.0f}")
        print(f"Spread / 0.0001 (pips): {spread_pips_0001:,.0f}")
    else:
        print("No tick data for BTCUSDm")
    
    mt5.shutdown()
else:
    print("MT5 not connected")