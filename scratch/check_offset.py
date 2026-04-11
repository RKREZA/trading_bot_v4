import MetaTrader5 as mt5
import time
from datetime import datetime, timezone

def check_sync_values():
    if not mt5.initialize():
        print("Failed to initialize MT5")
        return

    tick = mt5.symbol_info_tick("XAUUSDm")
    info = mt5.account_info()
    
    utc_now = datetime.now(timezone.utc)
    utc_ts = utc_now.timestamp()
    
    print(f"Current UTC: {utc_now.strftime('%Y-%m-%d %H:%M:%S')}")
    
    if tick:
        tick_time = datetime.fromtimestamp(tick.time, tz=timezone.utc) # Wrong tz but for printing raw
        print(f"Last Tick Time (Raw TS): {tick.time} ({tick_time.strftime('%Y-%m-%d %H:%M:%S')})")
        offset_tick = round((tick.time - utc_ts) / 3600.0)
        print(f"Calculated Offset from Tick: {offset_tick}")
    
    if info:
        srv_time = getattr(info, 'server_time', 0)
        srv_dt = datetime.fromtimestamp(srv_time, tz=timezone.utc)
        print(f"Account Server Time (Raw TS): {srv_time} ({srv_dt.strftime('%Y-%m-%d %H:%M:%S')})")
        offset_srv = round((srv_time - utc_ts) / 3600.0)
        print(f"Calculated Offset from Account Info: {offset_srv}")

    mt5.shutdown()

if __name__ == "__main__":
    check_sync_values()
