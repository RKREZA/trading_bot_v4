import json
from core.connection import MT5Connection
import MetaTrader5 as mt5

def check_real_bot_sync():
    # 1. Load config
    with open("config.json", "r") as f:
        config = json.load(f)
    
    conn = MT5Connection()
    conn.config = config
    
    # 2. Connect (Need to initialize MT5 for symbol_info_tick)
    if not mt5.initialize():
        print("Failed to initialize MT5")
        return
        
    # 3. Trigger offset calculation
    offset = conn._calculate_utc_offset()
    print(f"Bot's Calculated Offset: {offset}")
    
    tz = conn._get_broker_tz()
    loc = conn._get_broker_location()
    print(f"Timezone: {tz.tzname(None)}")
    print(f"Location: {loc}")
    
    mt5.shutdown()

if __name__ == "__main__":
    check_real_bot_sync()
