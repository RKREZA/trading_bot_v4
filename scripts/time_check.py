import time
import datetime
import os
import sys

# Add project root to path
sys.path.append(os.getcwd())

try:
    import MetaTrader5 as mt5
except ImportError:
    mt5 = None

def check_time_sync():
    print("--- Institutional Time Synchronization Check ---")
    
    # 1. Local System Time
    local_now = datetime.datetime.now()
    utc_now = datetime.datetime.now(datetime.timezone.utc)
    
    print(f"Local System Time: {local_now.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Universal Time (UTC): {utc_now.strftime('%Y-%m-%d %H:%M:%S')}")

    if mt5 is None:
        print("Error: MetaTrader5 package not found.")
        return

    # 2. MT5 Connection
    login = os.environ.get("MT5_LOGIN")
    password = os.environ.get("MT5_PASSWORD")
    server = os.environ.get("MT5_SERVER")
    
    if not all([login, password, server]):
        print("Error: MT5 credentials not found in environment. Please check your .env file.")
        return

    if not mt5.initialize(login=int(login), password=password, server=server):
        print(f"Error: MT5 Initialization failed: {mt5.last_error()}")
        return

    # 3. Broker Server Time
    # Get last tick of XAUUSDm to get 'server time'
    tick = mt5.symbol_info_tick("XAUUSDm")
    if tick:
        # MT5 tick time is usually Broker Server Time (naive)
        broker_ts = tick.time
        broker_dt = datetime.datetime.fromtimestamp(broker_ts)
        
        # Calculate Offset
        # Difference in hours between Broker Time and UTC Time
        # Note: We round to nearest hour to handle clock drift
        offset_seconds = broker_ts - utc_now.timestamp()
        offset_hours = round(offset_seconds / 3600.0)
        
        print(f"Broker Server Time (Exness): {broker_dt.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Calculated UTC Offset: {offset_hours:+} hours")
        
        # Session Detection Simulation
        from core.session_detector import SessionDetector
        # Simulate London Open (08h UTC)
        london_start_utc = utc_now.replace(hour=8, minute=0, second=0, microsecond=0)
        sim_session = SessionDetector.get_session(london_start_utc, broker_offset_hours=offset_hours)
        print(f"Session Simulation (at 08:00 UTC): {sim_session}")
        
        current_session = SessionDetector.get_session(utc_now, broker_offset_hours=offset_hours)
        print(f"Current Market Session detected: {current_session}")

    mt5.shutdown()

if __name__ == "__main__":
    check_time_sync()
