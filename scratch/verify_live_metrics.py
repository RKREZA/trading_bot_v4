import MetaTrader5 as mt5
import os
import json
import logging
from datetime import datetime
from core.config.loader import ConfigLoader
from core.session_detector import SessionDetector
from core.connection import MT5Connection
from dotenv import load_dotenv

load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("VERIFY_METRICS")

def verify_live_metrics():
    symbol = "XAUUSDm"
    conn = MT5Connection()
    
    print("\n[1] TERMINAL HANDSHAKE")
    print("-" * 50)
    if not conn.connect():
        print("CRITICAL: Failed to connect to MT5.")
        return
    
    ti = mt5.terminal_info()
    print(f"Terminal: {ti.name} | Path: {ti.path}")
    
    print("\n[2] ACCOUNT SNAPSHOT")
    print("-" * 50)
    acc = conn.get_account_snapshot()
    print(f"Login  : {acc.get('login', 'N/A')}")
    print(f"Balance: ${acc.get('balance', 0.0):,.2f}")
    print(f"Equity : ${acc.get('equity', 0.0):,.2f}")
    print(f"Margin : ${acc.get('margin_free', 0.0):,.2f} FREE")
    
    print("\n[3] SYMBOL TOPOLOGY & CONFIG ALIGNMENT")
    print("-" * 50)
    loader = ConfigLoader()
    config = loader.get_symbol_config(symbol)
    
    # Check if symbol is selected
    if not mt5.symbol_select(symbol, True):
        print(f"ERROR: Failed to select {symbol}")
        return
        
    s_info = mt5.symbol_info(symbol)
    tick = mt5.symbol_info_tick(symbol)
    
    if s_info and tick:
        spread_pts = s_info.spread
        point_val = s_info.point
        spread_pips = spread_pts * point_val / (10 * point_val) # Assuming 5-digit/3-digit system
        # Correct pip calc: spread / 10 if 3/5 digit.
        pips = spread_pts / 10.0
        
        print(f"Symbol : {symbol}")
        print(f"Digits : {s_info.digits}")
        print(f"Bid    : {tick.bid} | Ask: {tick.ask}")
        print(f"Spread : {spread_pts} points ({pips:.1f} pips)")
        
        # Verify Config Overrides
        print(f"\nConfig Check for {symbol}:")
        tf_cfg = config.get("TrendFollowing", {})
        print(f"  ADX Threshold (Config): {tf_cfg.get('adx_threshold')}")
        print(f"  Min Maturity  (Config): {tf_cfg.get('min_trend_maturity')}")
    
    print("\n[4] BROKER CLOCK & SESSION SYNC")
    print("-" * 50)
    server_time = conn.get_broker_time(symbol)
    if server_time:
        # We need the server_utc_offset from global config
        with open("config/config.json", "r") as f:
            g_cfg = json.load(f)
        offset = g_cfg.get("risk_governance", {}).get("server_utc_offset", 2)
        
        session = SessionDetector.get_session(server_time, offset)
        print(f"Broker Time : {server_time}")
        print(f"Detected Session: {session}")
    
    conn.shutdown()
    print("\n" + "="*50)
    print(" LIVE METRIC VERIFICATION COMPLETE ")
    print("="*50)

if __name__ == "__main__":
    verify_live_metrics()
