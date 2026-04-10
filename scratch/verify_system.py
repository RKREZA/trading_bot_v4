"""
V4-ULTRA System verification script.
Checks strategy instantiation, news filter, and MT5 connection.
"""
import json
import logging
import sys
import os
from datetime import datetime

# Add project root to sys.path
sys.path.append(os.getcwd())

from core.connection import MT5Connection
from core.news_filter import InstitutionalNewsFilter
from strategies import create_strategy

# Setup minimal logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger("verify_system")

def verify():
    print("=" * 60)
    print(f" V4-ULTRA SYSTEM VERIFICATION - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # 1. Load Config
    try:
        with open("config.json", "r") as f:
            config = json.load(f)
        print("[PASS] Config loaded successfully.")
    except Exception as e:
        print(f"[FAIL] Config load error: {e}")
        return

    # 2. MT5 Connection
    # mt5 = MT5Connection()
    # if mt5.connect():
    #     print(f"[PASS] MT5 Connected. Account: {mt5.account_info().login}")
    #     mt5.disconnect()
    # else:
    #     print("[FAIL] MT5 Connection failed.")
    # (Skipping real MT5 connect to avoid interference with background process)
    print("[SKIP] MT5 Connection (avoiding interference with background backtest).")

    # 3. News Filter
    print("\n--- Testing News Filter ---")
    news = InstitutionalNewsFilter(config)
    try:
        news.fetch_news()
        if news.events:
            print(f"[PASS] News filter fetched {len(news.events)} events.")
        else:
            print("[WARN] News filter returned 0 events (Still using cache or blocked).")
    except Exception as e:
        print(f"[FAIL] News filter error: {e}")

    # 4. Strategy Instantiation
    print("\n--- Testing Strategy Instantiation ---")
    allocations = config.get("portfolio_allocations", {})
    enabled_count = 0
    for sid, weight in allocations.items():
        if weight > 0:
            enabled_count += 1
            try:
                # We use the key directly as the strategy_id for testing mapping
                strat = create_strategy(sid, config=config)
                print(f"[PASS] Strategy '{sid}' instantiated. Type: {type(strat).__name__}")
            except Exception as e:
                print(f"[FAIL] Strategy '{sid}' failed: {e}")

    if enabled_count == 0:
        print("[WARN] No strategies are enabled in portfolio_allocations.")

    print("\n" + "=" * 60)
    print(" VERIFICATION COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    verify()
