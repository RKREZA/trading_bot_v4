import json, numpy as np
from datetime import datetime, timezone
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()
from core.data_fetcher import DataFetcher
from core.backtester import MultiStrategyBacktestEngine
from strategies import create_strategy
from core.connection import MT5Connection

with open("config.json") as f: config = json.load(f)

conn = MT5Connection(); conn.config = config; conn.connect()
fetcher = DataFetcher()
dt_from = datetime(2025, 10, 1, tzinfo=timezone.utc)
dt_to   = datetime(2026, 3, 31, tzinfo=timezone.utc)

for pair in ["EURUSDm", "USDJPYm"]:
    h1  = fetcher.fetch_candles_range(pair, "H1",  dt_from, dt_to)
    m15 = fetcher.fetch_candles_range(pair, "M15", dt_from, dt_to)
    m5  = fetcher.fetch_candles_range(pair, "M5",  dt_from, dt_to)
    d1  = fetcher.fetch_candles_range(pair, "D1",  dt_from, dt_to)

    strats = [create_strategy(s["id"], s["type"], {**config, **s}) for s in config["strategies"] if s.get("enabled", True)]
    engine  = MultiStrategyBacktestEngine(config, strats)
    res = engine.run(pair, h1, m15, m5, d1, quiet=True)
    
    print("\n--- " + pair + " ---")
    s1 = res.get("sniper_v1", {})
    s2 = res.get("smc_v1", {})
    print(f"Sniper Trades: {s1.get('total_trades')}, Net PnL: ${round(s1.get('net_profit', 0), 2)}, WR: {s1.get('win_rate')}%")
    print(f"SMC Trades: {s2.get('total_trades')}, Net PnL: ${round(s2.get('net_profit', 0), 2)}, WR: {s2.get('win_rate')}%")

conn.disconnect()
