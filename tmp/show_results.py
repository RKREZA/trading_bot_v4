"""Clean results output for v4 strategies — tqdm suppressed."""
import json, sys, os, numpy as np
from datetime import datetime, timezone
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.data_fetcher import DataFetcher
from core.connection import MT5Connection
from core.backtester import MultiStrategyBacktestEngine
from strategies import create_strategy

with open("config.json") as f:
    config = json.load(f)

conn = MT5Connection(); conn.config = config; conn.connect()
fetcher = DataFetcher()
dt_from = datetime(2025, 10, 1, tzinfo=timezone.utc)
dt_to   = datetime(2026, 3, 31, tzinfo=timezone.utc)

h1  = fetcher.fetch_candles_range("XAUUSDm", "H1",  dt_from, dt_to)
m15 = fetcher.fetch_candles_range("XAUUSDm", "M15", dt_from, dt_to)
m5  = fetcher.fetch_candles_range("XAUUSDm", "M5",  dt_from, dt_to)
d1  = fetcher.fetch_candles_range("XAUUSDm", "D1",  dt_from, dt_to)

strats = [
    create_strategy(s["id"], s["type"], {**config, **s})
    for s in config["strategies"] if s.get("enabled", True)
]

engine  = MultiStrategyBacktestEngine(config, strats)
results = engine.run("XAUUSDm", h1, m15, m5, d1, quiet=True)

for sid, res in results.items():
    if sid in ("portfolio", "walk_forward"):
        continue
    trades = res.get("trades", [])
    wins   = [t for t in trades if t.get("pnl", 0) > 0]
    losses = [t for t in trades if t.get("pnl", 0) <= 0]
    avg_w  = sum(t["pnl"] for t in wins)   / len(wins)   if wins   else 0
    avg_l  = sum(t["pnl"] for t in losses) / len(losses) if losses else 0
    best   = max((t["pnl"] for t in trades), default=0)
    worst  = min((t["pnl"] for t in trades), default=0)
    consec = res.get("max_consecutive_losses",0)
    total_comm = sum(t.get("commission", 0) for t in trades)
    total_swap = sum(t.get("swap", 0) for t in trades)

    print("Strategy:          " + sid)
    print("Net Profit:        $" + str(round(res.get("net_profit",0),2)))
    print("Final Balance:     $" + str(round(res.get("final_balance",1000),2)))
    print("Profit Factor:     " + str(res.get("profit_factor",0)))
    print("Win Rate:          " + str(res.get("win_rate",0)) + "%")
    print("Total Trades:      " + str(res.get("total_trades",0)))
    print("Max Drawdown:      " + str(res.get("max_drawdown_pct",0)) + "%")
    print("Sharpe Ratio:      " + str(res.get("sharpe_ratio",0)))
    print("Expectancy:        $" + str(res.get("expectancy",0)))
    print("Avg Win:           $" + str(round(avg_w,2)))
    print("Avg Loss:          $" + str(round(avg_l,2)))
    print("Best Trade:        $" + str(round(best,2)))
    print("Worst Trade:       $" + str(round(worst,2)))
    print("Max Consec Losses: " + str(consec))
    print("Commission paid:   $" + str(round(total_comm,2)))
    print("Swap paid:         $" + str(round(total_swap,2)))

    # Session breakdown
    from collections import defaultdict
    sdf = defaultdict(lambda: {"trades":0,"wins":0,"pnl":0.0})
    for t in trades:
        s2 = t.get("session","?")
        sdf[s2]["trades"] += 1
        sdf[s2]["pnl"] += t.get("pnl",0)
        if t.get("pnl",0) > 0: sdf[s2]["wins"] += 1
    print("Session Breakdown:")
    for sess in ["TOKYO","LONDON","LONDON/NY","NEW_YORK"]:
        if sess in sdf:
            d = sdf[sess]
            wr = round(d["wins"]/max(d["trades"],1)*100,1)
            print("  " + sess.ljust(12) + " Trades:" + str(d["trades"]).rjust(3) + "  WR:" + str(wr) + "%  PnL:$" + str(round(d["pnl"],2)))
    print()

if "portfolio" in results:
    r = results["portfolio"]
    print("COMBINED PORTFOLIO")
    print("Net Profit:        $" + str(r.get("net_profit",0)))
    print("Total Trades:      " + str(r.get("total_trades",0)))
    print("Win Rate:          " + str(r.get("win_rate",0)) + "%")
    print("Max Drawdown:      " + str(r.get("max_drawdown_pct",0)) + "%")

conn.disconnect()
