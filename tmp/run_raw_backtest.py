"""Script to run multi-strategy backtest and print raw results."""
import json
import sys
import os
from datetime import datetime, timezone
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.data_fetcher import DataFetcher
from core.connection import MT5Connection
from core.backtester import MultiStrategyBacktestEngine
from strategies import create_strategy

def run_test():
    symbol = "XAUUSDm"
    start, end = "2026-01-01", "2026-03-31"

    with open("config.json", "r") as f:
        config = json.load(f)

    conn = MT5Connection()
    conn.config = config
    if not conn.connect():
        print("Failed to connect"); return

    fetcher = DataFetcher()
    dt_from = datetime.fromisoformat(start).replace(tzinfo=timezone.utc)
    dt_to   = datetime.fromisoformat(end).replace(tzinfo=timezone.utc)

    print("Fetching candles...")
    h1  = fetcher.fetch_candles_range(symbol, "H1",  dt_from, dt_to)
    m15 = fetcher.fetch_candles_range(symbol, "M15", dt_from, dt_to)
    m5  = fetcher.fetch_candles_range(symbol, "M5",  dt_from, dt_to)
    d1  = fetcher.fetch_candles_range(symbol, "D1",  dt_from, dt_to)
    print(f"Candles: H1={len(h1)}, M15={len(m15)}, M5={len(m5)}, D1={len(d1)}")

    strategies_to_test = []
    for s_cfg in config.get("strategies", []):
        if s_cfg.get("enabled", True):
            merged = {**config, **s_cfg}
            strategies_to_test.append(create_strategy(s_cfg["id"], s_cfg["type"], merged))

    engine  = MultiStrategyBacktestEngine(config, strategies_to_test)
    results = engine.run(symbol, h1, m15, m5, d1, quiet=False)

    print("\n" + "="*55)
    print("  MULTI-STRATEGY BACKTEST RESULTS (2026 Q1)")
    print("="*55)

    for sid, res in results.items():
        if sid == "portfolio":
            continue
        trades = res.get("trades", [])
        wins   = [t for t in trades if t.get("pnl", 0) > 0]
        losses = [t for t in trades if t.get("pnl", 0) <= 0]
        avg_win  = sum(t["pnl"] for t in wins)  / len(wins)  if wins  else 0
        avg_loss = sum(t["pnl"] for t in losses) / len(losses) if losses else 0
        best  = max((t["pnl"] for t in trades), default=0)
        worst = min((t["pnl"] for t in trades), default=0)

        print(f"\n{'─'*55}")
        print(f"  STRATEGY: {sid}")
        print(f"{'─'*55}")
        print(f"  Net Profit:       ${res.get('net_profit', 0):.2f}")
        print(f"  Final Balance:    ${res.get('final_balance', 1000):.2f}")
        print(f"  Profit Factor:    {res.get('profit_factor', 0):.2f}")
        print(f"  Win Rate:         {res.get('win_rate', 0):.1f}%")
        print(f"  Total Trades:     {res.get('total_trades', 0)}")
        print(f"  Max Drawdown:     {res.get('max_drawdown_pct', 0):.2f}%  (${res.get('max_drawdown_abs', 0):.2f})")
        print(f"  Sharpe Ratio:     {res.get('sharpe_ratio', 0):.2f}")
        print(f"  Expectancy:       ${res.get('expectancy', 0):.2f}")
        print(f"  Avg Win:          ${avg_win:.2f}")
        print(f"  Avg Loss:         ${avg_loss:.2f}")
        print(f"  Best Trade:       ${best:.2f}")
        print(f"  Worst Trade:      ${worst:.2f}")
        print(f"  Cons. Losses Max: {res.get('max_consecutive_losses', 0)}")

        # Session breakdown
        import pandas as pd
        if trades:
            df = pd.DataFrame(trades)
            print(f"\n  Session Breakdown:")
            for sess in ["TOKYO", "LONDON", "LONDON/NY", "NEW_YORK"]:
                sdf = df[df["session"] == sess]
                if sdf.empty: continue
                sw = len(sdf[sdf["pnl"] > 0])
                wr = sw / len(sdf) * 100
                sp = sdf["pnl"].sum()
                print(f"    {sess:<12} Trades:{len(sdf):>3}  WR:{wr:>5.1f}%  PnL:${sp:>8.2f}")

    if "portfolio" in results:
        res = results["portfolio"]
        print(f"\n{'='*55}")
        print(f"  COMBINED PORTFOLIO")
        print(f"{'='*55}")
        print(f"  Net Profit:    ${res.get('net_profit', 0):.2f}")
        print(f"  Total Trades:  {res.get('total_trades', 0)}")
        print(f"  Win Rate:      {res.get('win_rate', 0):.1f}%")
        print(f"  Max Drawdown:  {res.get('max_drawdown_pct', 0):.2f}%")
        pt = res.get("per_strategy_trades", {})
        for k, v in pt.items():
            print(f"    {k}: {v} trades")

    conn.disconnect()
    print()

if __name__ == "__main__":
    run_test()
