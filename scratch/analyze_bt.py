import os
import sys
import csv
from pathlib import Path
from datetime import datetime, timezone

# Locate the latest backtest run
bt_dir = Path("backtest_results/XAUUSDm")
if not bt_dir.exists():
    print("No backtest directory found.")
    sys.exit(0)

# Sort runs to get the latest
runs = sorted(bt_dir.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True)
if not runs:
    print("No runs found.")
    sys.exit(0)

latest_run = runs[0]
print(f"Latest run: {latest_run}")

trades_file = latest_run / "trades.csv"
if not trades_file.exists():
    print("No trades.csv found.")
    sys.exit(0)

print(f"Analyzing {trades_file}...")
trades = []
with open(trades_file, "r", newline='') as f:
    reader = csv.DictReader(f)
    for row in reader:
        trades.append(row)

print(f"Total trades in CSV: {len(trades)}")
if len(trades) > 0:
    print("\nTrades Details:")
    for t in trades:
        entry_time = float(t.get('timestamp', 0))
        entry_dt = datetime.fromtimestamp(entry_time, tz=timezone.utc)
        exit_time_str = t.get('exit_time', '')
        if exit_time_str:
            exit_time = float(exit_time_str)
            exit_dt = datetime.fromtimestamp(exit_time, tz=timezone.utc)
        else:
            exit_dt = "OPEN (Forced Close)"
        
        pnl = float(t.get('pnl', 0))
        direction = t.get('direction', 'N/A')
        entry_price = t.get('fill_price', 'N/A')
        exit_price = t.get('exit_price', 'N/A')
        result = t.get('result', 'N/A')
        session = t.get('session', 'N/A')
        
        print(f"  Entry: {entry_dt} ({entry_price}) | Exit: {exit_dt} ({exit_price}) | Dir: {direction} | PnL: ${pnl:.2f} | Res: {result} | Session: {session}")
