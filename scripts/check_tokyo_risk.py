import pandas as pd
import numpy as np
import random
import os
import sys

# Load the latest trades CSV
csv_path = "backtest_results/XAUUSDm_trades_20260329_162426.csv"
if not os.path.exists(csv_path):
    # Try finding the latest if the specific one is missing
    import glob
    files = glob.glob("backtest_results/XAUUSDm_trades_*.csv")
    if files:
        csv_path = max(files, key=os.path.getctime)
        print(f"Warning: {csv_path} used instead of target.")
    else:
        print("Error: No trade history CSV found in backtest_results/")
        sys.exit(1)

df = pd.read_csv(csv_path)

# Filter for Tokyo session only
tokyo = df[df['session'] == 'TOKYO'].copy()
print(f"File used: {os.path.basename(csv_path)}")
print(f"Tokyo trades: {len(tokyo)}")
print(f"Tokyo win rate: {(tokyo['pnl'] > 0).mean()*100:.1f}%")
print(f"Tokyo total PnL: ${tokyo['pnl'].sum():.2f}")

# Monte Carlo simulation on trade sequence
iterations = 1000
initial_balance = 1000.0
max_drawdowns = []

for _ in range(iterations):
    shuffled_pnl = tokyo['pnl'].sample(frac=1).values  # random shuffle
    balance = initial_balance
    equity = [balance]
    for pnl in shuffled_pnl:
        balance += pnl
        equity.append(balance)
    
    equity_series = pd.Series(equity)
    rolling_max = equity_series.cummax()
    drawdown = (rolling_max - equity_series) / rolling_max * 100
    max_drawdowns.append(drawdown.max())

max_drawdowns.sort()
p95 = max_drawdowns[int(iterations * 0.95)]
mean_dd = np.mean(max_drawdowns)

print("\n=== TOKYO-ONLY MONTE CARLO RESULTS ===")
print(f"95th percentile max drawdown: {p95:.1f}%")
print(f"Mean max drawdown: {mean_dd:.1f}%")
print(f"Worst case (max of all shuffles): {max(max_drawdowns):.1f}%")

if p95 <= 25:
    print("\n[OK] RECOMMENDATION: Tokyo-only strategy is robust enough for LIVE trading with reduced position size (e.g., tenth-Kelly).")
elif p95 <= 35:
    print("\n[WARNING] RECOMMENDATION: PAPER trade Tokyo-only for 50 more trades before going live.")
else:
    print("\n[REJECT] RECOMMENDATION: REJECT - Tokyo-only still too fragile (95% DD > 35%).")
