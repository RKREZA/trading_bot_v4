import pandas as pd

df = pd.read_csv("c:/xampp/htdocs/trading_bot_v3/backtest_results/XAUUSDm_trades_20260403_055752.csv")
net_pnl = df['pnl'].sum()
total_trades = len(df)
wins = len(df[df['pnl'] > 0])
win_rate = (wins / total_trades) * 100 if total_trades > 0 else 0
avg_win = df[df['pnl'] > 0]['pnl'].mean() if wins > 0 else 0
avg_loss = df[df['pnl'] < 0]['pnl'].mean() if (total_trades - wins) > 0 else 0

print(f"Run 1 Metrics (Unbiased Baseline):")
print(f"Net P&L: ${net_pnl:.2f}")
print(f"Total Trades: {total_trades}")
print(f"Win Rate: {win_rate:.1f}%")
print(f"Avg Win: ${avg_win:.2f}")
print(f"Avg Loss: ${avg_loss:.2f}")
