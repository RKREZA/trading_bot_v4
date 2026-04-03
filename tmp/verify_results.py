import pandas as pd
import numpy as np

df = pd.read_csv('c:/xampp/htdocs/trading_bot_v3/backtest_results/XAUUSDm_trades_20260403_062012.csv')
pnl = df['pnl']
net_profit = pnl.sum()
win_rate = (df['result'] == 'TP').mean() * 100
pos_pnl = pnl[pnl > 0].sum()
neg_pnl = abs(pnl[pnl < 0].sum())
profit_factor = pos_pnl / neg_pnl if neg_pnl != 0 else np.inf

# Drawdown
cum_pnl = pnl.cumsum() + 1000 # Equity curve
peak = cum_pnl.cummax()
drawdown = (peak - cum_pnl)
max_dd = drawdown.max()
max_dd_pct = (drawdown / peak).max() * 100

print(f"Net Profit: ${net_profit:.2f}")
print(f"Win Rate: {win_rate:.2f}%")
print(f"Profit Factor: {profit_factor:.2f}")
print(f"Max Drawdown: ${max_dd:.2f} ({max_dd_pct:.2f}%)")
print(f"Total Trades: {len(df)}")

# Session Breakdown
for session in df['session'].unique():
    s_df = df[df['session'] == session]
    s_pnl = s_df['pnl'].sum()
    s_wr = (s_df['result'] == 'TP').mean() * 100
    print(f"Session {session}: {len(s_df)} trades, {s_wr:.2f}% WR, ${s_pnl:.2f} profit")
