import pandas as pd
import numpy as np
import sys

def analyze_backtest(csv_path):
    df = pd.read_csv(csv_path)
    
    total_trades = len(df)
    winners = df[df['pnl'] > 0]
    losers = df[df['pnl'] <= 0]
    
    win_rate = len(winners) / total_trades * 100
    avg_win = winners['pnl'].mean()
    avg_loss = losers['pnl'].mean()
    profit_factor = winners['pnl'].sum() / abs(losers['pnl'].sum())
    
    print(f"Total Trades: {total_trades}")
    print(f"Win Rate: {win_rate:.2f}%")
    print(f"Avg Win: ${avg_win:.2f}")
    print(f"Avg Loss: ${avg_loss:.2f}")
    print(f"Profit Factor: {profit_factor:.2f}")
    print(f"Reward/Risk (Realized): {abs(avg_win/avg_loss):.2f}")
    
    # Analyze results by regime
    print("\nPerformance by Regime:")
    regime_stats = df.groupby('regime')['pnl'].agg(['count', 'sum', 'mean'])
    print(regime_stats)
    
    # Analyze SL hits specifically
    sl_hits = df[df['result'] == 'SL']
    print(f"\nSL Hits: {len(sl_hits)}")
    print(f"Avg PnL on SL: ${sl_hits['pnl'].mean():.2f}")
    
    # Count Break-Evens (PnL close to 0)
    be_trades = df[abs(df['pnl']) < 1.0]
    print(f"Break-Even Trades (<$1): {len(be_trades)} ({len(be_trades)/total_trades*100:.1f}%)")

if __name__ == "__main__":
    analyze_backtest(sys.argv[1])
