import pandas as pd
import glob
import os

def analyze_sessions(results_dir="backtest_results"):
    csv_files = glob.glob(os.path.join(results_dir, "*.csv"))
    if not csv_files:
        print("No CSVs found.")
        return

    latest_files = {}
    for f in csv_files:
        base = os.path.basename(f)
        parts = base.split('_')
        if len(parts) < 3: continue
        key = f"{parts[0]}_{parts[1]}"
        if key not in latest_files or os.path.getmtime(f) > os.path.getmtime(latest_files[key]):
            latest_files[key] = f

    results = []
    for key, path in latest_files.items():
        symbol, strategy = key.split('_', 1)
        df = pd.read_csv(path).dropna(subset=['session'])
        
        session_stats = df.groupby('session')['pnl'].agg(['sum', 'count']).reset_index()
        session_stats['symbol'] = symbol
        session_stats['strategy'] = strategy
        results.append(session_stats)

    if not results:
        print("No stats to aggregate.")
        return

    full_stats = pd.concat(results)
    full_stats = full_stats.sort_values('sum')
    print(full_stats[['symbol', 'strategy', 'session', 'sum', 'count']].to_string(index=False))

if __name__ == "__main__":
    analyze_sessions()
