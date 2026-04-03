import pandas as pd
import glob
import os

def create_report(results_dir="backtest_results"):
    files = glob.glob(os.path.join(results_dir, "*.csv"))
    if not files:
        print("No CSVs found.")
        return

    latest = {}
    for f in files:
        key = "_".join(os.path.basename(f).split("_")[:2])
        if key not in latest or os.path.getmtime(f) > os.path.getmtime(latest[key]):
            latest[key] = f

    report = []
    for key, path in latest.items():
        df = pd.read_csv(path)
        profit = df.pnl.sum()
        trades = len(df)
        win_rate = (len(df[df.pnl > 0]) / trades * 100) if trades > 0 else 0
        report.append({
            "Label": key,
            "Profit ($)": round(profit, 2),
            "Trades": trades,
            "Win Rate (%)": round(win_rate, 2)
        })

    rdf = pd.DataFrame(report).sort_values("Profit ($)", ascending=False)
    print(rdf.to_markdown(index=False))
    
    total_prof = rdf["Profit ($)"].sum()
    print(f"\n[TOTAL PORTFOLIO PROFIT]: ${total_prof:,.2f}")

if __name__ == "__main__":
    create_report()
