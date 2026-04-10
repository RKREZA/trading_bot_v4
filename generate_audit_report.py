"""
Institutional Audit Report Generator (Dynamic V4)
Automatically analyzes the latest backtest results and certifies institutional readiness.
"""
import pandas as pd
import numpy as np
import os
import glob
from datetime import datetime, timezone

def find_latest_session(base_dir='backtest_results'):
    """Finds the most recently modified session directory."""
    sessions = glob.glob(os.path.join(base_dir, 'session_*'))
    if not sessions:
        return None
    # Filter for directories only
    sessions = [s for s in sessions if os.path.isdir(s)]
    if not sessions:
        return None
    return max(sessions, key=os.path.getmtime)

def generate_report():
    latest_session = find_latest_session()
    if not latest_session:
        print("Error: No backtest sessions found in 'backtest_results/'.")
        return

    print(f"Analyzing latest session: {latest_session}")
    
    trades_path = os.path.join(latest_session, 'trades.csv')
    
    if not os.path.exists(trades_path):
        print(f"Error: trades.csv not found in {latest_session}")
        return

    try:
        df = pd.read_csv(trades_path)
    except Exception as e:
        print(f"Error reading trades.csv: {e}")
        return

    if df.empty:
        print(f"Error: {trades_path} is empty. No trades to analyze.")
        return

    # Basic Data Prep
    df['date'] = pd.to_datetime(df['timestamp'], unit='s', errors='coerce').dt.date
    
    # Portfolio Metrics
    total_trades = len(df)
    grossp = df[df.pnl > 0].pnl.sum()
    grossl = df[df.pnl <= 0].pnl.sum()
    netp = df.pnl.sum()
    wr = (df.pnl > 0).sum() / total_trades * 100 if total_trades > 0 else 0
    pf = grossp / abs(grossl) if abs(grossl) > 0 else float('inf')
    
    # Equity Curve & Drawdown
    # Try to find initial balance from the first trade's balance_at_start
    initial_balance = df['balance_at_start'].iloc[0] if 'balance_at_start' in df.columns else 2000.0
    df['cum_pnl'] = df['pnl'].cumsum()
    df['equity'] = initial_balance + df['cum_pnl']
    df['peak'] = df['equity'].cummax()
    df['drawdown'] = (df['peak'] - df['equity']) / df['peak'] * 100
    maxdd_pct = df['drawdown'].max()
    maxdd_abs = (df['peak'] - df['equity']).max()

    # Statistical Ratios
    avg_pnl = df.pnl.mean()
    std_pnl = df.pnl.std()
    # Simple Sharpe (Trade-based)
    sharpe = (avg_pnl / std_pnl * np.sqrt(252)) if std_pnl > 0 else 0 
    
    # Institutional Verification Logic
    checks = {
        "Sharpe Ratio >= 1.0": sharpe >= 1.0,
        "Win Rate >= 40%": wr >= 40.0,
        "Profit Factor >= 1.2": pf >= 1.2,
        "Max Drawdown < 15%": maxdd_pct < 15.0,
    }
    
    passed_count = sum(checks.values())
    grade = "A+" if passed_count == 4 and sharpe > 2.0 else \
            "A" if passed_count == 4 else \
            "B" if passed_count >= 3 else \
            "C" if passed_count >= 2 else "F"
    
    status = "APPROVED FOR PRODUCTION" if passed_count >= 3 else "REJECTED - NEEDS OPTIMIZATION"

    # Aggregated Breakdowns
    session_breakdown = df.groupby('session').agg({
        'pnl': 'sum',
        'ticket': 'count'
    }).rename(columns={'ticket': 'trades'})
    
    symbol_breakdown = df.groupby('symbol').agg({
        'pnl': 'sum',
        'ticket': 'count'
    }).rename(columns={'ticket': 'trades'})

    report_content = f"""# INSTITUTIONAL AUDIT REPORT - {datetime.now().strftime('%Y-%m-%d')}
## SESSION: {os.path.basename(latest_session)}

### 1. EXECUTIVE SUMMARY
| Metric | Value |
| :--- | :--- |
| **Grade** | **{grade}** |
| **Status** | **{status}** |
| Total Trades | {total_trades} |
| Net Profit | ${netp:.2f} |
| Win Rate | {wr:.1f}% |
| Profit Factor | {pf:.2f} |
| Max Drawdown | {maxdd_pct:.2f}% (${maxdd_abs:.2f}) |
| Sharpe Ratio | {sharpe:.2f} |

---

### 2. INSTITUTIONAL GRADE VERIFICATION
| Requirement | Value | Status |
| :--- | :--- | :--- |
| Sharpe Ratio >= 1.0 | {sharpe:.2f} | {'✅ PASS' if checks["Sharpe Ratio >= 1.0"] else '❌ FAIL'} |
| Win Rate >= 40% | {wr:.1f}% | {'✅ PASS' if checks["Win Rate >= 40%"] else '❌ FAIL'} |
| Profit Factor >= 1.2 | {pf:.2f} | {'✅ PASS' if checks["Profit Factor >= 1.2"] else '❌ FAIL'} |
| Max Drawdown < 15% | {maxdd_pct:.2f}% | {'✅ PASS' if checks["Max Drawdown < 15%"] else '❌ FAIL'} |

---

### 3. SYMBOL PERFORMANCE
| Symbol | Trades | Net Profit |
| :--- | :--- | :--- |
"""
    for sym, row in symbol_breakdown.iterrows():
        report_content += f"| {sym} | {row['trades']} | ${row['pnl']:.2f} |\n"

    report_content += """
---

### 4. SESSION HEATMAP
| Session | Trades | Net Profit |
| :--- | :--- | :--- |
"""
    for sess, row in session_breakdown.iterrows():
        report_content += f"| {sess} | {row['trades']} | ${row['pnl']:.2f} |\n"

    report_content += """
---

### 5. TOP 5 TRADES (BY PNL)
| Time | Symbol | Direction | PnL | Result |
| :--- | :--- | :--- | :--- | :--- |
"""
    for idx, row in df.nlargest(5, 'pnl').iterrows():
        dt = datetime.fromtimestamp(row['timestamp'], tz=timezone.utc).strftime('%Y-%m-%d %H:%M')
        report_content += f"| {dt} | {row['symbol']} | {row['direction']} | ${row['pnl']:.2f} | {row['result']} |\n"

    report_content += """
---

### 6. BOTTOM 5 TRADES (BY PNL)
| Time | Symbol | Direction | PnL | Result |
| :--- | :--- | :--- | :--- | :--- |
"""
    for idx, row in df.nsmallest(5, 'pnl').iterrows():
        dt = datetime.fromtimestamp(row['timestamp'], tz=timezone.utc).strftime('%Y-%m-%d %H:%M')
        report_content += f"| {dt} | {row['symbol']} | {row['direction']} | ${row['pnl']:.2f} | {row['result']} |\n"

    # Save Markdown Report
    output_path = os.path.join(latest_session, 'FULL_AUDIT_REPORT.md')
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    print(f"\nReport generated successfully: {output_path}")
    print("=" * 40)
    print(f"GRADE: {grade}")
    print(f"STATUS: {status}")
    print(f"NET PROFIT: ${netp:.2f}")
    print(f"MAX DRAWDOWN: {maxdd_pct:.2f}%")
    print("=" * 40)

if __name__ == "__main__":
    generate_report()
