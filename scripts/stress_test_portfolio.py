import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

def run_stress_test(csv_path: str):
    print(f"--- Institutional Portfolio Stress Test: {csv_path} ---")
    
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return

    df = pd.read_csv(csv_path)
    if df.empty:
        print("Error: No trade history found in CSV.")
        return

    # Basic Metrics
    total_trades = len(df)
    net_profit = df['pnl'].sum()
    win_rate = (df['pnl'] > 0).mean() * 100
    avg_trade = df['pnl'].mean()
    
    print(f"Base Metrics: Trades={total_trades}, NetProfit=${net_profit:.2f}, WinRate={win_rate:.1f}%")

    # 1. Monte Carlo Simulation (1,000 Trials)
    print("\n[1] Running Monte Carlo Simulation (1,000 Trials)...")
    n_trials = 1000
    initial_balance = df['balance_at_start'].iloc[0] if 'balance_at_start' in df.columns else 10000
    
    returns = df['pnl'].values
    simulated_drawdowns = []
    simulated_final_profits = []
    
    for _ in range(n_trials):
        # Sample with replacement
        sample = np.random.choice(returns, size=len(returns), replace=True)
        equity_curve = initial_balance + np.cumsum(sample)
        
        # Calculate Peak-to-Valley Drawdown
        peak = np.maximum.accumulate(equity_curve)
        drawdown_pct = ((peak - equity_curve) / peak) * 100
        simulated_drawdowns.append(np.max(drawdown_pct))
        simulated_final_profits.append(equity_curve[-1] - initial_balance)

    mc_95_dd = np.percentile(simulated_drawdowns, 95)
    mc_avg_profit = np.mean(simulated_final_profits)
    risk_of_ruin = (np.array(simulated_drawdowns) > 15).mean() * 100 # 15% threshold

    print(f"Monte Carlo 95th Percentile DD: {mc_95_dd:.2f}%")
    print(f"Risk of Ruin (>15% DD): {risk_of_ruin:.2f}%")

    # 2. Cost Robustness (Slippage Stress)
    print("\n[2] Running Cost Robustness (Slippage Stress)...")
    # Estimate pips lost to slippage
    # Logic: if actual_slippage_pips double, profit reduces by (slippage * lots * point_value)
    # Since we have PnL, we can approximate PIP value: PnL / Lots
    df['pip_value_approx'] = df['pnl'] / df['lots']
    
    results = []
    for mult in [1.0, 1.5, 2.0, 3.0, 4.0]:
        # Penalty calculation (simplistic Pip based penalty)
        # Assuming XAUUSDm 0.01 lot = $0.10 per pip
        penalty = (mult - 1.0) * df['actual_slippage_pips'] * df['lots'] * 10 # Approx multiplier
        stressed_pnl = df['pnl'] - penalty
        results.append({
            "Multiplier": f"{mult}x",
            "NetProfit": stressed_pnl.sum(),
            "ProfitFactor": stressed_pnl[stressed_pnl > 0].sum() / abs(stressed_pnl[stressed_pnl < 0].sum()) if any(stressed_pnl < 0) else 99
        })

    cost_df = pd.DataFrame(results)
    print(cost_df.to_string(index=False))

    # 3. Strategy Correlation
    print("\n[3] Strategy Win-Correlation Audit...")
    # Map symbols/timestamps to specific buckets or just check raw sequence
    # For now, just check if strategies lose on the same days
    # This requires session/date buckets
    df['date'] = pd.to_datetime(df['timestamp'], unit='s').dt.date
    pivot = df.pivot_table(index='date', columns='strategy_id', values='pnl', aggfunc='sum').fillna(0)
    corr = pivot.corr()
    print("Strategy Correlation Matrix:")
    print(corr.to_string())

    # Generate Report
    with open("STRESS_TEST_REPORT.md", "w", encoding="utf-8") as f:
        f.write("# Portfolio Stress Test Report\n\n")
        f.write(f"**Source**: {csv_path}\n")
        f.write(f"**Trades**: {total_trades}\n\n")
        
        f.write("## Monte Carlo Analysis (1,000 Trials)\n")
        f.write(f"- **95% Confidence Drawdown**: {mc_95_dd:.2f}%\n")
        f.write(f"- **Probability of Ruin (>15%)**: {risk_of_ruin:.1f}%\n")
        f.write(f"- **Avg. Simulated Profit**: ${mc_avg_profit:.2f}\n\n")
        
        f.write("## Cost Robustness Table\n")
        f.write(cost_df.to_markdown(index=False) + "\n\n")
        
        f.write("## 📉 Correlation Risk\n")
        f.write("```\n")
        f.write(corr.to_string() + "\n")
        f.write("```\n\n")
        
        f.write("## 🏆 Certification Status\n")
        if risk_of_ruin < 1.0 and mc_95_dd < 10.0 and cost_df.iloc[2]['ProfitFactor'] > 1.2:
            f.write("> [!TIP]\n> **STATUS: INSTITUTIONAL GRADE (PASSED)**\n> The portfolio displays high robustness to cost decay and low probability of catastrophic drawdown.")
        else:
            f.write("> [!WARNING]\n> **STATUS: CAUTION REQUIRED**\n> Potential sensitivity to execution costs or tail-risk detected.")

    print(f"\nReport generated: STRESS_TEST_REPORT.md")

if __name__ == "__main__":
    # Institutional Fidelity: Automatically detect the latest backtest session
    import glob
    csv_files = glob.glob("backtest_results/*/trades.csv")
    if not csv_files:
        print("Error: No trades.csv found in any backtest_results subdirectory.")
        sys.exit(1)
        
    latest_session = max(csv_files, key=os.path.getmtime)
    run_stress_test(latest_session)
