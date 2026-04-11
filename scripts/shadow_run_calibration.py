import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
from tabulate import tabulate

def generate_calibration_report(csv_path="logs/shadow_fill_audit.csv"):
    if not os.path.exists(csv_path):
        print(f"Error: Audit log not found at {csv_path}")
        return

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error reading {csv_path}: {e}")
        return

    if df.empty:
        print("Audit log is empty. Execute more trades to generate calibration data.")
        return

    # 1. Clean Data
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    
    # 2. Executive Summary Metrics
    metrics = {
        "Total Audit Samples": len(df),
        "Mean Signed Drift (Pips)": df['signed_drift'].mean(),
        "Median Drift (Pips)": df['signed_drift'].median(),
        "P90 Absolute Drift": df['absolute_drift'].quantile(0.90),
        "P95 Absolute Drift": df['absolute_drift'].quantile(0.95),
        "P99 Absolute Drift (Tail Risk)": df['absolute_drift'].quantile(0.99),
        "Mean Latency (Actual)": df['actual_latency'].mean(),
        "Latency Drift (vs Sim)": (df['actual_latency'] - df['sim_latency']).mean()
    }

    # 3. Strategy Sensitivity Analysis
    strat_perf = df.groupby('strategy_id').agg({
        'signed_drift': ['mean', 'std'],
        'absolute_drift': ['mean', 'max', lambda x: x.quantile(0.95)],
        'actual_latency': 'mean'
    })
    strat_perf.columns = ['Mean Signed', 'Std Dev', 'Mean Abs', 'Max Abs', 'P95 Abs', 'Mean Latency']

    # 4. Latency Bucketing (Rule 6.2)
    bins = [0, 50, 150, 500, float('inf')]
    labels = ['0-50ms', '50-150ms', '150-500ms', '500ms+']
    df['latency_bucket'] = pd.cut(df['actual_latency'], bins=bins, labels=labels)
    latency_dist = df.groupby('latency_bucket').agg({
        'absolute_drift': 'mean',
        'timestamp': 'count'
    }).rename(columns={'timestamp': 'Count', 'absolute_drift': 'Mean Drift'})

    # 5. Regime Correlation Analysis
    regime_stats = df.groupby('regime').agg({
        'absolute_drift': 'mean',
        'spread': 'mean',
        'timestamp': 'count'
    }).rename(columns={'timestamp': 'Samples'})

    # 6. Correlation Engine (Drift vs Physical Market State)
    correlations = {
        "Drift vs Spread": df['absolute_drift'].corr(df['spread']),
        "Drift vs Latency": df['absolute_drift'].corr(df['actual_latency']),
        "Drift vs Spread_Z": df['absolute_drift'].corr(df['spread_zscore'])
    }

    # Generate Markdown Report
    report_md = f"""# INSTITUTIONAL SHADOW-RUN CALIBRATION REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 1. Executive Summary
| Metric | Value |
| :--- | :--- |
"""
    for k, v in metrics.items():
        report_md += f"| {k} | {v:.4f} |\n"

    report_md += "\n## 2. Strategy Sensitivity (Execution Parity)\n"
    report_md += tabulate(strat_perf, headers='keys', tablefmt='github')

    report_md += "\n\n## 3. Latency Bucketing (Broker Behavior Asymmetry)\n"
    report_md += tabulate(latency_dist, headers='keys', tablefmt='github')

    report_md += "\n\n## 4. Regime Analysis\n"
    report_md += tabulate(regime_stats, headers='keys', tablefmt='github')

    report_md += "\n\n## 5. Correlation Engine\n"
    report_md += "| Factor | Correlation Coefficient |\n| :--- | :--- |\n"
    for k, v in correlations.items():
        report_md += f"| {k} | {v:.4f} |\n"

    report_md += """
---
### Phase 1 Certification Thresholds:
- **Mean Signed Drift**: Should be < 0.05 pips (Bias Check)
- **P99 Absolute Drift**: Should be < 0.5 pips (Tail Risk Check)
- **Status**: ANALYSIS REQUIRED
"""

    output_path = "logs/PHASE_1_CALIBRATION_REPORT.md"
    with open(output_path, "w") as f:
        f.write(report_md)
    
    print(f"Calibration Report generated: {output_path}")
    print(tabulate([[k, f"{v:.4f}"] for k, v in metrics.items()], headers=["Metric", "Value"], tablefmt="fancy_grid"))

if __name__ == "__main__":
    generate_calibration_report()
