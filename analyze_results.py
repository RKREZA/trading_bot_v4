import pandas as pd
import matplotlib.pyplot as plt
import json
import os
from datetime import datetime

def generate_report(results_path="backtest_results.json", output_dir="reports"):
    if not os.path.exists(results_path):
        print(f"Error: {results_path} not found.")
        return

    with open(results_path, "r") as f:
        results = json.load(f)

    trades = results.get("trades", [])
    if not trades:
        print("No trades found in results.")
        return

    df = pd.DataFrame(trades)
    df['time'] = pd.to_datetime(df['time'])
    df = df.sort_values('time')

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    report_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(output_dir, f"report_{report_id}")
    os.makedirs(report_path, exist_ok=True)

    # 1. Equity Curve
    plt.figure(figsize=(12, 6))
    equity = results.get("equity_curve", [])
    plt.plot(equity, label="Equity", color="#2ecc71", linewidth=2)
    plt.title("Equity Curve - Backtest Results", fontsize=14, fontweight='bold')
    plt.xlabel("Trade Number")
    plt.ylabel("Balance ($)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(os.path.join(report_path, "equity_curve.png"))
    plt.close()

    # 2. Drawdown Chart
    equity_series = pd.Series(equity)
    rolling_max = equity_series.cummax()
    drawdown = (equity_series - rolling_max) / rolling_max * 100
    
    plt.figure(figsize=(12, 4))
    plt.fill_between(range(len(drawdown)), drawdown, 0, color="#e74c3c", alpha=0.3)
    plt.plot(drawdown, color="#c0392b", linewidth=1)
    plt.title("Drawdown (%)", fontsize=12)
    plt.ylabel("Drawdown %")
    plt.grid(True, alpha=0.2)
    plt.savefig(os.path.join(report_path, "drawdown.png"))
    plt.close()

    # 3. HTML Report
    metrics = {
        "Net Profit": f"${results.get('net_profit', 0):,.2f}",
        "Win Rate": f"{results.get('win_rate', 0):.2f}%",
        "Profit Factor": f"{results.get('profit_factor', 0):.2f}",
        "Sharpe Ratio": f"{results.get('sharpe_ratio', 0):.2f}",
        "Max Drawdown": f"{results.get('max_drawdown', 0):.2f}%",
        "Total Trades": results.get('total_trades', 0),
        "Final Balance": f"${results.get('final_balance', 0):,.2f}"
    }

    html_content = f"""
    <html>
    <head>
        <title>Trading Bot Backtest Report</title>
        <style>
            body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: #f4f7f6; color: #333; margin: 40px; }}
            .container {{ max-width: 1000px; margin: auto; background: white; padding: 30px; border-radius: 12px; box-shadow: 0 4px 15px rgba(0,0,0,0.1); }}
            h1 {{ color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }}
            .metrics-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin-top: 30px; }}
            .metric-card {{ background: #f8f9fa; padding: 20px; border-radius: 8px; border-left: 5px solid #3498db; }}
            .metric-label {{ font-size: 0.9em; color: #7f8c8d; text-transform: uppercase; }}
            .metric-value {{ font-size: 1.4em; font-weight: bold; color: #2c3e50; margin-top: 5px; }}
            .charts {{ margin-top: 40px; }}
            .charts img {{ max-width: 100%; border-radius: 8px; margin-bottom: 20px; box-shadow: 0 2px 10px rgba(0,0,0,0.05); }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Backtest Performance Report</h1>
            <p>Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
            
            <div class="metrics-grid">
                {"".join([f'<div class="metric-card"><div class="metric-label">{k}</div><div class="metric-value">{v}</div></div>' for k, v in metrics.items()])}
            </div>

            <div class="charts">
                <h2>Performance Visualization</h2>
                <img src="equity_curve.png" alt="Equity Curve">
                <img src="drawdown.png" alt="Drawdown">
            </div>
        </div>
    </body>
    </html>
    """

    with open(os.path.join(report_path, "report.html"), "w") as f:
        f.write(html_content)

    print(f"Report generated successfully in: {report_path}")

def generate_wf_report(results_path="wf_results.json", output_dir="reports"):
    if not os.path.exists(results_path):
        print(f"Error: {results_path} not found.")
        return

    with open(results_path, "r") as f:
        results = json.load(f)

    if not results:
        print("No WF results found.")
        return

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    report_id = datetime.now().strftime("WF_%Y%m%d_%H%M%S")
    report_path = os.path.join(output_dir, f"report_{report_id}")
    os.makedirs(report_path, exist_ok=True)

    # 1. IS vs OOS Sharpe Comparison Chart
    labels = [r['window'].split(' to ')[1] for r in results]
    is_sharpes = [r['is_metrics'].get('sharpe_ratio', 0) for r in results]
    oos_sharpes = [r['oos_metrics'].get('sharpe_ratio', 0) for r in results]

    plt.figure(figsize=(12, 6))
    x = range(len(labels))
    width = 0.35
    plt.bar([i - width/2 for i in x], is_sharpes, width, label='In-Sample Sharpe', color='#3498db')
    plt.bar([i + width/2 for i in x], oos_sharpes, width, label='Out-of-Sample Sharpe', color='#e67e22')
    plt.xticks(x, labels, rotation=45)
    plt.title("Walk-Forward Comparison: IS vs OOS Sharpe Ratio", fontsize=14)
    plt.ylabel("Sharpe Ratio")
    plt.legend()
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(report_path, "sharpe_comparison.png"))
    plt.close()

    # 2. Concatenated OOS Equity Curve
    oos_equity = [1000] # Starting balance
    for res in results:
        trades = res['oos_metrics'].get('trades', [])
        for t in trades:
            oos_equity.append(oos_equity[-1] + t.get('pnl', 0))

    plt.figure(figsize=(12, 6))
    plt.plot(oos_equity, color='#27ae60', linewidth=2, label="Concatenated OOS Equity")
    plt.title("Master Out-of-Sample Equity Curve", fontsize=14)
    plt.xlabel("Total OOS Trades")
    plt.ylabel("Balance ($)")
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(report_path, "oos_equity.png"))
    plt.close()

    # 3. Calculate Overfitting Score
    avg_is_sharpe = sum(is_sharpes) / len(is_sharpes) if is_sharpes else 0
    avg_oos_sharpe = sum(oos_sharpes) / len(oos_sharpes) if oos_sharpes else 0
    
    # Overfitting Score Calculation: 1 - (OOS/IS). Higher = More Overfit.
    of_score = 0.0
    if avg_is_sharpe > 0:
        of_score = max(0.0, 1.0 - (avg_oos_sharpe / avg_is_sharpe))

    # 4. Generate HTML
    rows = ""
    for res in results:
        is_m = res['is_metrics']
        oos_m = res['oos_metrics']
        of_ratio = 1 - (oos_m.get('sharpe_ratio', 0) / is_m.get('sharpe_ratio', 1)) if is_m.get('sharpe_ratio', 0) > 0 else 0
        status_color = "#27ae60" if oos_m.get('sharpe_ratio', 0) > 1.0 else "#e67e22"
        
        rows += f"""
        <tr>
            <td>{res['window']}</td>
            <td>{is_m.get('sharpe_ratio', 0):.2f}</td>
            <td style="color: {status_color}; font-weight: bold;">{oos_m.get('sharpe_ratio', 0):.2f}</td>
            <td>{is_m.get('profit_factor', 0):.2f}</td>
            <td>{oos_m.get('profit_factor', 0):.2f}</td>
            <td>{of_ratio*100:.1f}%</td>
        </tr>
        """

    html = f"""
    <html>
    <head>
        <title>Walk-Forward Validation Report</title>
        <style>
            body {{ font-family: sans-serif; margin: 40px; background: #fdfdfd; }}
            .container {{ max-width: 1100px; margin: auto; background: White; padding: 25px; border-radius: 10px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }}
            h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
            .of-score-box {{ background: #f8f9fa; padding: 20px; border-radius: 8px; margin: 20px 0; border: 1px solid #dee2e6; text-align: center; }}
            .of-value {{ font-size: 2em; font-weight: bold; color: {'#e74c3c' if of_score > 0.4 else '#27ae60'}; }}
            table {{ width: 100%; border-collapse: collapse; margin-top: 20px; }}
            th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #eee; }}
            th {{ background: #f8f9fa; }}
            .chart-row {{ display: flex; gap: 20px; margin-top: 30px; }}
            .chart-row img {{ width: 50%; border-radius: 5px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Walk-Forward Robustness Report</h1>
            <div class="of-score-box">
                <div style="color: #7f8c8d;">OVERFITTING SCORE (IS vs OOS DECAY)</div>
                <div class="of-value">{of_score*100:.1f}%</div>
                <p style="font-size: 0.9em;">{ "High Risk of Curve Fitting" if of_score > 0.4 else "Good Generalization" }</p>
            </div>
            
            <table>
                <thead>
                    <tr>
                        <th>Window (OOS)</th>
                        <th>IS Sharpe</th>
                        <th>OOS Sharpe</th>
                        <th>IS PF</th>
                        <th>OOS PF</th>
                        <th>Decay %</th>
                    </tr>
                </thead>
                <tbody>
                    {rows}
                </tbody>
            </table>
            
            <div class="chart-row">
                <img src="sharpe_comparison.png">
                <img src="oos_equity.png">
            </div>
        </div>
    </body>
    </html>
    """

    with open(os.path.join(report_path, "report_wf.html"), "w") as f:
        f.write(html)

    print(f"WF Report generated: {report_path}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--wf":
        generate_wf_report()
    else:
        generate_report()
