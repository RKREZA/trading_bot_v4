import os
import pandas as pd
import json
import sys
from datetime import datetime
sys.path.append(os.getcwd())

from backtesting.monte_carlo import MonteCarloSimulator

def run_certification(symbol, trades_path):
    print(f"--- Institutional Certification Suite: {symbol} ---")
    
    if not os.path.exists(trades_path):
        print(f"Error: {trades_path} not found.")
        return
        
    df = pd.read_csv(trades_path)
    history = df.to_dict('records')
    
    # 2. Run Monte Carlo
    print(f"Running Monte Carlo (2500 iterations) for {len(history)} trades...")
    mc = MonteCarloSimulator(iterations=2500)
    mc_results = mc.run(history, initial_balance=10000.0)
    
    # 3. Print Results
    print("\n" + "="*40)
    print(f"MONTE CARLO ROBUSTNESS RESULTS: {symbol}")
    print(f"Robustness Score: {mc_results['robustness_score']}/100")
    print(f"Median Final Balance: ${mc_results['median_final_balance']}")
    print(f"95% CI Worst Case DD: {mc_results['worst_case_dd_95ci']}")
    print(f"Probability of Ruin: {mc_results['probability_of_ruin']}")
    print("="*40)
    
    # 4. Final Certification Status
    passed = mc_results['robustness_score'] >= 80 and float(mc_results['probability_of_ruin'].replace('%','')) == 0
    status = "CERTIFIED FOR $10M+ DEPLOYMENT" if passed else "REJECTED - INSUFFICIENT ROBUSTNESS"
    
    print(f"\nSTATUS: {status}")
    
    # 5. Save Report
    report = {
        "symbol": symbol,
        "timestamp": datetime.now().isoformat(),
        "total_trades": len(history),
        "monte_carlo": mc_results,
        "certification_status": status
    }
    
    report_dir = os.path.dirname(trades_path)
    with open(os.path.join(report_dir, "certification_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(f"Report saved to {os.path.join(report_dir, 'certification_report.json')}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python scripts/final_certification.py <symbol> <trades_csv_path>")
        sys.exit(1)
    run_certification(sys.argv[1], sys.argv[2])
