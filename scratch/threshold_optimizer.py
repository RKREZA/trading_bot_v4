import subprocess
import json
import os
import re

def run_backtest(tokyo, london, ny, cross):
    # Temporary modify the strategy file
    strat_path = r"c:\xampp\htdocs\trading_bot_v3\strategies\n_pattern_grid.py"
    with open(strat_path, "r") as f:
        content = f.read()
    
    # Use regex to inject the test thresholds
    new_logic = f"""
        if "TOKYO" in session:
            body_ratio_thresh = {tokyo}
        elif "LONDON" in session:
            body_ratio_thresh = {london}
        elif "NEW_YORK" in session:
            body_ratio_thresh = {ny}
        elif "LONDON/NY" in session:
            body_ratio_thresh = {cross}
        else:
            body_ratio_thresh = 0.87
    """
    
    # Find the block and replace it
    pattern = r'if "TOKYO" in session:.*?else:.*?body_ratio_thresh = 0.87'
    modified_content = re.sub(pattern, new_logic.strip(), content, flags=re.DOTALL)
    
    with open(strat_path, "w") as f:
        f.write(modified_content)
        
    # Run backtest
    cmd = "python backtest.py --symbol XAUUSDm --from 2026-04-01 --to 2026-04-24 --strategy NPatternGrid"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=r"c:\xampp\htdocs\trading_bot_v3")
    
    # Parse output for Profit and Drawdown
    profit = 0
    drawdown = 100
    
    match_profit = re.search(r"Net Profit:\s+\$(-?\d+\.\d+)", result.stdout)
    match_dd = re.search(r"Max Drawdown:\s+(\d+\.\d+)%", result.stdout)
    
    if match_profit: profit = float(match_profit.group(1))
    if match_dd: drawdown = float(match_dd.group(1))
    
    return profit, drawdown

# Test Matrix
tests = [
    (0.92, 0.89, 0.85, 0.85), # Conservative Balanced
    (0.94, 0.90, 0.88, 0.88), # High Precision
    (0.90, 0.85, 0.80, 0.80), # Aggressive (Risk Check)
    (0.92, 0.87, 0.82, 0.82), # Optimized Impulse
]

results = []
for t in tests:
    print(f"Testing Matrix: {t}...")
    p, d = run_backtest(*t)
    results.append({"config": t, "profit": p, "drawdown": d})
    print(f" -> Profit: ${p}, DD: {d}%")

print("\n--- FINAL OPTIMIZATION REPORT ---")
for r in sorted(results, key=lambda x: x['profit'], reverse=True):
    print(f"Config {r['config']} | Profit: ${r['profit']} | DD: {r['drawdown']}%")
