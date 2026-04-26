import sys
import os
import re
import subprocess
import json

def set_sl_multiplier(m):
    path = r"c:\xampp\htdocs\trading_bot_v3\strategies\n_pattern_grid.py"
    with open(path, 'r') as f:
        content = f.read()
    
    # Replace the divisor in the SL calculation
    # Example: sl = current_price - (tp_dist / 1.5)
    new_content = re.sub(r"tp_dist / [\d\.]+", f"tp_dist / {m}", content)
    
    with open(path, 'w') as f:
        f.write(new_content)

def run_backtest():
    cmd = [
        "python", "backtest.py", 
        "--symbol", "XAUUSDm", 
        "--from", "2026-04-20", 
        "--to", "2026-04-26", 
        "--strategy", "NPatternGrid", 
        "--no-adaptive"
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=r"c:\xampp\htdocs\trading_bot_v3")
    output = result.stdout
    
    # Parse profit and DD from output
    profit = 0
    dd = 0
    wr = 0
    
    try:
        profit_match = re.search(r"Net Profit:\s+\$([\d\.-]+)", output)
        if profit_match: profit = float(profit_match.group(1))
        
        dd_match = re.search(r"Max Drawdown:\s+([\d\.]+)%", output)
        if dd_match: dd = float(dd_match.group(1))
        
        wr_match = re.search(r"Win Rate:\s+([\d\.]+)%", output)
        if wr_match: wr = float(wr_match.group(1))
    except:
        pass
        
    return profit, dd, wr

if __name__ == "__main__":
    # Test R/R ratios (Reward / Risk)
    # 0.5 means Risk is 2x Reward
    # 2.0 means Reward is 2x Risk
    ratios = [0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 3.0]
    
    summary = []
    for r in ratios:
        print(f"SIMULATING R/R {r}:1 ...")
        set_sl_multiplier(r)
        p, d, w = run_backtest()
        summary.append({"Ratio": r, "Profit": p, "DD": d, "WR": w})
        print(f"  Result: Profit=${p}, DD={d}%, WR={w}%")
        
    print("\nFINAL SIMULATION SUMMARY:")
    print("{:<10} {:<15} {:<10} {:<10}".format("Ratio", "Profit", "DD", "WR"))
    for s in summary:
        print("{:<10} ${:<14.2f} {:<10.2f}% {:<10.2f}%".format(s["Ratio"], s["Profit"], s["DD"], s["WR"]))
