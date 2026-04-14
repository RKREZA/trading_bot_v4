
import subprocess
import json
import os

def run_backtest(params):
    # Update config
    config_path = r"c:\xampp\htdocs\trading_bot_v3\configs\symbols\XAUUSDm.json"
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    config["strategies"]["SmartMeanReversion"].update(params)
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    # Run backtest
    cmd = [
        "python", "backtest.py",
        "--symbol", "XAUUSDm",
        "--from", "2025-01-01",
        "--to", "2026-04-14",
        "--strategy", "SmartMeanReversion",
        "--no-adaptive"
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=r"c:\xampp\htdocs\trading_bot_v3")
    
    # Parse results from output
    output = result.stdout
    try:
        profit_line = [l for l in output.split('\n') if "Net Profit:" in l][0]
        profit = float(profit_line.split("Net Profit:")[1].split("Profit Factor:")[0].strip().replace('$', '').replace(',', ''))
        
        wr_line = [l for l in output.split('\n') if "Win Rate:" in l][0]
        wr = float(wr_line.split("Win Rate:")[1].split("Total Trades:")[0].strip().replace('%', ''))
        
        trades_line = [l for l in output.split('\n') if "Total Trades:" in l][0]
        trades = int(trades_line.split("Total Trades:")[1].strip())
        
        return {"profit": profit, "wr": wr, "trades": trades}
    except Exception as e:
        return {"error": str(e), "output": output[:500]}

# Grid search space
search_space = [
    {"bb_std": 2.2, "rsi_overbought": 75, "rsi_oversold": 25, "sl_atr": 2.0, "tp_atr": 4.0},
    {"bb_std": 2.5, "rsi_overbought": 80, "rsi_oversold": 20, "sl_atr": 2.0, "tp_atr": 3.0},
    {"bb_std": 2.8, "rsi_overbought": 85, "rsi_oversold": 15, "sl_atr": 2.5, "tp_atr": 2.5},
]

results = []
for params in search_space:
    print(f"Testing: {params}")
    res = run_backtest(params)
    res["params"] = params
    results.append(res)
    print(f"Result: {res}")

with open("grid_results.json", "w") as f:
    json.dump(results, f, indent=2)
