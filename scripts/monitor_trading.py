import time
import os
import re
from datetime import datetime

LOG_FILE = "logs/v5_live.log"
REPORT_FILE = "logs/monitoring_report.md"

def get_last_n_lines(file_path, n=500):
    if not os.path.exists(file_path):
        return []
    with open(file_path, "r", errors="ignore") as f:
        # Simple tail implementation
        lines = f.readlines()
        return lines[-n:]

def generate_summary():
    lines = get_last_n_lines(LOG_FILE, 2000)
    if not lines:
        return "No log data found."

    # Parse data
    trades = [l for l in lines if "Execution Success" in l or "Order Sent" in l]
    errors = [l for l in lines if "ERROR" in l or "CRITICAL" in l]
    
    # Get latest ADX and npatt
    analysis_lines = [l for l in lines if "[ANALYSIS]" in l]
    latest_analysis = analysis_lines[-1] if analysis_lines else "N/A"
    
    # Extract values from [ANALYSIS] XAUUSDm ADX:48.4 (TRENDING) | npatt: NONE
    adx_match = re.search(r"ADX:([\d.]+)", latest_analysis)
    npatt_match = re.search(r"npatt: (\w+)", latest_analysis)
    
    adx = adx_match.group(1) if adx_match else "N/A"
    npatt = npatt_match.group(1) if npatt_match else "N/A"
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    summary = f"### Checkpoint: {timestamp}\n"
    summary += f"- **Status**: {'ALIVE' if analysis_lines else 'SILENT'}\n"
    summary += f"- **Latest ADX**: {adx}\n"
    summary += f"- **Latest npatt Signal**: {npatt}\n"
    summary += f"- **Trades in last 10m**: {len(trades)}\n"
    summary += f"- **Errors in last 10m**: {len(errors)}\n"
    
    if trades:
        summary += "#### Recent Trades:\n"
        for t in trades[-5:]:
            summary += f"  - `{t.strip()}`\n"
            
    if errors:
        summary += "#### Recent Errors:\n"
        for e in errors[-5:]:
            summary += f"  - `{e.strip()}`\n"
            
    summary += "\n---\n"
    return summary

def main():
    if not os.path.exists("logs"):
        os.makedirs("logs")
        
    with open(REPORT_FILE, "w") as f:
        f.write("# Trading Monitoring Report (4 Hour Window)\n\n")
        f.write(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

    for i in range(25): # 24 * 10m = 4 hours + initial check
        try:
            summary = generate_summary()
            with open(REPORT_FILE, "a") as f:
                f.write(summary)
            print(f"Cycle {i} completed.")
        except Exception as e:
            with open(REPORT_FILE, "a") as f:
                f.write(f"Monitoring Error: {e}\n")
        
        if i < 24:
            time.sleep(600) # 10 minutes

if __name__ == "__main__":
    main()
