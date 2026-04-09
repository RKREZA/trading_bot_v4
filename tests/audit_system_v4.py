import subprocess
import os
import json
import time
from datetime import datetime
from pathlib import Path

SYMBOLS = ["XAUUSDm"]
START_DATE = "2025-10-01"
END_DATE = "2026-04-01"

def run_backtest(symbol, start, end):
    print(f"\n[AUDIT] Starting backtest for {symbol} ({start} to {end})...")
    cmd = [
        "python", "backtest.py",
        "--symbol", symbol,
        "--from", start,
        "--to", end,
        "--monte-carlo",
        "--walk-forward",
        "--stress-test",
        "--debug-signals"
    ]
    
    try:
        # Use Popen to stream output in real-time
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        full_output = []
        for line in process.stdout:
            print(line, end="")
            full_output.append(line)
        
        process.wait()
        if process.returncode == 0:
            print(f"\n[AUDIT] {symbol} completed successfully.")
            return "".join(full_output)
        else:
            print(f"\n[AUDIT] {symbol} FAILED with exit code {process.returncode}")
            return None
    except Exception as e:
        print(f"[AUDIT] Unexpected error running {symbol}: {e}")
        return None

def find_latest_session():
    results_dir = Path("backtest_results")
    if not results_dir.exists():
        return None
    sessions = sorted([d for d in results_dir.iterdir() if d.is_dir()], key=os.path.getmtime, reverse=True)
    return sessions[0] if sessions else None

def parse_summary(session_path):
    summary_file = session_path / "summary.md"
    if not summary_file.exists():
        return "No summary found."
    with open(summary_file, "r") as f:
        return f.read()

def main():
    start_time = time.time()
    all_summaries = {}
    
    for symbol in SYMBOLS:
        output = run_backtest(symbol, START_DATE, END_DATE)
        session = find_latest_session()
        if session:
            summary = parse_summary(session)
            all_summaries[symbol] = {
                "session": str(session),
                "summary": summary,
                "raw_output": output
            }
        else:
            all_summaries[symbol] = {"error": "No session directory found."}

    # Generate Audit Report
    report_path = Path("backtest_results/FULL_AUDIT_REPORT.md")
    with open(report_path, "w") as f:
        f.write("# V4-ULTRA INSTITUTIONAL CERTIFICATION REPORT\n")
        f.write(f"Generated at: {datetime.now()}\n")
        f.write(f"Audit Range: {START_DATE} to {END_DATE}\n\n")
        
        for symbol, data in all_summaries.items():
            f.write(f"## {symbol} Audit Analysis\n")
            if "error" in data:
                f.write(f"ERROR: {data['error']}\n\n")
            else:
                f.write(f"Session: `{data['session']}`\n\n")
                f.write(data["summary"])
                f.write("\n\n" + "-"*50 + "\n\n")
    
    print(f"\n[AUDIT] Master Certification Report generated at: {report_path}")
    print(f"[AUDIT] Total duration: {time.time() - start_time:.1f} seconds.")

if __name__ == "__main__":
    main()
