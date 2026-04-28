
import time
import os
from datetime import datetime, timedelta

def get_last_n_lines(file_path, n=20):
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
            return lines[-n:]
    except Exception as e:
        return [f"Error reading {file_path}: {e}"]

def monitor_bot():
    log_path = "logs/v5_live.log"
    report_path = "logs/monitoring_report.md"
    
    start_time = datetime.now()
    end_time = start_time + timedelta(hours=1)
    
    print(f"Starting bot health monitoring for 1 hour...")
    print(f"Start Time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"End Time:   {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("-" * 50)

    while datetime.now() < end_time:
        now_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f"[{now_str}] Checking bot status...")
        
        # Check if log file exists and is being updated
        if os.path.exists(log_path):
            mtime = os.path.getmtime(log_path)
            last_update = datetime.fromtimestamp(mtime)
            age = (datetime.now() - last_update).total_seconds()
            print(f"  Log Update: {last_update.strftime('%H:%M:%S')} ({age:.1f}s ago)")
            
            # Check for errors in the last 20 lines
            last_lines = get_last_n_lines(log_path, 20)
            errors = [line for line in last_lines if "ERROR" in line or "CRITICAL" in line or "FATAL" in line]
            if errors:
                print(f"  [!] Found {len(errors)} issues in recent logs:")
                for err in errors[-3:]:
                    print(f"      - {err.strip()}")
            else:
                print("  Logs: OK (No recent errors)")
        else:
            print(f"  [!] Log file {log_path} not found!")

        # Check monitoring report
        if os.path.exists(report_path):
            report_lines = get_last_n_lines(report_path, 10)
            status_line = next((line for line in report_lines if "Status" in line), "Status: Unknown")
            print(f"  Report {status_line.strip()}")
        
        print("-" * 50)
        time.sleep(600)  # Check every 10 minutes

    print(f"Monitoring complete at {datetime.now().strftime('%H:%M:%S')}")

if __name__ == "__main__":
    monitor_bot()
