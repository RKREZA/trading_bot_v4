import subprocess
import time
import os
import sys

def simulate_crash():
    """Created a dummy script that crashes after 2 seconds."""
    dummy_script = "logs/dummy_app.py"
    with open(dummy_script, "w") as f:
        f.write("import sys\nimport time\nprint('Dummy App Starting...')\ntime.sleep(2)\nprint('Dummy App Crashing...')\nsys.exit(1)")
    
    # Start the watchdog to monitor this dummy app
    cmd = [sys.executable, "watchdog.py", "--script", dummy_script]
    print(f"Executing: {' '.join(cmd)}")
    
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    
    # Monitor for 15 seconds to see at least one restart
    start_time = time.time()
    while time.time() - start_time < 20:
        line = process.stdout.readline()
        if line:
            print(f"[WATCHDOG OUTPUT]: {line.strip()}")
        if "Attempt 2/10" in line:
            print("\nSUCCESS: Watchdog detected crash and initiated Attempt 2.")
            process.terminate()
            return True
            
    process.terminate()
    print("\nFAILURE: Watchdog did not seem to restart the process in time.")
    return False

if __name__ == "__main__":
    os.makedirs("logs", exist_ok=True)
    simulate_crash()
