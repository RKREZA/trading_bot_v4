import subprocess
import sys
import time
import logging
import os
import argparse

# Production Logging setup for Windows VPS
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | WATCHDOG | %(levelname)s | %(message)s',
    handlers=[
        logging.FileHandler("logs/watchdog.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("watchdog")

def run_with_recovery(target_script: str, command_args: list):
    """
    Executes the target script (main.py or backtest.py) and monitors for crashes.
    Includes a "Stability Reset" to allow indefinite 24/5 uptime.
    """
    attempt = 0
    max_attempts = 10
    base_delay = 5
    stability_threshold_seconds = 3600  # 1 hour
    
    cmd = [sys.executable, target_script] + command_args
    
    while attempt < max_attempts:
        start_time = time.time()
        try:
            logger.info(f"Starting {target_script} (Attempt {attempt+1}/{max_attempts}): {' '.join(cmd)}")
            
            # Start the process
            process = subprocess.Popen(cmd)
            
            # Wait for completion or crash
            exit_code = process.wait()
            
            run_duration = time.time() - start_time
            
            # Graceful Exit (e.g., User pressed Ctrl+C)
            if exit_code == 0:
                logger.info(f"{target_script} exited cleanly. Shutting down watchdog.")
                break
            else:
                logger.error(f"Process crashed with exit code {exit_code}. Ran for {run_duration:.1f} seconds.")
                
        except KeyboardInterrupt:
            logger.info("Watchdog stopped by user.")
            break
        except Exception as e:
            logger.error(f"Watchdog encountered an internal error: {e}")
            run_duration = time.time() - start_time
            
        # ── INSTITUTIONAL FIX: Stability Reset ──
        # If the bot ran successfully for over an hour before crashing, it wasn't a boot loop.
        # Reset the attempt counter to allow continuous 24/5 recovery.
        if run_duration > stability_threshold_seconds:
            logger.info(f"System was stable for {run_duration/60:.1f} minutes. Resetting crash counter to 0.")
            attempt = 0
            
        attempt += 1
        
        # Exponential backoff, cap at 60s
        delay = min(base_delay * (2 ** (attempt - 1)), 60)  
        logger.info(f"Waiting {delay}s before retry to allow OS/MT5 handles to clear...")
        time.sleep(delay)
        
        # State-Safe Recovery parameter (Mostly for backtesting, but safe to pass)
        if target_script == "backtest.py" and "--resume" not in cmd:
            cmd.append("--resume")
            logger.info("Modified command to include --resume for state-safe recovery.")
            
    if attempt >= max_attempts:
        logger.critical(f"FATAL: {target_script} failed to stabilize after {max_attempts} consecutive attempts. Manual intervention required.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="V4-ULTRA System Watchdog")
    parser.add_argument("--script", type=str, default="main.py", help="Target script to monitor (e.g., main.py or backtest.py)")
    
    # Parse known args for watchdog, pass the rest to the target script
    args, unknown_args = parser.parse_known_args()
        
    os.makedirs("logs", exist_ok=True)
    run_with_recovery(args.script, unknown_args)