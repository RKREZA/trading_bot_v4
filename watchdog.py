import subprocess
import sys
import time
import logging
import os

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

def run_with_recovery(command_args: list):
    """
    Executes the backtest engine and monitors for crashes.
    Automatically appends '--resume' on restarts.
    """
    attempt = 0
    max_attempts = 10
    base_delay = 5
    cmd = [sys.executable, "backtest.py"] + command_args
    
    while attempt < max_attempts:
        try:
            logger.info(f"Starting V4-ULTRA Engine (Attempt {attempt+1}/{max_attempts}): {' '.join(cmd)}")
            
            # Start the process
            process = subprocess.Popen(cmd)
            
            # Wait for completion or crash
            exit_code = process.wait()
            
            if exit_code == 0:
                logger.info("V4-ULTRA Engine completed successfully. Shutting down watchdog.")
                break
            else:
                logger.error(f"V4-ULTRA Engine crashed with exit code {exit_code}. Restarting...")
                
        except Exception as e:
            logger.error(f"Watchdog encountered an error: {e}")
            
        # Preparation for restart
        attempt += 1
        delay = min(base_delay * (2 ** (attempt - 1)), 60)  # Exponential backoff, cap at 60s
        logger.info(f"Waiting {delay}s before retry...")
        time.sleep(delay)
        
        # Append --resume if not already present
        if "--resume" not in cmd:
            cmd.append("--resume")
            logger.info("Modified command to include --resume for state-safe recovery.")
    
    if attempt >= max_attempts:
        logger.critical(f"V4-ULTRA Engine failed after {max_attempts} attempts. Manual intervention required.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python watchdog.py --symbol SYMBOL --from YYYY-MM-DD --to YYYY-MM-DD [other args]")
        sys.exit(1)
        
    os.makedirs("logs", exist_ok=True)
    backtest_args = sys.argv[1:]
    run_with_recovery(backtest_args)
