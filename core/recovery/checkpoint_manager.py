import os
import json
import logging
import datetime
import numpy as np
import time
from typing import Dict, Any, Optional
import dataclasses

logger = logging.getLogger("trading_bot.recovery.checkpoint")

class InstitutionalEncoder(json.JSONEncoder):
    """Handles V5-INSIGNIA native types for JSON serialization."""
    def default(self, obj):
        if isinstance(obj, datetime.datetime):
            return obj.isoformat()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if dataclasses.is_dataclass(obj):
            return dataclasses.asdict(obj)
        if hasattr(obj, "__dict__"):
            return obj.__dict__
        return super().default(obj)

class CheckpointManager:
    """
    V5-INSIGNIA State Persistence System (Step 3).
    Saves and restores system state to ensure 'Zero Data Loss' on VPS crashes.
    """
    
    def __init__(self, state_dir: str = "state"):
        self.state_dir = state_dir
        os.makedirs(self.state_dir, exist_ok=True)
        self.main_state_path = os.path.join(self.state_dir, "backtest_state.json")

    def save_checkpoint(self, state: Dict[str, Any]):
        """
        Atomic save of the global backtest state.
        Includes current index, balances, open positions, and strategy states.
        """
        temp_path = self.main_state_path + ".tmp"
        try:
            # 1. Serialize to string first to ensure we don't hold a file lock if serialization fails
            content = json.dumps(state, indent=4, cls=InstitutionalEncoder)
            
            # 2. Write to temp file
            with open(temp_path, "w") as f:
                f.write(content)
            
            # 3. Windows-Safe Atomic Swap
            if os.path.exists(self.main_state_path):
                # On Windows, we often need to retry delete if a process is heartbeat-scanning
                for _ in range(3):
                    try:
                        os.remove(self.main_state_path)
                        break
                    except PermissionError:
                        time.sleep(0.1)
            
            os.rename(temp_path, self.main_state_path)
            logger.debug(f"Checkpoint saved at step {state.get('current_index')}")
            
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            # Cleanup temp file if it exists and isn't locked
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except:
                    pass

    def load_checkpoint(self) -> Optional[Dict[str, Any]]:
        """Loads the last saved state for recovery."""
        if not os.path.exists(self.main_state_path):
            return None
            
        try:
            with open(self.main_state_path, "r") as f:
                state = json.load(f)
            logger.info(f"Checkpoint loaded. Resuming from step {state.get('current_index')}")
            return state
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}. State might be corrupted.")
            return None

    def clear_checkpoint(self):
        """Clears state after successful completion of a run."""
        if os.path.exists(self.main_state_path):
            try:
                os.remove(self.main_state_path)
                logger.info("Checkpoint cleared.")
            except Exception as e:
                logger.warning(f"Failed to clear checkpoint file: {e}")

    def validate_integrity(self, saved_equity: float, calculated_equity: float) -> bool:
        """
        Strict Rule 3.3: Integrity Check After Recovery.
        """
        diff = abs(saved_equity - calculated_equity)
        if diff > 1e-5: # Tolerance for float precision
            logger.critical(f"INTEGRITY VIOLATION: State equity mismatch! Saved: {saved_equity}, Calculated: {calculated_equity}")
            return False
        return True
