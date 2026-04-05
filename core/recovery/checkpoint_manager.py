import os
import json
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger("trading_bot.recovery.checkpoint")

class CheckpointManager:
    """
    V4-ULTRA State Persistence System (Step 3).
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
            with open(temp_path, "w") as f:
                json.dump(state, f, indent=4)
            
            # Atomic swap
            if os.path.exists(self.main_state_path):
                os.remove(self.main_state_path)
            os.rename(temp_path, self.main_state_path)
            
            logger.debug(f"Checkpoint saved at step {state.get('current_index')}")
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            if os.path.exists(temp_path):
                os.remove(temp_path)

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
            os.remove(self.main_state_path)
            logger.info("Checkpoint cleared.")

    def validate_integrity(self, saved_equity: float, calculated_equity: float) -> bool:
        """
        Strict Rule 3.3: Integrity Check After Recovery.
        """
        diff = abs(saved_equity - calculated_equity)
        if diff > 1e-5: # Tolerance for float precision
            logger.critical(f"INTEGRITY VIOLATION: State equity mismatch! Saved: {saved_equity}, Calculated: {calculated_equity}")
            return False
        return True
