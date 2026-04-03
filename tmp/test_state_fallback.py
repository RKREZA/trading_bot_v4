import os
import logging
from core.state_manager import SecureStateManager

# Setup basic logging to see the output
logging.basicConfig(level=logging.CRITICAL)

def test_fallback():
    # Ensure env var is NOT set
    if "BOT_STATE_KEY" in os.environ:
        del os.environ["BOT_STATE_KEY"]
    
    print("Initializing SecureStateManager without BOT_STATE_KEY...")
    manager = SecureStateManager()
    print("Initialization complete.")

if __name__ == "__main__":
    test_fallback()
