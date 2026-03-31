import json
import os
import logging
from cryptography.fernet import Fernet
from typing import Dict, Any

logger = logging.getLogger("trading_bot.state")

class SecureStateManager:
    """Handles saving and loading the bot state using symmetric encryption."""
    def __init__(self, key_env_var="BOT_STATE_KEY"):
        key = os.getenv(key_env_var)
        if not key:
            # Generate a new key and warn the user
            key = Fernet.generate_key().decode()
            logger.critical(f"Generated new state key: {key} — save this to .env as {key_env_var}!")
        
        # Ensure key is bytes
        self.cipher = Fernet(key.encode() if isinstance(key, str) else key)

    def save(self, data: Dict[str, Any], path: str):
        """Encrypt and save state to disk."""
        try:
            plaintext = json.dumps(data).encode('utf-8')
            with open(path, "wb") as f:
                f.write(self.cipher.encrypt(plaintext))
        except Exception as e:
            logger.error(f"Failed to encrypt state to {path}: {e}")

    def load(self, path: str) -> Dict[str, Any]:
        """Load and decrypt state from disk."""
        if not os.path.exists(path):
            return {}
            
        try:
            with open(path, "rb") as f:
                encrypted_data = f.read()
                
            # Handle empty files gracefully
            if not encrypted_data:
                return {}
                
            plaintext = self.cipher.decrypt(encrypted_data)
            return json.loads(plaintext.decode('utf-8'))
        except json.JSONDecodeError:
            # Fallback if the file somehow got saved as plaintext previously
            try:
                with open(path, "r", encoding='utf-8') as f:
                    data = json.load(f)
                    logger.warning("Loaded PLAINTEXT state file. It will be encrypted on next save.")
                    return data
            except Exception as e:
                logger.error(f"Failed to parse plaintext fallback for {path}: {e}")
                return {}
        except Exception as e:
            logger.error(f"Failed to decrypt state from {path}: {e}")
            return {}
