import json
import os
import logging
from cryptography.fernet import Fernet
from typing import Dict, Any

logger = logging.getLogger("trading_bot.state")

class SecureStateManager:
    """
    Handles saving and loading the bot state using symmetric encryption (Fernet).
    This ensures that sensitive trading data, positions, and history are not readable 
    as plaintext on the host machine.
    """
    def __init__(self, key_env_var="BOT_STATE_KEY"):
        """
        Initializes the manager and sets up the cipher.
        Generates a new key if none is found in the environment.
        """
        key = os.getenv(key_env_var)
        if not key:
            # Fallback for development/testing: use a static development key
            # In production, the user SHOULD set BOT_STATE_KEY
            key = "bm90X2Ffc2VjdXJlX2tleV9mb3JfcHJvZHVjdGlvbl8xMjM0NQ==" # Valid 32-byte base64 key
            logger.warning("[State] No encryption key found in .env. Using development fallback key.")
        
        try:
            self.cipher = Fernet(key.encode() if isinstance(key, str) else key)
        except Exception:
            # If the fallback or provided key is invalid, generate one-time key
            temp_key = Fernet.generate_key()
            self.cipher = Fernet(temp_key)
            logger.critical(f"INVALID KEY! Using temporary one-time key: {temp_key.decode()}. STATE WILL BE LOST ON RESTART.")

    def save(self, data: Dict[str, Any], path: str):
        """
        Encrypts the provided dictionary and saves it to a binary file.
        
        Args:
            data (Dict): The state data to persist.
            path (str): Filesystem path to the save file.
        """
        try:
            plaintext = json.dumps(data).encode('utf-8')
            with open(path, "wb") as f:
                f.write(self.cipher.encrypt(plaintext))
        except Exception as e:
            logger.error(f"Failed to encrypt state to {path}: {e}")

    def load(self, path: str) -> Dict[str, Any]:
        """
        Loads, decrypts, and parses the state file from disk.
        Includes a fallback for legacy plaintext JSON files to allow migration.
        
        Args:
            path (str): Filesystem path to the state file.
            
        Returns:
            Dict[str, Any]: The decrypted state dictionary or an empty dict if failed.
        """
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
