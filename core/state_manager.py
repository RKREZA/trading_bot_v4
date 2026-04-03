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
        
        if key:
            try:
                self.cipher = Fernet(key.encode() if isinstance(key, str) else key)
                return
            except Exception as e:
                logger.error(f"Provided {key_env_var} is invalid: {e}")

        # Fallback: Generate a one-time key and alert the user
        temp_key = Fernet.generate_key().decode()
        self.cipher = Fernet(temp_key.encode())
        
        logger.critical("\n" + "="*60 +
                        f"\nCRITICAL: No valid {key_env_var} set in environment."
                        f"\nGenerated one-time key: {temp_key}"
                        "\n"
                        "\nSet this in your .env to persist state across restarts:"
                        f"\n{key_env_var}={temp_key}"
                        "\n" + "="*60 + "\n")

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
