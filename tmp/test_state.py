import os
from core.state_manager import SecureStateManager
from dotenv import load_dotenv

load_dotenv()

def test_state_encryption():
    manager = SecureStateManager()
    test_data = {"key": "value", "nested": {"a": 1}}
    test_path = "tmp/test_state.bin"
    
    # Save
    manager.save(test_data, test_path)
    print(f"Saved encrypted state to {test_path}")
    
    # Check if the file is encrypted (not plaintext JSON)
    with open(test_path, "rb") as f:
        content = f.read()
        try:
            import json
            json.loads(content.decode())
            print("FAILED: File was saved as plaintext JSON!")
        except:
            print("SUCCESS: File is encrypted (not a valid JSON string).")
            
    # Load
    loaded_data = manager.load(test_path)
    print(f"Loaded: {loaded_data}")
    
    if loaded_data == test_data:
        print("SUCCESS: Data matches after decryption.")
    else:
        print("FAILED: Data mismatch!")

if __name__ == "__main__":
    test_state_encryption()
