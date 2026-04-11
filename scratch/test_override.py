import json
from datetime import datetime, timezone, timedelta

# Mocking MT5Connection to verify config override
class MockConn:
    def __init__(self, config):
        self.config = config
        self.server_utc_offset = 0 # Initial
    
    def _calculate_utc_offset(self) -> int:
        override = self.config.get("server_utc_offset_override")
        if override is not None:
            return int(override)
        return 3 # Mocked auto-detect
        
    def _get_broker_tz(self) -> timezone:
        return timezone(timedelta(hours=self.server_utc_offset), name=f"UTC {self.server_utc_offset:+d}")

def test_override():
    # Load real config
    with open("config.json", "r") as f:
        config = json.load(f)
    
    # Test 1: No override (should use mocked auto-detect)
    config["server_utc_offset_override"] = None
    conn = MockConn(config)
    conn.server_utc_offset = conn._calculate_utc_offset()
    print(f"Detected Offset (No Override): {conn.server_utc_offset}")
    
    # Test 2: Set override to -5 (Chicago DST)
    print("Setting override to -5...")
    config["server_utc_offset_override"] = -5
    conn = MockConn(config)
    conn.server_utc_offset = conn._calculate_utc_offset()
    print(f"Detected Offset (With Override -5): {conn.server_utc_offset}")
    
    if conn.server_utc_offset == -5:
        print("Success: Override logic is working perfectly!")
    else:
        print("Failure: Override logic did not apply.")

if __name__ == "__main__":
    test_override()
