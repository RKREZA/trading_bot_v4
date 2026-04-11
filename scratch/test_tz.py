import sys
import os
from datetime import datetime, timezone, timedelta

# Mocking parts of MT5Connection to verify logic
class MockConn:
    def __init__(self, offset):
        self.server_utc_offset = offset
    
    def _get_broker_tz(self) -> timezone:
        return timezone(timedelta(hours=self.server_utc_offset), name=f"UTC{self.server_utc_offset:+d}")

def test_time_logic():
    # User case: Broker is UTC-6
    conn = MockConn(-6)
    tz = conn._get_broker_tz()
    print(f"Timezone name: {tz.tzname(None)}")
    
    # Let's say it's 2:52 AM Saturday UTC
    utc_now = datetime(2026, 4, 11, 2, 52, 0, tzinfo=timezone.utc)
    broker_now = utc_now.astimezone(tz)
    
    print(f"UTC Now: {utc_now.strftime('%d-%b-%Y %I:%M:%S %p (%Z)')}")
    print(f"Broker Now: {broker_now.strftime('%d-%b-%Y %I:%M:%S %p (%Z)')}")
    
    # This should match user's bot output: 10-Apr 08:52 PM (UTC-6)
    if broker_now.day == 10 and broker_now.hour == 20: # 8 PM
        print("Success: Broker time aligns with user's example!")
    else:
        print(f"Mismatch: Broker hour is {broker_now.hour}")

if __name__ == "__main__":
    test_time_logic()
