import sys
import os
sys.path.append(os.getcwd())

from core.session_detector import SessionDetector
from datetime import datetime, timezone

dt = datetime(2025, 5, 20, 10, 0, 0, tzinfo=timezone.utc)
session = SessionDetector.get_session(dt, 0)
active = SessionDetector.is_session_active(dt, 0, ["LONDON", "NEW_YORK"])

print(f"DIAGNOSTIC: Session @ 10:00 UTC = {session}")
print(f"DIAGNOSTIC: Is Active (LONDON/NY) = {active}")

# Test with MT5 typical offset (UTC+3)
session_mt5 = SessionDetector.get_session(dt, 3)
active_mt5 = SessionDetector.is_session_active(dt, 2, ["LONDON", "NEW_YORK"]) # Often DST is +1
print(f"DIAGNOSTIC: Session @ 10:00 UTC (Offset 3) = {session_mt5}")
print(f"DIAGNOSTIC: Is Active (LONDON/NY, Offset 2) = {active_mt5}")
