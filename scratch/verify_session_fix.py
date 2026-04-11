from core.session_detector import SessionDetector
from datetime import datetime, timezone, timedelta

def test_session_fix():
    # 1. Mocking Saturday 03:15 UTC (Saturday morning)
    # Dhaka Saturday 09:15 AM
    dt_sat = datetime(2026, 4, 11, 3, 15, 0, tzinfo=timezone.utc)
    
    session = SessionDetector.get_session(dt_sat)
    print(f"Mocking Sat 03:15 UTC -> Session: {session}")
    
    if "TOKYO (CLOSED)" in session:
        print("Success: Saturday morning correctly identifies as TOKYO (CLOSED)!")
    else:
        print(f"Failure: Expected TOKYO (CLOSED), got {session}")

    # 2. Mocking Friday 18:00 UTC (New York afternoon)
    dt_fri = datetime(2026, 4, 10, 18, 0, 0, tzinfo=timezone.utc)
    session_fri = SessionDetector.get_session(dt_fri)
    print(f"Mocking Fri 18:00 UTC -> Session: {session_fri}")
    
    if session_fri == "NEW_YORK":
        print("Success: Friday 18:00 UTC correctly identifies as NEW_YORK!")
    else:
        print(f"Failure: Expected NEW_YORK, got {session_fri}")

if __name__ == "__main__":
    test_session_fix()
