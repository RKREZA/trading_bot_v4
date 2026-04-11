from datetime import datetime, timezone, timedelta

def format_local_tz(dt: datetime) -> str:
    """Returns local timezone in format: (UTC +6) Dhaka"""
    try:
        offset = dt.strftime("%z") # e.g. +0600
        if not offset: return ""
        hours = int(offset[:3])
        # Specific requested format for Dhaka
        if hours == 6:
            return f"(UTC +6) Dhaka"
        # Generic fallback
        sign = "+" if hours >= 0 else "-"
        return f"(UTC {sign}{abs(hours)})"
    except:
        return ""

def test_dhaka_format():
    # Mocking a datetime object in Dhaka (+0600)
    # timedelta is positive for East of UTC
    tz_dhaka = timezone(timedelta(hours=6))
    dt = datetime(2026, 4, 11, 9, 0, 0, tzinfo=tz_dhaka)
    
    formatted_tz = format_local_tz(dt)
    print(f"Formatted TZ: {formatted_tz}")
    
    expected = "(UTC +6) Dhaka"
    if formatted_tz == expected:
        print("Success: Dhaka format is correct!")
    else:
        print(f"Failure: Expected {expected}, got {formatted_tz}")

if __name__ == "__main__":
    test_dhaka_format()
