from datetime import datetime, timezone

def test_news_localization():
    # Mocking a news timestamp (e.g. 2:00 PM UTC)
    ts = 1775829600 # 2026-04-10 14:00:00 UTC
    
    # Existing UTC formatting
    utc_time = datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%H:%M (%Z)")
    print(f"Original UTC format: {utc_time}")
    
    # New Local formatting
    local_time = datetime.fromtimestamp(ts, tz=timezone.utc).astimezone().strftime("%I:%M %p")
    print(f"New Local format: {local_time}")
    
    # For Dhaka (+6), it should be 8:00 PM
    if "08:00 PM" in local_time:
         print("Success: News correctly localized to target zone!")
    else:
         print(f"Warning: News formatted as {local_time}. Verify if this matches your expectation for 2PM UTC.")

if __name__ == "__main__":
    test_news_localization()
