from datetime import datetime, timezone

timestamps = [1777319940.0, 1777330560.0]

for ts in timestamps:
    dt = datetime.fromtimestamp(ts, tz=timezone.utc)
    print(f"Timestamp {ts} -> {dt} (UTC)")
