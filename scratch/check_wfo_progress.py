import sqlite3
from datetime import datetime

conn = sqlite3.connect('config/wfo_gate.db')
cursor = conn.cursor()

print("WFO Progress Audit:")
print("-------------------")

# Get counts per strategy
cursor.execute("SELECT strategy_id, count(*) FROM wfo_gate GROUP BY strategy_id")
counts = cursor.fetchall()
for sid, count in counts:
    # Get last update for this strategy
    cursor.execute("SELECT recorded_at FROM wfo_gate WHERE strategy_id = ? ORDER BY recorded_at DESC LIMIT 1", (sid,))
    last_update = cursor.fetchone()[0]
    print(f"Strategy: {sid:30} | Windows: {count:3} | Last Update: {last_update}")

# Get total unique windows for today
today = datetime.now().strftime('%Y-%m-%d')
cursor.execute("SELECT count(*) FROM wfo_gate WHERE recorded_at >= ?", (today,))
today_count = cursor.fetchone()[0]
print(f"\nTotal windows completed today: {today_count}")

conn.close()
