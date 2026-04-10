import json, sys, os
sys.path.append(os.getcwd())
from dotenv import load_dotenv
load_dotenv()
from datetime import datetime, timezone, timedelta
from core.data.manager import DataManager
from core.connection import MT5Connection
from core.regime_detector import RegimeDetector
from core.indicator_engine import IndicatorEngine
from collections import Counter

with open("config.json") as f: config = json.load(f)
conn = MT5Connection()
conn.connect()
dm = DataManager(config)

end_dt = datetime.now(timezone.utc)
start_dt = end_dt - timedelta(days=90)
m5 = dm.prepare_data("XAUUSDm", "M5", start_dt)
m5._indicators = IndicatorEngine.precalculate_all("XAUUSDm", "M5", m5)

rd = RegimeDetector()
regimes = Counter()
for i in range(200, len(m5.time)):
    m5.set_limit(i)
    info = rd.detect(m5)
    regimes[info.market_type.value] += 1

total = sum(regimes.values())
print("\n=== REGIME DISTRIBUTION (90 days, XAUUSDm) ===")
for r, c in regimes.most_common():
    print(f"  {r:12s}: {c:6d} bars ({c/total*100:.1f}%)")
print(f"  {'TOTAL':12s}: {total:6d} bars")
