import json, sys, os
sys.path.append(os.getcwd())
from dotenv import load_dotenv
load_dotenv()
from datetime import datetime, timezone, timedelta
from core.data.manager import DataManager
from core.connection import MT5Connection
from core.indicator_engine import IndicatorEngine
from core.regime_detector import RegimeDetector
from core.regime_gater import RegimeGater
from core.session_detector import SessionDetector
from core.base_strategy import MarketData
from strategies import create_strategy
import numpy as np

with open("config.json") as f: config = json.load(f)
conn = MT5Connection()
conn.connect()
dm = DataManager(config)

end_dt = datetime.now(timezone.utc)
start_dt = end_dt - timedelta(days=30)
m5 = dm.prepare_data("XAUUSDm", "M5", start_dt)
m15 = dm.prepare_data("XAUUSDm", "M15", start_dt)
h1 = dm.prepare_data("XAUUSDm", "H1", start_dt)

m5._indicators = IndicatorEngine.precalculate_all("XAUUSDm", "M5", m5)
m15._indicators = IndicatorEngine.precalculate_all("XAUUSDm", "M15", m15)
h1._indicators = IndicatorEngine.precalculate_all("XAUUSDm", "H1", h1)

strat = create_strategy("TrendFollowing", "TRENDFOLLOWING", config)
rd = RegimeDetector()

from collections import Counter
reasons = Counter()
regime_blocks = 0

for i in range(200, len(m5.time), 5):  # Sample every 5 bars
    m5.set_limit(i)
    dt = datetime.fromtimestamp(int(m5.time[i-1]), tz=timezone.utc)
    
    # Check regime gating
    regime_info = rd.detect(m5)
    regime = regime_info.market_type
    if not RegimeGater.is_strategy_allowed("TrendFollowingStrategy", regime):
        regime_blocks += 1
        continue
    
    # Build MarketData
    h1_idx = min(len(h1.time)-1, max(30, np.searchsorted(h1.time, m5.time[i-1], side='right')))
    h1.set_limit(h1_idx)
    m15_idx = min(len(m15.time)-1, max(100, np.searchsorted(m15.time, m5.time[i-1], side='right')))
    m15.set_limit(m15_idx)
    
    session = SessionDetector.get_session(dt, 0)
    md = MarketData(
        symbol="XAUUSDm", htf_candles=h1, m15_candles=m15, m5_candles=m5,
        d1_candles=None, current_price=float(m5.close[i-1]),
        bid=float(m5.close[i-1]), ask=float(m5.close[i-1]),
        spread=0, session=session, timestamp=dt
    )
    
    sig = strat.generate_signal(md)
    if sig:
        reasons["SIGNAL_GENERATED"] += 1
    else:
        reason = getattr(strat, "last_rejection_reason", "Unknown")
        reasons[reason] += 1

total = sum(reasons.values())
print(f"\n=== TRENDFOLLOWING DIAGNOSTIC (30 days, sampled every 5 bars) ===")
print(f"Regime Blocks: {regime_blocks}")
print(f"Strategy Evaluations: {total}")
print(f"\nTop Rejection Reasons:")
for r, c in reasons.most_common(15):
    print(f"  {r:50s}: {c:5d} ({c/total*100:.1f}%)")
