"""
Deep signal diagnostic: trace exactly why Sniper fires so rarely.
Walks through backtest loop manually for first 5000 candles.
"""
import json, sys, os, numpy as np
from datetime import datetime, timezone
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.data_fetcher import DataFetcher
from core.connection import MT5Connection
from core.strategy_engine import StrategyEngine
from core.base_strategy import MarketData
from strategies import create_strategy

with open("config.json") as f:
    config = json.load(f)

conn = MT5Connection(); conn.config = config; conn.connect()
fetcher = DataFetcher()
dt_from = datetime(2025, 10, 1, tzinfo=timezone.utc)
dt_to   = datetime(2026, 3, 31, tzinfo=timezone.utc)

h1  = fetcher.fetch_candles_range("XAUUSDm","H1", dt_from, dt_to)
m15 = fetcher.fetch_candles_range("XAUUSDm","M15",dt_from, dt_to)
m5  = fetcher.fetch_candles_range("XAUUSDm","M5", dt_from, dt_to)
d1  = fetcher.fetch_candles_range("XAUUSDm","D1", dt_from, dt_to)
print("M5 candles: " + str(len(m5)))

# Build Sniper strategy
s_cfg  = next(s for s in config["strategies"] if s["id"] == "sniper_v1")
merged = {**config, **s_cfg, "research_mode": False}  # normal mode (as backtester does)
sniper = create_strategy("sniper_v1", "SNIPER", merged)

# Preprocess full history
ctx    = sniper.preprocess(h1, m15, m5, d1)
m5_meta = ctx.get("m5", []) if ctx else []
print("Preprocessed rows: " + str(len(m5_meta)))

# Count signal-level hits
counts = {
    "total_candles": 0,
    "session_skip": 0,
    "bias_neutral_tokyo": 0,
    "cooldown_block": 0,
    "consec_loss_block": 0,
    "session_traded_block": 0,
    "t1_hit": 0,
    "t2_hit": 0,
    "t3_hit": 0,
    "no_signal": 0,
    "conf_fail": 0,
    "final_signal": 0,
}

ACTIVE_SESSIONS = {"LONDON","NEW_YORK","LONDON/NY","TOKYO"}

# Replicate what backtester does for each candle
m5_times  = m5.time
m5_closes = m5.close

for i in range(50, min(len(m5_times), 35000)):
    t = m5_times[i]
    candle_dt = datetime.fromtimestamp(float(t), tz=timezone.utc)

    from core.strategy_engine import StrategyEngine as SE
    session = SE.get_session_from_hour(None, candle_dt.hour, 3)

    counts["total_candles"] += 1

    # Skip non-trading sessions
    if session not in ACTIVE_SESSIONS:
        counts["session_skip"] += 1
        continue

    meta_i = m5_meta[i] if i < len(m5_meta) else {}

    market_data = MarketData(
        symbol="XAUUSDm",
        htf_candles=h1[:np.searchsorted(h1.time, t, side='right')],
        m15_candles=m15[:np.searchsorted(m15.time, t, side='right')],
        m5_candles=m5[:i + 1],
        d1_candles=d1,
        current_price=float(m5_closes[i]),
        session=session,
        timestamp=candle_dt,
        preprocessed=meta_i,
    )

    sig = sniper.generate_signal(market_data)
    if sig:
        counts["final_signal"] += 1
    else:
        counts["no_signal"] += 1

# Also analyze the meta directly
bias_counts  = {}
rej_bull_cnt = 0
vol_ok_cnt   = 0
all_cnt      = 0

for i, row in enumerate(m5_meta[50:35000], start=50):
    t         = m5_times[i] if i < len(m5_times) else 0
    candle_dt = datetime.fromtimestamp(float(t), tz=timezone.utc)
    session   = SE.get_session_from_hour(None, candle_dt.hour, 3)
    if session not in ACTIVE_SESSIONS:
        continue

    all_cnt += 1
    b = row.get("m_bias", "NEUTRAL")
    bias_counts[b] = bias_counts.get(b, 0) + 1

    if row.get("rej_bull"):
        rej_bull_cnt += 1

    # Compute vol_ok
    vol_sma = row.get("vol_sma", 1.0) or 1.0
    if i < len(m5.tick_volume):
        cur_vol = float(m5.tick_volume[i])
        if cur_vol > vol_sma * 1.15:
            vol_ok_cnt += 1

print("\n--- Backtest Signal Walk ---")
for k, v in counts.items():
    print("  " + k + ": " + str(v))

print("\n--- Active-Session Meta Analysis ---")
print("  Total active-session candles: " + str(all_cnt))
for b, c in sorted(bias_counts.items()):
    print("  Bias " + b + ": " + str(c) + " (" + str(round(c/max(all_cnt,1)*100,1)) + "%)")
print("  rej_bull (active sessions): " + str(rej_bull_cnt))
print("  vol_ok (1.15x, active sessions): " + str(vol_ok_cnt))

# T2 theoretical: BULLISH + rej_bull + vol_ok
bull_cnt = bias_counts.get("BULLISH", 0)
# estimate T2: assume rej_bull and vol_ok are ~independent within BULLISH candles
t2_est = round(rej_bull_cnt * (bull_cnt / max(all_cnt, 1)) * (vol_ok_cnt / max(all_cnt, 1)) * all_cnt, 0)
print("  T2 BUY estimate (BULLISH*rej_bull*vol_ok): ~" + str(int(t2_est)))

# Tick volume analysis
tv = m5.tick_volume[:35000]
print("\n--- Tick Volume Stats ---")
print("  Volume == 0: " + str(int((tv == 0).sum())))
print("  Volume mean: " + str(round(float(tv[tv > 0].mean()), 1)) if (tv > 0).any() else "  No volume data")
print("  Volume > vol_sma*1.15 (sample): " + str(vol_ok_cnt) + "/" + str(all_cnt))

conn.disconnect()
