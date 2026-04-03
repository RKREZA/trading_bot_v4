"""
Diagnostic: analyze what the preprocessor produces to understand
why sniper fires so rarely and what inputs we can rely on.
"""
import json, sys, os
from datetime import datetime, timezone
from dotenv import load_dotenv
import numpy as np

load_dotenv()
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.data_fetcher import DataFetcher
from core.connection import MT5Connection
from core.strategy_engine import StrategyEngine

with open("config.json") as f:
    config = json.load(f)

conn = MT5Connection()
conn.config = config
conn.connect()

fetcher = DataFetcher()
dt_from = datetime(2025, 10, 1, tzinfo=timezone.utc)
dt_to   = datetime(2026, 3, 31, tzinfo=timezone.utc)

h1  = fetcher.fetch_candles_range("XAUUSDm", "H1",  dt_from, dt_to)
m15 = fetcher.fetch_candles_range("XAUUSDm", "M15", dt_from, dt_to)
m5  = fetcher.fetch_candles_range("XAUUSDm", "M5",  dt_from, dt_to)
d1  = fetcher.fetch_candles_range("XAUUSDm", "D1",  dt_from, dt_to)

print("Candles: H1=" + str(len(h1)) + " M15=" + str(len(m15)) + " M5=" + str(len(m5)))

engine = StrategyEngine(config, silent=True)
result = engine.preprocess_history(h1, m15, m5, m5)
rows   = result.get("m5", [])

print("Total preprocessed rows: " + str(len(rows)))

# Count how often each signal fires
n = len(rows)
bias_bull   = sum(1 for r in rows if r["m_bias"] == "BULLISH")
bias_bear   = sum(1 for r in rows if r["m_bias"] == "BEARISH")
bias_neu    = sum(1 for r in rows if r["m_bias"] == "NEUTRAL")
rej_bull    = sum(1 for r in rows if r["rej_bull"])
rej_bear    = sum(1 for r in rows if r["rej_bear"])
sweep_bull  = sum(1 for r in rows if r["sweep_bull"])
sweep_bear  = sum(1 for r in rows if r["sweep_bear"])
htf_demand  = sum(1 for r in rows if r["in_htf_demand"])
htf_supply  = sum(1 for r in rows if r["in_htf_supply"])

# Combinations
t1_buy  = sum(1 for r in rows if r["sweep_bull"] and r["rej_bull"])
t1_sell = sum(1 for r in rows if r["sweep_bear"] and r["rej_bear"])
t2_buy  = sum(1 for r in rows if r["rej_bull"] and r["m_bias"] in ("BULLISH","NEUTRAL"))
t2_sell = sum(1 for r in rows if r["rej_bear"] and r["m_bias"] in ("BEARISH","NEUTRAL"))

smc_buy  = sum(1 for r in rows if r["in_htf_demand"] and r["rej_bull"])
smc_sell = sum(1 for r in rows if r["in_htf_supply"] and r["rej_bear"])

print("\n--- M5 Preprocessor Signal Frequency (out of " + str(n) + " candles) ---")
print("Bias BULLISH:  " + str(bias_bull) + "  (" + str(round(bias_bull/n*100,1)) + "%)")
print("Bias BEARISH:  " + str(bias_bear) + "  (" + str(round(bias_bear/n*100,1)) + "%)")
print("Bias NEUTRAL:  " + str(bias_neu)  + "  (" + str(round(bias_neu/n*100,1))  + "%)")
print("Rejection Bull:" + str(rej_bull)  + "  (" + str(round(rej_bull/n*100,1))  + "%)")
print("Rejection Bear:" + str(rej_bear)  + "  (" + str(round(rej_bear/n*100,1))  + "%)")
print("Sweep Bull:    " + str(sweep_bull) + "  (" + str(round(sweep_bull/n*100,1)) + "%)")
print("Sweep Bear:    " + str(sweep_bear) + "  (" + str(round(sweep_bear/n*100,1)) + "%)")
print("HTF Demand:    " + str(htf_demand) + "  (" + str(round(htf_demand/n*100,1)) + "%)")
print("HTF Supply:    " + str(htf_supply) + "  (" + str(round(htf_supply/n*100,1)) + "%)")
print("\n--- Combined Signal Opportunities ---")
print("Sniper T1 BUY  (sweep+rej):            " + str(t1_buy))
print("Sniper T1 SELL (sweep+rej):            " + str(t1_sell))
print("Sniper T2 BUY  (rej+bias):             " + str(t2_buy))
print("Sniper T2 SELL (rej+bias):             " + str(t2_sell))
print("SMC BUY  (htf_demand+rej_bull):        " + str(smc_buy))
print("SMC SELL (htf_supply+rej_bear):        " + str(smc_sell))

# Rejection candle thresholds -- what % with 55% vs 65% wick threshold
def count_rej(rows_in, wick_thresh, body_thresh):
    b, r2 = 0, 0
    for row in rows_in:
        pass  # we need raw candles for this
    return 0

# Show vol_sma stats
vols = [r.get("vol_sma", 0) for r in rows if r.get("vol_sma", 0) > 0]
if vols:
    print("\n--- Volume SMA Stats ---")
    print("Mean vol_sma: " + str(round(np.mean(vols),1)))
    print("Min  vol_sma: " + str(round(np.min(vols),1)))

conn.disconnect()
