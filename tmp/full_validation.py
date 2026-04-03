"""Run full validation: backtest + walk-forward + Monte Carlo, print clean results."""
import json, sys, os
from datetime import datetime, timezone
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.data_fetcher import DataFetcher
from core.connection import MT5Connection
from core.backtester import MultiStrategyBacktestEngine
from core.walk_forward import WalkForwardValidator
from core.monte_carlo import MonteCarlo
from strategies import create_strategy

with open("config.json") as f:
    config = json.load(f)

conn = MT5Connection()
conn.config = config
conn.connect()

fetcher = DataFetcher()
dt_from = datetime(2025, 10, 1, tzinfo=timezone.utc)
dt_to   = datetime(2026, 3, 31, tzinfo=timezone.utc)

print("Fetching 6 months of data...")
h1  = fetcher.fetch_candles_range("XAUUSDm", "H1",  dt_from, dt_to)
m15 = fetcher.fetch_candles_range("XAUUSDm", "M15", dt_from, dt_to)
m5  = fetcher.fetch_candles_range("XAUUSDm", "M5",  dt_from, dt_to)
d1  = fetcher.fetch_candles_range("XAUUSDm", "D1",  dt_from, dt_to)
print("Candles: H1=" + str(len(h1)) + " M15=" + str(len(m15)) + " M5=" + str(len(m5)))

strats = [
    create_strategy(s["id"], s["type"], {**config, **s})
    for s in config["strategies"] if s.get("enabled", True)
]

# ── 1. Full backtest with all cost deductions ──────────────────────────────
print("\nRunning full 6-month backtest (with commissions, swap, slippage)...")
engine  = MultiStrategyBacktestEngine(config, strats)
results = engine.run("XAUUSDm", h1, m15, m5, d1, quiet=True)

print("\n" + "="*55)
print("  HONEST BACKTEST — Oct 2025 → Mar 2026 (with costs)")
print("="*55)

for sid, res in results.items():
    if sid in ("portfolio", "walk_forward"):
        continue
    trades = res.get("trades", [])
    wins   = [t for t in trades if t.get("pnl", 0) > 0]
    losses = [t for t in trades if t.get("pnl", 0) <= 0]
    avg_w  = sum(t["pnl"] for t in wins)   / len(wins)   if wins   else 0
    avg_l  = sum(t["pnl"] for t in losses) / len(losses) if losses else 0
    total_comm = sum(t.get("commission", 0) for t in trades)
    total_swap = sum(t.get("swap", 0) for t in trades)

    print("\n  Strategy:        " + sid)
    print("  Net Profit (after costs): $" + str(round(res.get("net_profit", 0), 2)))
    print("  Final Balance:   $" + str(round(res.get("final_balance", 1000), 2)))
    print("  Profit Factor:   " + str(res.get("profit_factor", 0)))
    print("  Win Rate:        " + str(res.get("win_rate", 0)) + "%")
    print("  Total Trades:    " + str(res.get("total_trades", 0)))
    print("  Max Drawdown:    " + str(res.get("max_drawdown_pct", 0)) + "%")
    print("  Sharpe Ratio:    " + str(res.get("sharpe_ratio", 0)))
    print("  Expectancy:      $" + str(res.get("expectancy", 0)))
    print("  Avg Win:         $" + str(round(avg_w, 2)))
    print("  Avg Loss:        $" + str(round(avg_l, 2)))
    print("  Total Commission:$" + str(round(total_comm, 2)))
    print("  Total Swap:      $" + str(round(total_swap, 2)))

    # ── Monte Carlo ──────────────────────────────────────────────────────
    if trades:
        mc = MonteCarlo(trades, initial_balance=1000.0, n_simulations=2000)
        mc_res = mc.run()
        MonteCarlo.print_report(mc_res, sid)

if "portfolio" in results:
    r = results["portfolio"]
    print("\n" + "="*55)
    print("  COMBINED PORTFOLIO SUMMARY")
    print("="*55)
    print("  Net Profit:    $" + str(r.get("net_profit", 0)))
    print("  Total Trades:  " + str(r.get("total_trades", 0)))
    print("  Win Rate:      " + str(r.get("win_rate", 0)) + "%")
    print("  Max Drawdown:  " + str(r.get("max_drawdown_pct", 0)) + "%")

# ── 2. Walk-Forward Validation ────────────────────────────────────────────
print("\n\nRunning Walk-Forward Validation (70% IS / 30% OOS)...")
# Fresh strategy instances for WFO
strats2 = [
    create_strategy(s["id"], s["type"], {**config, **s})
    for s in config["strategies"] if s.get("enabled", True)
]
wf = WalkForwardValidator(config, strats2, MultiStrategyBacktestEngine, is_pct=0.70)
wf.run("XAUUSDm", h1, m15, m5, d1, quiet=False)

conn.disconnect()
print("\nDone.")
