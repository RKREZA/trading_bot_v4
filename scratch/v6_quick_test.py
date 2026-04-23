import json, os, sys
os.environ['PYTHONIOENCODING'] = 'utf-8'
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.append(os.getcwd())
import numpy as np, pandas as pd
import logging
logging.basicConfig(level=logging.WARNING)

from core.common.types import CandleArray
from backtesting.backtester import PortfolioBacktester
from core.performance_tracker import PerformanceTracker
from strategies.trend_following import TrendFollowingStrategy
from strategies.liquidity_sweep_breakout import LiquiditySweepBreakoutStrategy
from strategies.smart_mean_reversion import SmartMeanReversionStrategy

def ld(tf):
    df = pd.read_parquet(f'data_cache/XAUUSDm/{tf}.parquet')
    return CandleArray(time=df['time'].values, open=df['open'].values, high=df['high'].values,
                       low=df['low'].values, close=df['close'].values,
                       tick_volume=df['tick_volume'].values, spread=df['spread'].values)

config = json.load(open('config/config.json'))

# Override for testing
for s in ["TrendFollowing", "LiquiditySweepBreakout", "SmartMeanReversion"]:
    if s in config:
        config[s]["enabled"] = True
config["risk_governance"]["min_tick_density"] = 1

m5, m15, h1, m1 = ld('M5'), ld('M15'), ld('H1'), ld('M1')

strategies_to_test = [
    ("TrendFollowing", TrendFollowingStrategy("trendfollowing_v4", config)),
    ("LiquiditySweepBreakout", LiquiditySweepBreakoutStrategy("liquiditysweepbreakout_v4", config)),
    ("SmartMeanReversion", SmartMeanReversionStrategy("smartmeanreversion_v4", config)),
]

for name, strat in strategies_to_test:
    print(f"\n{'='*60}")
    print(f"  {name} (enabled={strat.enabled})")
    
    cfg = config.copy()
    cfg["portfolio_allocations"] = {name: 1.0}
    # Zero out others
    for other in ["TrendFollowing", "LiquiditySweepBreakout", "SmartMeanReversion"]:
        if other != name:
            cfg["portfolio_allocations"][other] = 0.0
    
    cfg["backtest"] = {"initial_balance_per_strategy": 10000.0, "deterministic": True}
    
    bt = PortfolioBacktester(cfg)
    hist, eq = bt.run("XAUUSDm", [strat], m5, h1, m15, m5, m1)
    
    if hist:
        pnl = sum(t['pnl'] for t in hist)
        metrics = PerformanceTracker.calculate_metrics(hist, 10000.0, eq)
        print(f"  Trades: {len(hist)} | PnL: ${pnl:.2f}")
        print(f"  WR: {metrics.get('win_rate')} | Sharpe: {metrics.get('sharpe_ratio')} | DD: {metrics.get('max_drawdown')} | PF: {metrics.get('profit_factor')}")
    else:
        print(f"  NO TRADES")

# Full portfolio
print(f"\n{'='*60}")
print("  Full Portfolio (3-Strategy)")
cfg = config.copy()
cfg["portfolio_allocations"] = {"TrendFollowing": 0.33, "LiquiditySweepBreakout": 0.33, "SmartMeanReversion": 0.34}
cfg["backtest"] = {"initial_balance_per_strategy": 10000.0, "deterministic": True}

all_strats = [
    TrendFollowingStrategy("trendfollowing_v4", config),
    LiquiditySweepBreakoutStrategy("liquiditysweepbreakout_v4", config),
    SmartMeanReversionStrategy("smartmeanreversion_v4", config),
]

bt = PortfolioBacktester(cfg)
hist, eq = bt.run("XAUUSDm", all_strats, m5, h1, m15, m5, m1)

if hist:
    pnl = sum(t['pnl'] for t in hist)
    metrics = PerformanceTracker.calculate_metrics(hist, 30000.0, eq)
    print(f"  Trades: {len(hist)} | PnL: ${pnl:.2f}")
    print(f"  WR: {metrics.get('win_rate')} | Sharpe: {metrics.get('sharpe_ratio')} | DD: {metrics.get('max_drawdown')} | PF: {metrics.get('profit_factor')}")
else:
    print(f"  NO TRADES")
