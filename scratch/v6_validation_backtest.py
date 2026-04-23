"""
V6 Strategy Quick Validation Backtest
Runs TF, LSB, and SMR on the full 1-year dataset with V6 fixes.
Outputs a concise performance comparison.
"""
import sys
import os
os.environ['PYTHONIOENCODING'] = 'utf-8'
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.append(os.getcwd())

import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timezone

from core.common.types import CandleArray
from backtesting.backtester import PortfolioBacktester
from core.performance_tracker import PerformanceTracker
from strategies.trend_following import TrendFollowingStrategy
from strategies.liquidity_sweep_breakout import LiquiditySweepBreakoutStrategy
from strategies.smart_mean_reversion import SmartMeanReversionStrategy

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("v6_validation")

def load_data(symbol, timeframe):
    path = f"data_cache/{symbol}/{timeframe}.parquet"
    if not os.path.exists(path):
        return None
    df = pd.read_parquet(path)
    return CandleArray(
        time=df['time'].values,
        open=df['open'].values,
        high=df['high'].values,
        low=df['low'].values,
        close=df['close'].values,
        tick_volume=df['tick_volume'].values,
        spread=df['spread'].values,
    )

def run_single_strategy(name, strategy, config, m5, h1, m15, m1):
    bt_config = config.copy()
    bt_config["backtest"] = {"initial_balance_per_strategy": 10000.0, "deterministic": True, "debug_signals": False}
    bt_config["risk_governance"] = bt_config.get("risk_governance", {}).copy()
    bt_config["risk_governance"]["min_tick_density"] = 1
    
    # Re-enable the strategy for this test
    if name in bt_config:
        bt_config[name] = bt_config[name].copy()
        bt_config[name]["enabled"] = True
    
    bt = PortfolioBacktester(bt_config)
    history, equity = bt.run("XAUUSDm", [strategy], m5, h1, m15, m5, m1)
    
    if not history:
        return {"status": "NO_TRADES", "trades": 0}
    
    metrics = PerformanceTracker.calculate_metrics(history, 10000.0, equity)
    return metrics

def main():
    with open("config/config.json") as f:
        config = json.load(f)
    
    # Force-enable all strategies for validation
    for s in ["TrendFollowing", "LiquiditySweepBreakout", "SmartMeanReversion"]:
        if s in config:
            config[s]["enabled"] = True
    
    # Override portfolio allocations for equal-weight test
    config["portfolio_allocations"] = {
        "TrendFollowing": 0.33,
        "LiquiditySweepBreakout": 0.33,
        "SmartMeanReversion": 0.34
    }
    
    print("=" * 70)
    print("  V6-INSIGNIA STRATEGY VALIDATION BACKTEST")
    print("  1-Year XAUUSDm (Apr 2025 - Apr 2026)")
    print("=" * 70)
    
    # Load data
    print("\nLoading data...")
    m5 = load_data("XAUUSDm", "M5")
    m15 = load_data("XAUUSDm", "M15")
    h1 = load_data("XAUUSDm", "H1")
    m1 = load_data("XAUUSDm", "M1")
    
    if m5 is None:
        print("ERROR: Missing M5 data")
        return
    
    print(f"  M5: {len(m5)} bars | M15: {len(m15) if m15 else 0} | H1: {len(h1) if h1 else 0} | M1: {len(m1) if m1 else 0}")
    
    strategies = {
        "TrendFollowing": TrendFollowingStrategy("trendfollowing_v4", config),
        "LiquiditySweepBreakout": LiquiditySweepBreakoutStrategy("liquiditysweepbreakout_v4", config),
        "SmartMeanReversion": SmartMeanReversionStrategy("smartmeanreversion_v4", config),
    }
    
    results = {}
    for name, strategy in strategies.items():
        print(f"\n  Testing {name}...")
        try:
            metrics = run_single_strategy(name, strategy, config, m5, h1, m15, m1)
            results[name] = metrics
            
            if metrics.get("status") == "NO_TRADES":
                print(f"    [--] NO TRADES")
            else:
                pnl = metrics.get("net_profit", 0)
                wr = metrics.get("win_rate", "0%")
                sharpe = metrics.get("sharpe_ratio", 0)
                dd = metrics.get("max_drawdown", "0%")
                pf = metrics.get("profit_factor", 0)
                trades = metrics.get("total_trades", 0)
                icon = "[OK]" if pnl > 0 else "[FAIL]"
                print(f"    {icon} PnL: ${pnl:>8.2f} | Trades: {trades:>3} | WR: {wr:>6} | Sharpe: {sharpe:>5} | DD: {dd:>6} | PF: {pf:>5}")
        except Exception as e:
            print(f"    [ERR] {e}")
            import traceback
            traceback.print_exc()
            results[name] = {"status": "ERROR", "error": str(e)}
    
    # Portfolio test (all 3 combined)
    print(f"\n  Testing Full Portfolio (3-Strategy)...")
    try:
        all_strats = [
            TrendFollowingStrategy("trendfollowing_v4", config),
            LiquiditySweepBreakoutStrategy("liquiditysweepbreakout_v4", config),
            SmartMeanReversionStrategy("smartmeanreversion_v4", config),
        ]
        
        bt_config = config.copy()
        bt_config["backtest"] = {"initial_balance_per_strategy": 10000.0, "deterministic": True}
        bt_config["risk_governance"] = bt_config.get("risk_governance", {}).copy()
        bt_config["risk_governance"]["min_tick_density"] = 1
        
        for s in ["TrendFollowing", "LiquiditySweepBreakout", "SmartMeanReversion"]:
            if s in bt_config:
                bt_config[s] = bt_config[s].copy()
                bt_config[s]["enabled"] = True
        
        bt = PortfolioBacktester(bt_config)
        history, equity = bt.run("XAUUSDm", all_strats, m5, h1, m15, m5, m1)
        
        if history:
            portfolio_metrics = PerformanceTracker.calculate_metrics(history, 30000.0, equity)
            pnl = portfolio_metrics.get("net_profit", 0)
            wr = portfolio_metrics.get("win_rate", "0%")
            sharpe = portfolio_metrics.get("sharpe_ratio", 0)
            dd = portfolio_metrics.get("max_drawdown", "0%")
            pf = portfolio_metrics.get("profit_factor", 0)
            trades = portfolio_metrics.get("total_trades", 0)
            icon = "[OK]" if pnl > 0 else "[FAIL]"
            print(f"    {icon} PnL: ${pnl:>8.2f} | Trades: {trades:>3} | WR: {wr:>6} | Sharpe: {sharpe:>5} | DD: {dd:>6} | PF: {pf:>5}")
            results["Portfolio"] = portfolio_metrics
        else:
            print(f"    [--] NO TRADES")
    except Exception as e:
        print(f"    [ERR] {e}")
        import traceback
        traceback.print_exc()
    
    # Save results
    with open("backtest_results/v6_validation_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n{'='*70}")
    print(f"  Results saved to backtest_results/v6_validation_results.json")
    print(f"{'='*70}")

if __name__ == "__main__":
    main()
