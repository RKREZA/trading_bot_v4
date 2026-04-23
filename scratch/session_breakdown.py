import sys
import os
import json
from datetime import datetime, timezone

sys.path.append(os.getcwd())
from comprehensive_backtest import ComprehensiveBacktestSuite
from backtesting.backtester import PortfolioBacktester
from strategies import create_strategy

def get_session(dt):
    hour = dt.hour
    if 0 <= hour < 8:
        return "TOKYO"
    elif 8 <= hour < 13:
        return "LONDON"
    elif 13 <= hour < 17:
        return "LONDON/NY"
    elif 17 <= hour < 21:
        return "NEW_YORK"
    else:
        return "SYDNEY_DEAD_ZONE"

def run_session_analysis():
    print("Initializing session analysis...")
    suite = ComprehensiveBacktestSuite()
    config = suite._create_base_config()
    
    # Enable all strategies for analysis
    config["portfolio_allocations"] = {
        "TrendFollowing": 0.33,
        "LiquiditySweepBreakout": 0.33,
        "SmartMeanReversion": 0.33
    }
    
    # Reduce max spread so we actually get trades
    config["max_spread_points"] = 1500
    
    st_names = ["SmartMeanReversion", "LiquiditySweepBreakout", "TrendFollowing"]
    strategy_objs = []
    for name in st_names:
        strategy_objs.append(create_strategy(f"{name.lower()}_v4", name.upper(), config))
        
    print("Loading data...")
    m5 = suite.load_real_data(timeframe="M5", n_bars=30000)
    m15 = suite.load_real_data(timeframe="M15", n_bars=10000)
    h1 = suite.load_real_data(timeframe="H1", n_bars=2500)
    m1 = suite.load_real_data(timeframe="M1", n_bars=150000)
    
    m5, m15, h1, m1 = suite._align_timeframes(m1, m5, m15, h1)
    
    bt = PortfolioBacktester(config)
    print("Running backtest engine... (this might take a few seconds)")
    history, equity = bt.run("XAUUSDm", strategy_objs, m5, h1, m15, m5, m1)
    
    if not history:
        print("No trades executed.")
        return
        
    print(f"Total Trades: {len(history)}")
    
    # Aggregation dicts
    stats = {}
    
    for tr in history:
        # trade keys typically: entry_time, exit_time, pnl, strategy, direction, etc.
        strat = tr.get("strategy_id", "Unknown").replace("_v4", "").upper()
        sess = tr.get("session", "UNKNOWN_SESSION")
        pnl = tr.get("pnl", 0)
        win = 1 if pnl > 0 else 0
        
        if strat not in stats:
            stats[strat] = {}
            
        if sess not in stats[strat]:
            stats[strat][sess] = {"trades": 0, "wins": 0, "pnl": 0.0}
            
        stats[strat][sess]["trades"] += 1
        stats[strat][sess]["wins"] += win
        stats[strat][sess]["pnl"] += pnl

    # Display results
    print("\n" + "="*70)
    print(" SESSION-WISE & STRATEGY-WISE BACKTEST RESULTS ")
    print("="*70)
    
    for strat, session_data in stats.items():
        print(f"\n[{strat}]")
        print(f"{'Session':<18} | {'Trades':<8} | {'Win %':<8} | {'Net PnL ($)':<10}")
        print("-" * 55)
        
        strat_trades = sum(s["trades"] for s in session_data.values())
        strat_pnl = sum(s["pnl"] for s in session_data.values())
        
        for sess, data in sorted(session_data.items(), key=lambda x: -x[1]['pnl']):
            trades = data["trades"]
            win_rate = (data["wins"] / trades * 100) if trades > 0 else 0
            pnl = data["pnl"]
            print(f"{sess:<18} | {trades:<8} | {win_rate:>6.1f}% | {pnl:>10.2f}")
            
        print("-" * 55)
        print(f"{'TOTAL':<18} | {strat_trades:<8} | {'-':<8} | {strat_pnl:>10.2f}")

if __name__ == '__main__':
    run_session_analysis()
