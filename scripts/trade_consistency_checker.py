import os
import sys
import logging
import json
import pandas as pd
from datetime import datetime, timedelta, timezone
import MetaTrader5 as mt5

# Add the project root to sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from core.data_fetcher import DataFetcher
from core.strategy_engine import StrategyEngine
from core.backtester import BacktestEngine

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)-8s] %(name)s: %(message)s')
logger = logging.getLogger("consistency_checker")

def run_consistency_analysis(symbol="XAUUSDm", days=2):
    if not mt5.initialize():
        print("Failed to initialize MT5")
        return

    # 1. Load Config
    config_path = os.path.join(project_root, "config.json")
    with open(config_path, "r") as f:
        config = json.load(f)

    # 2. Setup Time Range
    # Yesterday and Today
    now = datetime.now(timezone.utc)
    # Start of yesterday (server time usually offset, but we'll use UTC/Local as proxy)
    start_date = (now - timedelta(days=days)).replace(hour=0, minute=0, second=0, microsecond=0)
    end_date = now

    print(f"\n--- ANALYZING PERIOD: {start_date} to {end_date} ---")

    # 3. Fetch Data for Backtest
    fetcher = DataFetcher()
    
    # We need a buffer for indicators (EMA20, EMA50, etc.)
    # 500 candles buffer for M5, maybe 100 for H1/H4
    buffer_days = 30
    fetch_start = start_date - timedelta(days=buffer_days)
    
    print(f"Fetching historical candles (Buffer from {fetch_start})...")
    m5_candles = fetcher.fetch_candles_range(symbol, "M5", fetch_start, end_date)
    m30_candles = fetcher.fetch_candles_range(symbol, "M30", fetch_start, end_date)
    h1_candles = fetcher.fetch_candles_range(symbol, "H1", fetch_start, end_date)
    h4_candles = fetcher.fetch_candles_range(symbol, "H4", fetch_start, end_date)
    d1_candles = fetcher.fetch_candles_range(symbol, "D1", fetch_start, end_date)
    
    if not m5_candles:
        print("Failed to fetch M5 candles. MT5 might be closed or history unavailable.")
        mt5.shutdown()
        return

    # 4. Run Backtest
    print(f"Running Backtest for {symbol}...")
    strategy = StrategyEngine(config)
    
    # Monkey-patch analyze to see rejections
    original_analyze = strategy.analyze
    rejections = {} # reason -> count
    
    def debug_analyze(*args, **kwargs):
        res, trend, reg = original_analyze(*args, **kwargs)
        if res is None:
            rejections[trend] = rejections.get(trend, 0) + 1
        return res, trend, reg
    
    strategy.analyze = debug_analyze
    
    backtester = BacktestEngine(config, strategy)
    
    # Backtester.run returns performance metrics including 'trades'
    results = backtester.run(symbol, h4_candles, h1_candles, m30_candles, m5_candles, d1_candles, quiet=True)
    bt_trades = results.get('trades', [])
    
    # Filter BT trades to the target window (start_date to end_date)
    bt_trades = [t for t in bt_trades if t['time'] >= start_date]
    print(f"Backtest generated {len(bt_trades)} signals in the target window.")

    # 5. Fetch Live Trades
    print("Fetching live MT5 history...")
    deals = mt5.history_deals_get(start_date, end_date)
    live_trades = []
    if deals:
        # We need to pair entry/exit deals into "trades"
        # For simplicity, we'll look at the OUT deals
        for d in deals:
            if d.entry == mt5.DEAL_ENTRY_OUT:
                # Find the matching entry deal to get the entry time and price
                pos_id = d.position_id
                # Fetch all deals for this position
                pos_deals = mt5.history_deals_get(position=pos_id)
                if pos_deals:
                    entry_deal = next((pd for pd in pos_deals if pd.entry == mt5.DEAL_ENTRY_IN), None)
                    if entry_deal:
                        live_trades.append({
                            "ticket": entry_deal.ticket,
                            "time": datetime.fromtimestamp(entry_deal.time, tz=timezone.utc),
                            "exit_time": datetime.fromtimestamp(d.time, tz=timezone.utc),
                            "direction": "BUY" if entry_deal.type == mt5.ORDER_TYPE_BUY else "SELL",
                            "entry": entry_deal.price,
                            "exit": d.price,
                            "lot": d.volume,
                            "pnl": d.profit,
                            "reason": d.reason,
                            "symbol": d.symbol
                        })
    
    print(f"Live history contains {len(live_trades)} completed trades in the target window.")

    # 6. Comparison
    print("\n--- COMPARISON REPORT ---")
    
    report = []
    
    # Match BT signals to Live executions
    # A match is defined by: same symbol, same direction, time within 5 minutes
    matched_live_indices = set()
    
    for bt in bt_trades:
        match = None
        for i, lt in enumerate(live_trades):
            if i in matched_live_indices: continue
            if lt['symbol'] == symbol and lt['direction'] == bt['direction']:
                time_diff = abs((lt['time'] - bt['time']).total_seconds())
                if time_diff < 300: # 5 minute window
                    match = lt
                    matched_live_indices.add(i)
                    break
        
        if match:
            slippage = match['entry'] - bt['entry'] if bt['direction'] == "BUY" else bt['entry'] - match['entry']
            report.append({
                "Status": "MATCHED",
                "Time": bt['time'].strftime('%m-%d %H:%M'),
                "Dir": bt['direction'],
                "BT_Entry": bt['entry'],
                "Live_Entry": match['entry'],
                "Slippage": round(slippage, 5),
                "BT_PnL": bt['pnl'],
                "Live_PnL": match['pnl'],
                "Discrepancy": round(match['pnl'] - bt['pnl'], 2)
            })
        else:
            report.append({
                "Status": "MISSING_LIVE",
                "Time": bt['time'].strftime('%m-%d %H:%M'),
                "Dir": bt['direction'],
                "BT_Entry": bt['entry'],
                "Live_Entry": "-",
                "Slippage": "-",
                "BT_PnL": bt['pnl'],
                "Live_PnL": "-",
                "Discrepancy": "-"
            })

    # Unmatched Live trades (Ghost Trades)
    for i, lt in enumerate(live_trades):
        if i not in matched_live_indices:
            report.append({
                "Status": "GHOST_LIVE",
                "Time": lt['time'].strftime('%m-%d %H:%M'),
                "Dir": lt['direction'],
                "BT_Entry": "-",
                "Live_Entry": lt['entry'],
                "Slippage": "-",
                "BT_PnL": "-",
                "Live_PnL": lt['pnl'],
                "Discrepancy": "-"
            })

    df_report = pd.DataFrame(report)
    if not df_report.empty:
        print(df_report.to_string(index=False))
        
        # Summary Stats
        matches = df_report[df_report['Status'] == "MATCHED"]
        missing = df_report[df_report['Status'] == "MISSING_LIVE"]
        ghosts = df_report[df_report['Status'] == "GHOST_LIVE"]
        
        print("\n--- STATISTICS ---")
        print(f"Total Backtest Signals: {len(bt_trades)}")
        print(f"Total Live Executions: {len(live_trades)}")
        print(f"Successful Matches: {len(matches)}")
        print(f"Missing in Live: {len(missing)} (Signals the bot IGNORED)")
        print(f"Ghost Executions: {len(ghosts)} (Trades not in strategy logic)")
        
        print("\n--- REJECTION DIAGNOSTICS (Backtest) ---")
        for reason, count in rejections.items():
            print(f"  - {reason}: {count}")
        
        if not matches.empty:
            avg_slip = matches[matches['Slippage'] != "-"]['Slippage'].astype(float).mean()
            print(f"Average Entry Slippage: {avg_slip:.5f}")
    else:
        print("No trades to compare.")

    mt5.shutdown()

if __name__ == "__main__":
    # You can change 'days' to analyze a longer period
    run_consistency_analysis(symbol="XAUUSDm", days=2)
