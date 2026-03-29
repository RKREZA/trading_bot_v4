import json
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from dotenv import load_dotenv
from core.backtester import BacktestEngine
from core.strategy_engine import StrategyEngine
from core.data_fetcher import DataFetcher
from core.connection import MT5Connection
from core.performance import PerformanceMetrics

def analyze_sessions(df):
    """Group trades by UTC session."""
    def get_session(hour):
        if 8 <= hour < 14: return "LONDON"
        if 14 <= hour < 17: return "LONDON/NY"
        if 17 <= hour < 22: return "NEW_YORK"
        return "TOKYO"
    
    df['session'] = df['time'].dt.hour.apply(get_session)
    summary = df.groupby('session').agg({
        'pnl': ['sum', 'count', 'mean'],
        'result': lambda x: (x == 'TP').mean() * 100
    })
    summary.columns = ['Total P&L', 'Count', 'Avg P&L', 'Win Rate (%)']
    return summary

def analyze_regimes(df):
    """Group trades by market regime."""
    summary = df.groupby('regime').agg({
        'pnl': ['sum', 'count', 'mean'],
        'result': lambda x: (x == 'TP').mean() * 100
    })
    summary.columns = ['Total P&L', 'Count', 'Avg P&L', 'Win Rate (%)']
    return summary

def main():
    load_dotenv()
    with open("config.json") as f:
        config = json.load(f)
    
    symbol = config["symbol"]
    conn = MT5Connection()
    if not conn.connect():
        print("Failed to connect")
        return

    fetcher = DataFetcher()
    # Fetch from config or defaults
    bt_candles = config.get("backtest", {}).get("candles", {"H4": 600, "M30": 4800, "M5": 9600})
    h4 = fetcher.fetch_candles(symbol, "H4", bt_candles.get("H4", 2500))
    h1 = fetcher.fetch_candles(symbol, "H1", bt_candles.get("H1", 9000))
    m30 = fetcher.fetch_candles(symbol, "M30", bt_candles.get("M30", 18000))
    m5 = fetcher.fetch_candles(symbol, "M5", bt_candles.get("M5", 54000))
    d1 = fetcher.fetch_candles(symbol, "D1", bt_candles.get("D1", 500))
    
    conn.disconnect()

    strategy = StrategyEngine(config)
    engine = BacktestEngine(config, strategy)
    
    print(f"Running research backtest for {symbol}...")
    results = engine.run(symbol, h4, h1, m30, m5, d1, quiet=True)
    
    trades = results['trades']
    df = pd.DataFrame(trades)
    df['time'] = pd.to_datetime(df['time'])
    
    print("\n" + "="*50)
    print("SESSION ANALYSIS")
    print("="*50)
    print(analyze_sessions(df))
    
    print("\n" + "="*50)
    print("REGIME ANALYSIS")
    print("="*50)
    print(analyze_regimes(df))
    
    print("\n" + "="*50)
    print("GLOBAL STATS")
    print("="*50)
    for k, v in results.items():
        if k != 'trades' and k != 'equity_curve':
            print(f"{k}: {v}")

if __name__ == "__main__":
    main()
