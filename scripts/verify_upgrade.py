import json
import logging
from core.strategy_engine import StrategyEngine
from core.backtester import BacktestEngine
from core.connection import MT5Connection
from core.data_fetcher import DataFetcher

logging.basicConfig(level=logging.INFO)

def run_verify():
    with open("config.json", "r") as f:
        config = json.load(f)
    
    symbol = "XAUUSDm"
    conn = MT5Connection()
    conn.config = config
    if not conn.connect():
        print("MT5 not connected, skipping live data fetch.")
        return

    fetcher = DataFetcher()
    # Fetch small amount for verification
    h4 = fetcher.fetch_candles(symbol, "H4", 100)
    h1 = fetcher.fetch_candles(symbol, "H1", 200)
    m30 = fetcher.fetch_candles(symbol, "M30", 500)
    m5 = fetcher.fetch_candles(symbol, "M5", 1000)
    d1 = fetcher.fetch_candles(symbol, "D1", 50)
    conn.disconnect()

    if not all([h4, h1, m30, m5, d1]):
        print("Data fetch failed.")
        return

    strategy = StrategyEngine(config)
    backtester = BacktestEngine(config, strategy)
    
    print("Running verification backtest...")
    results = backtester.run(symbol, h4, h1, m30, m5, d1, quiet=False)
    
    print("\n--- Backtest Results ---")
    print(f"Net Profit: ${results['net_profit']:.2f}")
    print(f"Win Rate: {results['win_rate']:.2f}%")
    print(f"Profit Factor: {results['profit_factor']:.2f}")
    
    # Save results for analyze_results.py
    with open("backtest_results.json", "w") as f:
        # Filter out objects that can't be serialized if any (like datetime)
        # PerformanceMetrics already returns clean dict mostly
        serializable_results = results.copy()
        if 'trades' in serializable_results:
            for t in serializable_results['trades']:
                t['time'] = str(t['time'])
                t['exit_time'] = str(t['exit_time'])
        json.dump(serializable_results, f, indent=4)
        
    print("\nResults saved to backtest_results.json. Testing reporting...")
    import subprocess
    subprocess.run(["python", "analyze_results.py"])

if __name__ == "__main__":
    run_verify()
