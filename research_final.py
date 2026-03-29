import sys
import os
import json

def load_config(path):
    with open(path, 'r') as f:
        return json.load(f)

def main():
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except:
        pass
        
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", type=str, default="XAUUSDm")
    args = parser.parse_args()
    print(f"SYMBOL: {args.symbol}")

    project_root = os.path.dirname(os.path.abspath(__file__))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    from core.connection import MT5Connection
    from core.data_fetcher import DataFetcher
    from core.strategy_engine import StrategyEngine
    from core.walk_forward import WalkForwardValidation

    conn = MT5Connection()
    if not conn.connect():
        print("MT5 FAIL")
        return
    
    fetcher = DataFetcher()
    h4 = fetcher.fetch_candles(args.symbol, "H4", 300)
    h1 = fetcher.fetch_candles(args.symbol, "H1", 600)
    m30 = fetcher.fetch_candles(args.symbol, "M30", 1000)
    m5 = fetcher.fetch_candles(args.symbol, "M5", 2000)
    d1 = fetcher.fetch_candles(args.symbol, "D1", 200)
    conn.disconnect()

    if not all([h4, h1, m30, m5, d1]):
        print("DATA FAIL")
        return

    config = load_config("config/backtest_config.json")
    strat_config = load_config("config.json")
    
    strategy = StrategyEngine(strat_config)
    wf = WalkForwardValidation(strat_config, strategy)
    
    print("RUNNING WFV (1wk train, 2d test)...")
    results = wf.run_validation(args.symbol, h4, h1, m30, m5, d1, train_days=7, test_days=2)
    
    with open("wf_results.json", "w") as f:
        json.dump(results, f, indent=4, default=str)
    
    print("\nRES: WFV COMPLETE. Saved to wf_results.json")
    for r in results:
        print(f"Window {r['window']}: OOS Profit ${r['oos_metrics'].get('net_profit',0):.2f}")

if __name__ == "__main__":
    main()
