import sys
import os

def main():
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except:
        pass
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", type=str)
    args = parser.parse_args()
    print(f"SYMBOL FROM ARGS: {args.symbol}")

    project_root = os.path.dirname(os.path.abspath(__file__))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    print("STARTING MT5...")
    from core.connection import MT5Connection
    conn = MT5Connection()
    if not conn.connect():
        print("MT5 FAILED")
        return
    print("MT5 CONNECTED")

    print("STARTING DATA FETCH...")
    from core.data_fetcher import DataFetcher
    fetcher = DataFetcher()
    h4 = fetcher.fetch_candles(args.symbol, "H4", 100)
    print(f"FETCHED {len(h4) if h4 else 0} CANDLES")
    conn.disconnect()

    print("STARTING VALIDATION...")
    from core.walk_forward import WalkForwardValidation
    print("SUCCESS: Module imported.")

if __name__ == "__main__":
    main()
