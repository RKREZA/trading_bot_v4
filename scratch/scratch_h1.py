import json
from datetime import datetime
from comprehensive_backtest import ComprehensiveBacktestSuite
from core.data.manager import DataManager

suite = ComprehensiveBacktestSuite()
config = suite.config_loader.get_symbol_config("XAUUSDm")
dm = DataManager(suite.config_loader.config)

dt_from = datetime(2025, 1, 1)

try:
    h1 = dm.prepare_data("XAUUSDm", "H1", dt_from)
    print(f"H1 total length: {len(h1.time)}")
    print(f"First H1 time: {datetime.fromtimestamp(h1.time[0])}")
    print(f"H1 time at index 240: {datetime.fromtimestamp(h1.time[min(240, len(h1)-1)])}")
except Exception as e:
    print(f"Error: {e}")
