import sys
import os
sys.path.append(os.getcwd())
from comprehensive_backtest import ComprehensiveBacktestSuite
suite = ComprehensiveBacktestSuite()
res = suite.run_single_strategy_backtest('TrendFollowing', 'XAUUSDm', 'BULLISH', 3.0)
print(res)

import json; config = json.load(open('config/config.json')); print('Config symbols_config:', config.get('symbols_config', {}).get('XAUUSDm', 'MISSING_XAUUSDM'))

from core.config.loader import ConfigLoader
c = ConfigLoader().get_symbol_config('XAUUSDm')
print('Loader output:', c.get('symbols_config', {}).get('XAUUSDm'))

from core.performance_tracker import PerformanceTracker
import os, datetime
timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
out_dir = f'backtest_results/session_final_{timestamp}'
PerformanceTracker.save_audit_pack(suite.last_history, {'portfolio': res, 'strategies': {'TrendFollowing': res}}, out_dir)
print(f'Saved to {out_dir}')
