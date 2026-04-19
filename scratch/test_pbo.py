import sys
sys.path.append('.')
from comprehensive_backtest import ComprehensiveBacktestSuite

suite = ComprehensiveBacktestSuite()
result = suite.run_single_strategy_backtest('PureBreakoutOneMinute', 'XAUUSDm', 'BULLISH', 3.0)
print(result)
