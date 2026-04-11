import os

path = 'comprehensive_backtest.py'
with open(path, 'r') as f:
    content = f.read()

# Fix hardcoded symbols and method signatures
content = content.replace(' Riverside', '')
content = content.replace('bt.run("XAUUSDm"', 'bt.run(symbol')
content = content.replace('symbol="XAUUSDm"', 'symbol=symbol')
content = content.replace('def run_walk_forward_optimization(self, strategy_name):', 'def run_walk_forward_optimization(self, strategy_name, symbol="XAUUSDm"):')
content = content.replace('wfo.run_validation("XAUUSDm"', 'wfo.run_validation(symbol')

# Fix WFO loop in run_all_tests
old_wfo_loop = """    for strategy in ["TrendFollowing", "LiquiditySweepBreakout"]:
        wfo_result = suite.run_walk_forward_optimization(strategy)
        results["walk_forward"][strategy] = wfo_result"""

new_wfo_loop = """    for symbol in symbols:
        print(f"\\n  --- WFO Symbol: {symbol} ---")
        for strategy in ["TrendFollowing", "LiquiditySweepBreakout"]:
            wfo_result = suite.run_walk_forward_optimization(strategy, symbol)
            results["walk_forward"][f"{symbol}_{strategy}"] = wfo_result"""

content = content.replace(old_wfo_loop, new_wfo_loop)

with open(path, 'w') as f:
    f.write(content)

print("Backtest suite refactored for Grade A+ (Multi-Symbol WFO).")
