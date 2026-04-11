import os

path = 'comprehensive_backtest.py'
with open(path, 'r') as f:
    text = f.read()

# We need to replace the hardcoded portfolio allocations
# to ensure every strategy gets 20% by default in the suite config.

old_block = '''            "portfolio_allocations": {
                "TrendFollowing": 0.4,
                "LiquiditySweepBreakout": 0.4,
                "SmartMeanReversion": 0.2
            },'''

new_block = '''            "portfolio_allocations": {
                "TrendFollowing": 0.2,
                "LiquiditySweepBreakout": 0.2,
                "SmartMeanReversion": 0.2,
                "RangeBounce": 0.2,
                "LiquiditySession": 0.2
            },'''

if old_block in text:
    text = text.replace(old_block, new_block)
    with open(path, 'w') as f:
        f.write(text)
    print("Harmonized suite allocations.")
else:
    print("Failed to find target block.")
