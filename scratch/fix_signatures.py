path = 'comprehensive_backtest.py'
with open(path, 'r') as f:
    content = f.read()

# Fix the method signatures that are causing NameError
content = content.replace('symbol=symbol, market_condition', 'symbol="XAUUSDm", market_condition')

with open(path, 'w') as f:
    f.write(content)

print("Backtest suite signatures corrected.")
