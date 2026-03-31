"""
Quick script to list available symbols in MT5
"""

import MetaTrader5 as mt5

# Initialize
if not mt5.initialize(login=413559204, password="Insia@483311", server="Exness-MT5Trial6"):
    print("Failed to connect")
    exit()

print("\n" + "=" * 50)
print("AVAILABLE SYMBOLS (containing XAU, GBP, BTC, USD)")
print("=" * 50)

symbols = mt5.symbols_get()
found = []

for s in symbols:
    name = s.name
    if any(x in name for x in ['XAU', 'GBP', 'BTC', 'USD']) and 'USD' in name:
        if not name.startswith('#') and not name.startswith('.'):
            found.append({
                'name': name,
                'description': s.description,
                'point': s.point,
                'digits': s.digits,
                'contract_size': s.trade_contract_size,
                'spread': s.spread
            })

# Sort and print
found.sort(key=lambda x: x['name'])

print(f"\nFound {len(found)} symbols:\n")
for s in found[:30]:  # Show first 30
    print(f"{s['name']:15} | {s['description']:30} | Point: {s['point']} | Spread: {s['spread']}")

if len(found) > 30:
    print(f"\n... and {len(found) - 30} more")

print("\n" + "=" * 50)
print("RECOMMENDED SYMBOLS FOR BACKTEST")
print("=" * 50)
print("\nUse these exact names:")
for s in found:
    if 'm' in s['name'] or any(x == s['name'] for x in ['XAUUSDm', 'BTCUSDm']):
        print(f"  - {s['name']}")

mt5.shutdown()
