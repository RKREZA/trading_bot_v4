import sys
sys.path.append('.')

path = 'strategies/trend_following.py'
with open(path, 'r', encoding='utf-8') as f:
    content = f.read()

old = 'min_rr = self.config.get("risk_governance", {}).get("min_rr", 1.5)'
new = 'min_rr = self.config.get("risk_governance", {}).get("min_rr", 1.1)'
content2 = content.replace(old, new)

if content2 == content:
    print("WARNING: pattern not found, no replacement made")
else:
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content2)
    print("SUCCESS: min_rr default lowered to 1.1")
