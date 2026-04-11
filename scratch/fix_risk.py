path = 'core/risk/risk_guardian.py'
with open(path, 'r') as f:
    content = f.read()

# Fix the 'Riverside' hallucination if it exists
content = content.replace(' Riverside', '')

# Replace old de-scaling logic with tighter A+ Vault logic
old_code = """        if self.max_equity > balance:
            drawdown = ((self.max_equity - balance) / self.max_equity) * 100
            if drawdown > 4.0:
                # Penalty slope: reduce risk by 20% for every 1% additional DD above 4%
                penalty = max(0, 1.0 - (drawdown - 4.0) * 0.20)
                risk_pct *= penalty
                self.logger.info(f"[RISK] DD Scaling: {drawdown:.2f}% DD detected. Risk throttled to {risk_pct:.3f}%")"""

new_code = """        if self.max_equity > balance:
            drawdown = ((self.max_equity - balance) / self.max_equity) * 100
            if drawdown > 4.0:
                # Penalty slope: reduce risk toward 0 at the limit (e.g. 8%)
                limit = float(self.config.get("risk_governance", {}).get("max_drawdown_halt_pct", 8.0))
                room = limit - 4.0
                penalty = max(0, 1.0 - (drawdown - 4.0) / room) if room > 0 else 0
                risk_pct = risk_pct * penalty
                self.logger.info(f"[RISK] A+ Vault Scaling: {drawdown:.2f}% DD. Risk throttled: {risk_pct:.3f}% (Penalty: {penalty:.2f})")"""

# Note: Using a slightly more flexible matching if whitespace differs
import re
# Escape special chars and replace whitespace with \s+
pattern = re.escape(old_code).replace(r'\ ', r'\s+').replace(r'\n', r'\s+')
if re.search(pattern, content):
    content = re.sub(pattern, new_code, content)
    with open(path, 'w') as f:
        f.write(content)
    print("Risk Guardian hardened with A+ Vault de-scaling.")
else:
    # Try literal match as fallback
    if old_code in content:
        content = content.replace(old_code, new_code)
        with open(path, 'w') as f:
            f.write(content)
        print("Risk Guardian hardened (Literal Match).")
    else:
        print("Could not find the target code block in risk_guardian.py")
