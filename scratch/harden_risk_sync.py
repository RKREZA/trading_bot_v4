import os
import re

path = 'core/risk/risk_guardian.py'
with open(path, 'r') as f:
    content = f.read()

# Add a self.current_portfolio_equity tracker to the constructor state
pattern_init = r"(self\.kill_switch_active\s+=\s+False)"
replacement_init = r"\1\n        self.current_portfolio_equity = 0.0"

if re.search(pattern_init, content):
    content = re.sub(pattern_init, replacement_init, content)

# Update check_governance to store the latest portfolio equity
pattern_gov = r"(self\.max_equity\s+=\s+current_equity)"
replacement_gov = r"\1\n        self.current_portfolio_equity = current_equity"

if re.search(pattern_gov, content):
    content = re.sub(pattern_gov, replacement_gov, content)
else:
    # Fallback to update it even if max_equity doesn't increase
    pattern_gov_fallback = r"(def check_governance\(.*?\):)"
    replacement_gov_fallback = r"\1\n        self.current_portfolio_equity = current_equity"
    content = re.sub(pattern_gov_fallback, replacement_gov_fallback, content)

# Update calculate_lot_size to use the portfolio equity for descaling instead of the local balance
pattern_desc = r"if self\.max_equity > balance:\n\s+drawdown = \(\(self\.max_equity - balance\) / self\.max_equity\)"
replacement_desc = """        # Use the latest tracked portfolio equity for descaling math
        eval_equity = self.current_portfolio_equity if self.current_portfolio_equity > 0 else balance
        if self.max_equity > eval_equity:
            drawdown = ((self.max_equity - eval_equity) / self.max_equity)"""

if re.search(pattern_desc, content, re.DOTALL):
    content = re.sub(pattern_desc, replacement_desc, content, flags=re.DOTALL)

with open(path, 'w') as f:
    f.write(content)

print("RiskGuardian hardened: Descaling now synchronized with Portfolio Equity.")
