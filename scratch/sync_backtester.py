import os
import re

path = 'backtesting/backtester.py'
with open(path, 'r') as f:
    content = f.read()

# Fix the RiskGuardian initialization in reset() to avoid 'Ghost Drawdown'
# The Guardian must start its max_equity tracking from the actual total_pool size.
sync_logic = """        self.balances = {}
        for strat in active_strategies:
            sid = strat.strategy_id
            bal = self.portfolio_manager.get_strategy_balance(total_pool, sid)
            self.balances[sid] = bal
            self.equities[sid] = bal
            self.peak_equity[sid] = bal
            self.max_drawdowns[sid] = 0.0
            
        # Institutional Integrity: Sync RiskGuardian to the actual allocated pool
        self.risk_guardian.initial_balance = total_pool
        self.risk_guardian.max_equity = total_pool
        self.risk_guardian.kill_switch_active = False"""

pattern = r"self\.balances\s+=\s+\{\}.*?self\.max_drawdowns\[sid\]\s+=\s+0\.0"
if re.search(pattern, content, re.DOTALL):
    content = re.sub(pattern, sync_logic, content, flags=re.DOTALL)
    with open(path, 'w') as f:
        f.write(content)
    print("Backtester reset() synchronized with RiskGuardian (Ghost Drawdown Fixed).")
else:
    print("Could not find reset loop in backtester.py")
