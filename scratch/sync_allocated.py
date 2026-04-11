import os
import re

path = 'backtesting/backtester.py'
with open(path, 'r') as f:
    content = f.read()

sync_logic = """        self.balances = {}
        allocated_sum = 0.0
        for strat in active_strategies:
            sid = strat.strategy_id
            bal = self.portfolio_manager.get_strategy_balance(total_pool, sid)
            self.balances[sid] = bal
            self.equities[sid] = bal
            self.peak_equity[sid] = bal
            self.max_drawdowns[sid] = 0.0
            allocated_sum += bal
            
        # Institutional Integrity: Sync RiskGuardian to the actual REALIZED allocated capital
        self.risk_guardian.initial_balance = allocated_sum
        self.risk_guardian.max_equity = allocated_sum
        self.risk_guardian.kill_switch_active = False"""

pattern = r"self\.balances\s+=\s+\{\}.*?self\.risk_guardian\.kill_switch_active\s+=\s+False"
if re.search(pattern, content, re.DOTALL):
    content = re.sub(pattern, sync_logic, content, flags=re.DOTALL)
    with open(path, 'w') as f:
        f.write(content)
    print("Backtester reset() synchronized with REALIZED allocated capital.")
else:
    print("Could not find sync loop in backtester.py")
