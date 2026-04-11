import os

path = 'backtesting/backtester.py'
with open(path, 'r') as f:
    lines = f.readlines()

# Line 53 is index 52 in 0-indexed list
# We need to ensure lines 53 through 67 have exactly 8 spaces of indentation.

corrected_lines = [
    '        self.balances = {}\\n',
    '        allocated_sum = 0.0\\n',
    '        for strat in active_strategies:\\n',
    '            sid = strat.strategy_id\\n',
    '            bal = self.portfolio_manager.get_strategy_balance(total_pool, sid)\\n',
    '            self.balances[sid] = bal\\n',
    '            self.equities[sid] = bal\\n',
    '            self.peak_equity[sid] = bal\\n',
    '            self.max_drawdowns[sid] = 0.0\\n',
    '            allocated_sum += bal\\n',
    '            \\n',
    '        # Institutional Integrity: Sync RiskGuardian to the actual REALIZED allocated capital\\n',
    '        self.risk_guardian.initial_balance = allocated_sum\\n',
    '        self.risk_guardian.max_equity = allocated_sum\\n',
    '        self.risk_guardian.kill_switch_active = False\\n'
]

# Verify the starting line is what we expect
if 'self.balances = {}' in lines[52]:
    lines[52:67] = [c.replace('\\n', '\n') for c in corrected_lines]
    with open(path, 'w') as f:
        f.writelines(lines)
    print("Backtester reset() indentation fixed.")
else:
    print(f"Error: Expected 'self.balances = {{}}' at index 52, but found '{lines[52].strip()}'")
