import os

path = 'backtesting/backtester.py'
with open(path, 'r') as f:
    lines = f.readlines()

# Localizing the reset method in PortfolioBacktester
# Lines 53-66 were corrupted with bad indentation.
# Index 52 corresponds to Line 53.

corrected_lines = [
    '        self.balances = {}\\n',
    '        for strat in active_strategies:\\n',
    '            sid = strat.strategy_id\\n',
    '            bal = self.portfolio_manager.get_strategy_balance(total_pool, sid)\\n',
    '            self.balances[sid] = bal\\n',
    '            self.equities[sid] = bal\\n',
    '            self.peak_equity[sid] = bal\\n',
    '            self.max_drawdowns[sid] = 0.0\\n',
    '            \\n',
    '        # Institutional Integrity: Sync RiskGuardian to the actual allocated pool\\n',
    '        self.risk_guardian.initial_balance = total_pool\\n',
    '        self.risk_guardian.max_equity = total_pool\\n',
    '        self.risk_guardian.kill_switch_active = False\\n'
]

# Replacement
lines[52:66] = [c.replace('\\n', '\n') for c in corrected_lines]

with open(path, 'w') as f:
    f.writelines(lines)

print("Backtester.reset() indentation corrected.")
