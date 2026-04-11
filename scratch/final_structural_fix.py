import os

path = 'backtesting/backtester.py'
with open(path, 'r') as f:
    lines = f.readlines()

# We need to find the __init__ method's internal state section
# and replace it and the reset method with correct indentation.

start_idx = -1
for i, line in enumerate(lines):
    if 'self.open_trades = {}' in line:
        start_idx = i + 1
        break

if start_idx != -1:
    end_idx = -1
    for i in range(start_idx, len(lines)):
        if 'def get_state' in lines[i]:
            end_idx = i
            break
            
    if end_idx != -1:
        corrected_block = [
            '        self.balances = {}\\n',
            '        self.equities = {}\\n',
            '        self.peak_equity = {}\\n',
            '        self.max_drawdowns = {}\\n',
            '        self.volatility_history = []\\n',
            '        self.equity_history = []\\n',
            '\\n',
            '    def reset(self, active_strategies: list):\\n',
            '        """Full reset of the simulation state with capital allocation (Step 9)."""\\n',
            '        self.current_index = 0\\n',
            '        self.history = []\\n',
            '        self.open_trades = {}\\n',
            '        \\n',
            '        # Institutional Allocation: Use PortfolioManager to split total pool based on config\\n',
            '        total_pool = len(active_strategies) * self.initial_partition_balance\\n',
            '        \\n',
            '        self.balances = {}\\n',
            '        self.equities = {}\\n',
            '        self.peak_equity = {}\\n',
            '        self.max_drawdowns = {}\\n',
            '        allocated_sum = 0.0\\n',
            '        \\n',
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
            '        # Also sync governance thresholds from the current config to ensure no drift.\\n',
            '        risk_cfg = self.config.get("risk_governance", {})\n',
            '        self.risk_guardian.max_drawdown_halt_pct = float(risk_cfg.get("max_drawdown_halt_pct", 8.0))\\n',
            '        self.risk_guardian.max_daily_loss_pct = float(risk_cfg.get("max_daily_loss_pct", 5.0))\\n',
            '        self.risk_guardian.initial_balance = allocated_sum\\n',
            '        self.risk_guardian.max_equity = allocated_sum\\n',
            '        self.risk_guardian.kill_switch_active = False\\n',
            '\\n',
            '        self.equity_history = []\\n',
            '        self.checkpoint_manager.clear_checkpoint()\\n',
            '\\n'
        ]
        
        lines[start_idx:end_idx] = [c.replace('\\\\n', '\\n').replace('\\n', '\n') for c in corrected_block]
        with open(path, 'w') as f:
            f.writelines(lines)
        print("Backtester structure fixed (Zero-Ruin Synchronized).")
    else:
        print("Error: Could not find get_state method.")
else:
    print("Error: Could not find open_trades initialization.")
