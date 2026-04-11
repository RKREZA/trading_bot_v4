import os

path = 'backtesting/backtester.py'
with open(path, 'r') as f:
    text = f.read()

# We will re-write the reset() logically to be bulletproof.
# It MUST synchronize the RiskGuardian's max_equity to the SUM of allocated balances.
# This ensures that Drawdown % is calculated relative to the capital actually being risked.

new_reset_logic = """    def reset(self, active_strategies: list):
        \"\"\"Full reset of the simulation state with capital allocation (Step 9).\"\"\"
        self.current_index = 0
        self.history = []
        self.open_trades = {}
        
        # Institutional Allocation: Use PortfolioManager to split total pool based on config
        # WFO/Backtest uses initial_partition_balance (e.g. 5,000 or 10,000)
        total_pool = len(active_strategies) * self.initial_partition_balance
        
        self.balances = {}
        self.equities = {}
        self.peak_equity = {}
        self.max_drawdowns = {}
        allocated_sum = 0.0
        
        for strat in active_strategies:
            sid = strat.strategy_id
            # Get the actual $ amount this strategy starts with
            bal = self.portfolio_manager.get_strategy_balance(total_pool, sid)
            self.balances[sid] = bal
            self.equities[sid] = bal
            self.peak_equity[sid] = bal
            self.max_drawdowns[sid] = 0.0
            allocated_sum += bal
            
        # Institutional Integrity: HARD SYNC RiskGuardian
        # Its 'Drawdown' Must be based on the sum of what was allocated, not the global config.
        # This prevents 'Ghost Drawdown' during single-strategy or filtered runs.
        self.risk_guardian.initial_balance = allocated_sum
        self.risk_guardian.max_equity = allocated_sum
        self.risk_guardian.kill_switch_active = False
        
        risk_cfg = self.config.get("risk_governance", {})
        self.risk_guardian.max_drawdown_halt_pct = float(risk_cfg.get("max_drawdown_halt_pct", 8.0))
        self.risk_guardian.max_daily_loss_pct = float(risk_cfg.get("max_daily_loss_pct", 5.0))
            
        self.equity_history = []
        self.checkpoint_manager.clear_checkpoint()"""

# Find the reset method and replace it
import re
pattern = r"def reset\(self, active_strategies: list\):.*?self\.checkpoint_manager\.clear_checkpoint\(\)"
if re.search(pattern, text, re.DOTALL):
    text = re.sub(pattern, new_reset_logic, text, flags=re.DOTALL)
    with open(path, 'w') as f:
        f.write(text)
    print("Backtester reset() method rebuilt and synchronized.")
else:
    print("Could not find reset() method in backtester.py")
