import os

path = 'backtesting/backtester.py'
with open(path, 'r') as f:
    lines = f.readlines()

# We identify the methods and fix their structure.
# PortfolioBacktester starts at line 21.
# __init__ starts at 33.
# We want to ensure 'reset' is a sibling of '__init__'.

fixed_content = """import logging
import numpy as np
import os
from tqdm import tqdm
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional

from core.base_strategy import MarketData
from core.regime_detector import RegimeDetector
from core.risk.risk_guardian import RiskGuardian
from core.session_detector import SessionDetector
from core.portfolio_manager import PortfolioManager
from core.regime_gater import RegimeGater
from core.recovery.checkpoint_manager import CheckpointManager
from core.execution.order_manager import OrderManager
from core.volatility_detector import VolatilityDetector, VolatilityLevel
from strategies.adaptive_manager import AdaptiveStrategyManager, RegimeAwareStrategy

logger = logging.getLogger(\"trading_bot.backtester\")

class PortfolioBacktester:
    def __init__(self, config: dict):
        self.config = config
        self.regime_detector = RegimeDetector()
        self.volatility_detector = VolatilityDetector(atr_period=14, lookback=100)
        self.risk_guardian = RiskGuardian(config)
        self.order_manager = OrderManager(config)
        self.portfolio_manager = PortfolioManager(config)
        self.checkpoint_manager = CheckpointManager()

        bt_cfg = config.get(\"backtest\", {})
        self.initial_partition_balance = float(bt_cfg.get(\"initial_balance_per_strategy\", 1000.0))
        
        vol_cfg = config.get(\"volatility_adaptive\", {})
        self.volatility_adaptive_enabled = vol_cfg.get(\"enabled\", True)
        self.min_volatility_for_trades = vol_cfg.get(\"min_volatility_for_trades\", \"VERY_LOW\")
        
        self.current_index = 0
        self.history = []
        self.open_trades = {}
        self.balances = {}
        self.equities = {}
        self.peak_equity = {}
        self.max_drawdowns = {}
        self.volatility_history = []
        self.equity_history = []

    def reset(self, active_strategies: list):
        \"\"\"Full reset of the simulation state with capital allocation (Step 9).\"\"\"
        self.current_index = 0
        self.history = []
        self.open_trades = {}
        
        total_pool = len(active_strategies) * self.initial_partition_balance
        
        self.balances = {}
        self.equities = {}
        self.peak_equity = {}
        self.max_drawdowns = {}
        allocated_sum = 0.0
        
        for strat in active_strategies:
            sid = strat.strategy_id
            bal = self.portfolio_manager.get_strategy_balance(total_pool, sid)
            self.balances[sid] = bal
            self.equities[sid] = bal
            self.peak_equity[sid] = bal
            self.max_drawdowns[sid] = 0.0
            allocated_sum += bal
            
        risk_cfg = self.config.get(\"risk_governance\", {})
        self.risk_guardian.max_drawdown_halt_pct = float(risk_cfg.get(\"max_drawdown_halt_pct\", 8.0))
        self.risk_guardian.max_daily_loss_pct = float(risk_cfg.get(\"max_daily_loss_pct\", 5.0))
        self.risk_guardian.initial_balance = allocated_sum
        self.risk_guardian.max_equity = allocated_sum
        self.risk_guardian.kill_switch_active = False
            
        self.equity_history = []
        self.checkpoint_manager.clear_checkpoint()

    def get_state(self) -> Dict[str, Any]:
        return {
            \"current_index\": self.current_index,
            \"balances\": self.balances,
            \"equities\": self.equities,
            \"peak_equity\": self.peak_equity,
            \"max_drawdowns\": self.max_drawdowns,
            \"open_trades\": self.open_trades,
            \"history\": self.history
        }

    def set_state(self, state: Dict[str, Any]):
        self.current_index = state[\"current_index\"]
        self.balances = state[\"balances\"]
        self.equities = state[\"equities\"]
        self.peak_equity = state[\"peak_equity\"]
        self.max_drawdowns = state[\"max_drawdowns\"]
        self.open_trades = state[\"open_trades\"]
        self.history = state[\"history\"]
"""

# We need to preserve the run() method and everything AFTER it.
# run() usually starts after set_state.

start_running_idx = -1
for i, line in enumerate(lines):
    if 'def run(self' in line:
        start_running_idx = i
        break

if start_running_idx != -1:
    with open(path, 'w') as f:
        f.write(fixed_content)
        f.writelines(lines[start_running_idx:])
    print("Backtester class structure restored and synchronized.")
else:
    print("Could not find run method.")
