import logging
from typing import Dict, Any, Optional, Tuple
from datetime import datetime

from core.strategy.engine import TradeSignal
from core.time.time_service import time_service

logger = logging.getLogger("trading_bot.risk_engine")

class RiskEngine:
    """
    Validates trade signals against global and per-symbol risk parameters.
    Implements the Kill Switch logic.
    """
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.max_drawdown = config.get("max_drawdown", 0.10) # 10%
        self.daily_loss_limit = config.get("daily_loss_limit", 0.03) # 3%
        self.max_concurrent_trades = config.get("max_concurrent_trades", 3)
        self.risk_per_trade = config.get("risk_per_trade", 0.01) # 1%
        
        self.is_killed = False
        self.kill_reason = ""
        
        # Runtime state
        self.current_drawdown = 0.0
        self.today_loss = 0.0
        self.active_trades_count = 0

    def validate_signal(self, signal: TradeSignal, account_info: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Validates if a signal can be executed.
        Returns (is_valid, reason)
        """
        if self.is_killed:
            return False, f"Kill switch active: {self.kill_reason}"

        # 1. Check Drawdown
        if self.current_drawdown >= self.max_drawdown:
            self.trigger_kill_switch("Max drawdown breach")
            return False, "Max drawdown breach"

        # 2. Check Daily Loss
        if self.today_loss >= self.daily_loss_limit:
            return False, "Daily loss limit reached"

        # 3. Check Concurrent Trades
        if self.active_trades_count >= self.max_concurrent_trades:
            return False, "Max concurrent trades reached"

        # 4. Check Signal Distance (SL/TP)
        # Avoid signals with too small SL (spread risk)
        point_diff = abs(signal.entry - signal.stop_loss)
        # Assuming we have access to symbol info for min points, but for now just a threshold
        if point_diff <= 0:
            return False, "Invalid SL/TP configuration"

        return True, "Success"

    def trigger_kill_switch(self, reason: str):
        self.is_killed = True
        self.kill_reason = reason
        logger.critical(f"KILL SWITCH TRIGGERED: {reason}")
        # In a real scenario, this would notify the ExecutionEngine to close all positions.

    def update_state(self, metrics: Dict[str, Any]):
        """Updates internal risk metrics from the current portfolio state."""
        self.current_drawdown = metrics.get("drawdown", 0.0)
        self.today_loss = metrics.get("today_loss", 0.0)
        self.active_trades_count = metrics.get("active_trades", 0)
        
        # Auto-trigger kill switch if drawdown is too high
        if self.current_drawdown >= self.max_drawdown:
            self.trigger_kill_switch(f"Max drawdown reached: {self.current_drawdown:.2%}")
