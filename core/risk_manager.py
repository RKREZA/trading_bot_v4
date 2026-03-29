import logging
from typing import Dict, Optional

logger = logging.getLogger("trading_bot.risk")

class RiskManager:
    def __init__(self, config: dict):
        self.full_config = config
        self.risk_config = config.get("risk", {})
        self.base_risk_pct = self.risk_config.get("risk_per_trade_pct", 1.0)
        self.max_drawdown_limit = self.risk_config.get("max_drawdown_halt_pct", 10.0)
        self.drawdown_scaling_enabled = self.risk_config.get("drawdown_scaling", True)
        self.silent = False
        
        self.initial_balance = None
        self.current_max_balance = 0.0
        self.session_cfg = config.get("session_config", {})

    def calculate_scaled_risk(self, current_balance: float, session: Optional[str] = None) -> float:
        """
        Scale risk percentage based on current drawdown and trading session.
        Example: If drawdown is 5%, reduce risk. If session is LONDON, apply multiplier.
        """
        if self.initial_balance is None:
            self.initial_balance = current_balance
            self.current_max_balance = current_balance
        
        # Base Scaling
        risk_pct = self.base_risk_pct
        
        # Apply Session Multiplier from session_config
        if session:
            session_data = self.session_cfg.get(session, {})
            # Look for risk_multiplier in session data, fallback to 1.0
            multiplier = session_data.get("risk_multiplier", 1.0)
            risk_pct *= multiplier

        if current_balance > self.current_max_balance:
            self.current_max_balance = current_balance
            return risk_pct

        drawdown_pct = (self.current_max_balance - current_balance) / self.current_max_balance * 100
        
        if drawdown_pct >= self.max_drawdown_limit:
            if not self.silent: logger.critical(f"Max Drawdown Limit Reached: {drawdown_pct:.2f}%")
            return 0.0

        if not self.drawdown_scaling_enabled:
            return risk_pct

        # Scaling logic: Reduce risk linearly after 2% drawdown
        if drawdown_pct > 2.0:
            # Scale factor: 1.0 at 2% DD, 0.2 at max_drawdown_limit
            scale = max(0.2, 1.0 - (drawdown_pct - 2.0) / (self.max_drawdown_limit - 2.0))
            scaled_risk = risk_pct * scale
            if not self.silent: logger.info(f"Risk Scaled: {risk_pct:.2f}% -> {scaled_risk:.2f}% (DD: {drawdown_pct:.2f}%)")
            return scaled_risk

        return risk_pct

    def check_daily_stop(self, daily_pnl_pct: float) -> bool:
        """Check if daily loss limit (e.g., -3%) is hit."""
        limit = self.risk_config.get("daily_loss_limit_pct", 3.0)
        if daily_pnl_pct <= -limit:
            if not self.silent: logger.warning(f"Daily Loss Limit Hit: {daily_pnl_pct:.2f}%")
            return True
        return False
