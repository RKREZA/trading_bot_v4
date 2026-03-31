from datetime import datetime, timezone
import logging
from typing import Dict, Optional, Tuple

logger = logging.getLogger("trading_bot.risk")

class CircuitBreaker:
    """Hard-stop safety mechanisms that go beyond Kelly/drawdown scaling."""
    def __init__(self, config: dict):
        self.risk_cfg = config.get("risk", {})
        self.max_consecutive_losses = self.risk_cfg.get("max_consecutive_losses", 4)
        self.max_hourly_trades = self.risk_cfg.get("max_hourly_trades", 3)
        self.hourly_trades: Dict[int, int] = {}  # hour -> count
    
    def record_trade(self):
        hour = datetime.now(timezone.utc).hour
        self.hourly_trades[hour] = self.hourly_trades.get(hour, 0) + 1

    def check_all(self, context: dict) -> Tuple[bool, str]:
        """Returns (allowed, reason)."""
        # 1. Consecutive losses
        if context.get("consecutive_losses", 0) >= self.max_consecutive_losses:
            return False, "CIRCUIT_BREAKER: Max consecutive losses"
        
        # 2. Hourly rate limit
        hour = datetime.now(timezone.utc).hour
        if self.hourly_trades.get(hour, 0) >= self.max_hourly_trades:
            return False, "CIRCUIT_BREAKER: Hourly trade limit reached"
        
        # 3. Margin check
        margin_level = context.get("margin_level", 9999)
        if margin_level < 200:  # Below 200% margin level
            return False, "CIRCUIT_BREAKER: Low margin"
        
        return True, "OK"


class RiskManager:
    def __init__(self, config: dict):
        self.full_config = config
        self.risk_config = config.get("risk", {})
        self.base_risk_pct = self.risk_config.get("risk_per_trade", 1.0)
        self.max_drawdown_limit = self.risk_config.get("max_drawdown_halt_pct", 10.0)
        self.drawdown_scaling_enabled = self.risk_config.get("drawdown_scaling", True)
        self.circuit_breaker = CircuitBreaker(config)
        self.silent = False
        
        self.initial_balance = None
        self.current_max_balance = 0.0
        self.session_cfg = config.get("session_config", {})
        self.trade_history = []

    def _calculate_kelly_fraction(self) -> float:
        """Calculate Kelly Fraction based on trade history. Thread-safe."""
        # Fix Memory Leak: only run on recent trades window
        recent_trades = self.trade_history[-100:]
        
        min_trades = self.risk_config.get("kelly_min_trades", 15)
        if len(recent_trades) < min_trades:
            # Not enough data for statistical significance, return base risk
            return self.base_risk_pct / 100.0
            
        wins = [t['pnl'] for t in recent_trades if t['pnl'] > 0]
        losses = [abs(t['pnl']) for t in recent_trades if t['pnl'] <= 0]
        
        if not wins or not losses:
            return self.base_risk_pct / 100.0
            
        win_rate = len(wins) / len(recent_trades)
        avg_win = sum(wins) / len(wins)
        avg_loss = sum(losses) / len(losses)
        
        if avg_win == 0: return 0.0
        
        # Kelly % = (win_rate / avg_loss_ratio) - ((1 - win_rate) / avg_win_ratio)
        # Simplified: (bp - q) / b where b is odds (avg_win/avg_loss)
        odds = avg_win / avg_loss if avg_loss > 0 else 1.0
        kelly = (win_rate * odds - (1 - win_rate)) / odds if odds > 0 else 0.0
        
        return max(0.0, kelly)

    def calculate_scaled_risk(self, current_balance: float, session: Optional[str] = None) -> float:
        """
        Scale risk percentage based on Kelly Fraction, current drawdown and trading session.
        """
        if self.initial_balance is None:
            self.initial_balance = current_balance
            self.current_max_balance = current_balance
        
        # 1. Dynamic Kelly Risk
        kelly = self._calculate_kelly_fraction()
        risk_pct = (kelly * 0.25) * 100.0 # Quarter-Kelly
        
        # Clamp between 0.5% and 2.0%
        risk_pct = max(0.5, min(risk_pct, 2.0))
        
        # 2. Apply Session Multiplier
        if session:
            session_data = self.session_cfg.get(session, {})
            multiplier = session_data.get("risk_multiplier", 1.0)
            risk_pct *= multiplier
            
        # 3. Handle Drawdown Tracking
        if current_balance > self.current_max_balance:
            self.current_max_balance = current_balance
            return risk_pct

        drawdown_pct = (self.current_max_balance - current_balance) / self.current_max_balance * 100
        
        if drawdown_pct >= self.max_drawdown_limit:
            if not self.silent: logger.critical(f"Max Drawdown Limit Reached: {drawdown_pct:.2f}%")
            return 0.0

        if not self.drawdown_scaling_enabled:
            return risk_pct

        # 4. Drawdown Scaling (Reduce risk linearly after 2% drawdown)
        if drawdown_pct > 2.0:
            scale = max(0.2, 1.0 - (drawdown_pct - 2.0) / (self.max_drawdown_limit - 2.0))
            scaled_risk = risk_pct * scale
            if not self.silent: logger.info(f"Risk Scaled: {risk_pct:.2f}% -> {scaled_risk:.2f}% (DD: {drawdown_pct:.2f}%)")
            return scaled_risk

        return risk_pct

    def update_history(self, trade_record: Dict):
        """Update internal trade history for Kelly calculation. Prevents duplicates by ticket."""
        ticket = trade_record.get('ticket')
        if ticket:
            if any(t.get('ticket') == ticket for t in self.trade_history):
                return
        self.trade_history.append(trade_record)
        # Keep window to 200 trades to prevent memory bloat
        if len(self.trade_history) > 200:
            self.trade_history = self.trade_history[-200:]

    def check_daily_stop(self, daily_pnl_pct: float) -> bool:
        """Check if daily loss limit (e.g., -5%) is hit."""
        limit = self.risk_config.get("max_daily_loss_percent", 5.0)
        if daily_pnl_pct <= -limit:
            if not self.silent: logger.warning(f"Daily Loss Limit Hit: {daily_pnl_pct:.2f}%")
            return True
        return False
