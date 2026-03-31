import logging
from typing import Dict, Optional

logger = logging.getLogger("trading_bot.risk")

class RiskManager:
    def __init__(self, config: dict):
        self.full_config = config
        self.risk_config = config.get("risk", {})
        self.base_risk_pct = self.risk_config.get("risk_per_trade", 1.0)
        self.max_drawdown_limit = self.risk_config.get("max_drawdown_halt_pct", 10.0)
        self.drawdown_scaling_enabled = self.risk_config.get("drawdown_scaling", True)
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
