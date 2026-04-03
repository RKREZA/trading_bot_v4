from datetime import datetime, timezone
import logging
from typing import Dict, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from .broker_clock import BrokerClock

logger = logging.getLogger("trading_bot.risk")

class CircuitBreaker:
    """
    Hard-stop safety mechanisms that go beyond Kelly/drawdown scaling.
    Designed to prevent 'revenge trading' and account blowouts during extreme volatility.
    """
    def __init__(self, config: dict, broker_clock: 'BrokerClock' = None):
        """
        Initializes the CircuitBreaker with risk thresholds.
        
        Args:
            config (dict): Global configuration dictionary.
            broker_clock: BrokerClock instance for authoritative time.
        """
        self.risk_cfg = config.get("risk", {})
        self.max_consecutive_losses = self.risk_cfg.get("max_consecutive_losses", 4)
        self.max_hourly_trades = self.risk_cfg.get("max_hourly_trades", 3)
        self.hourly_trades: Dict[int, int] = {}  # hour -> count
        self._broker_clock = broker_clock
    
    def _get_hour(self) -> int:
        """Get current hour from broker clock or fallback to UTC."""
        if self._broker_clock:
            return self._broker_clock.hour()
        return datetime.now(timezone.utc).hour

    def record_trade(self):
        """Increments the trade counter for the current broker hour."""
        hour = self._get_hour()
        self.hourly_trades[hour] = self.hourly_trades.get(hour, 0) + 1
    
    def reset(self):
        """Clears all trade counters for a fresh day/session."""
        self.hourly_trades = {}

    def check_all(self, context: dict) -> Tuple[bool, str]:
        """
        Evaluates all safety checks.
        
        Args:
            context (dict): Current trading context (losses, margin, etc.).
            
        Returns:
            Tuple[bool, str]: (Allowed, Reason).
        """
        # 1. Consecutive losses
        if context.get("consecutive_losses", 0) >= self.max_consecutive_losses:
            return False, "CIRCUIT_BREAKER: Max consecutive losses"
        
        # 2. Hourly rate limit
        hour = self._get_hour()
        if self.hourly_trades.get(hour, 0) >= self.max_hourly_trades:
            return False, "CIRCUIT_BREAKER: Hourly trade limit reached"
        
        # 3. Margin check
        margin_level = context.get("margin_level", 9999)
        if margin_level < 200:  # Below 200% margin level
            return False, "CIRCUIT_BREAKER: Low margin"
        
        return True, "OK"


class RiskManager:
    """
    Orchestrates position sizing and drawdown protection.
    Features:
    - Dynamic Kelly Criterion scaling.
    - Asymmetric Risk (reducing risk when equity is below SMA).
    - Linear Drawdown Scaling.
    - Session-specific risk multipliers.
    """
    def __init__(self, config: dict, broker_clock: 'BrokerClock' = None):
        """
        Initializes the RiskManager.
        
        Args:
            config (dict): Global configuration dictionary.
            broker_clock: BrokerClock instance for authoritative time.
        """
        self.full_config = config
        self.risk_config = config.get("risk", {})
        self.base_risk_pct = self.risk_config.get("risk_per_trade_pct", 1.0)
        self.max_drawdown_limit = self.risk_config.get("max_drawdown_halt_pct", 10.0)
        self.drawdown_scaling_enabled = self.risk_config.get("drawdown_scaling", True)
        self.circuit_breaker = CircuitBreaker(config, broker_clock=broker_clock)
        self.silent = False
        
        self.max_daily_loss_limit = self.risk_config.get("max_daily_loss_pct", 2.0)
        self.equity_sma_period = 20
        self.daily_equity_history = []  # List of daily closing equities
        
        self.initial_balance = None
        self.day_start_balance = None
        self.current_max_balance = 0.0
        self.session_cfg = config.get("session_config", {})
        self.trade_history = []

    def reset_daily_stats(self, balance: float):
        """Reset daily tracking for circuit breakers."""
        self.day_start_balance = balance
        self.current_max_balance = balance # Reset drawdown peak daily for BT consistency
        self.circuit_breaker.reset()
        if not self.silent: logger.info(f"Daily Risk Stats Reset. Day Start Balance: ${balance:.2f}")

    def check_circuit_breakers(self, current_balance: float, current_equity: float, daily_trades: int, daily_losses: int, consecutive_losses: int) -> Tuple[bool, str]:
        """
        Hard Circuit Breakers based on daily and account-level stats.
        
        Args:
            current_balance (float): Current account balance.
            current_equity (float): Current account equity.
            daily_trades (int): Total trades today.
            daily_losses (int): Total losses today.
            consecutive_losses (int): Current losing streak.
            
        Returns:
            Tuple[bool, str]: (Allowed, Reason).
        """
        # Ensure we have a day start balance
        if self.day_start_balance is None:
            self.day_start_balance = current_balance

        # 1. Max Daily Trades
        if daily_trades >= self.risk_config.get("max_daily_trades", 3):
            return False, "CIRCUIT_BREAKER: Max daily trades reached"

        # 2. Max Consecutive Losses
        # [MOD] Changed default from 3 to 4. 
        # Statistical basis: At a 40% win rate, 3 consecutive losses happen in ~21% of sessions (normal variance).
        # 4 consecutive losses happen in ~13%, making it a more reliable signal of a "bad day" or regime shift.
        if consecutive_losses >= self.risk_config.get("max_consecutive_losses", 4):
            return False, "CIRCUIT_BREAKER: Max consecutive losses reached"

        # 3. Daily Loss Percent (Equity vs Day Start)
        daily_loss_pct = (self.day_start_balance - current_equity) / self.day_start_balance * 100
        if daily_loss_pct >= self.max_daily_loss_limit:
            return False, f"CIRCUIT_BREAKER: Daily loss limit hit ({daily_loss_pct:.2f}%)"

        # 4. Max Drawdown Halt (Equity vs Current Max Balance)
        if self.current_max_balance == 0: self.current_max_balance = current_balance
        max_drawdown_pct = (self.current_max_balance - current_equity) / self.current_max_balance * 100
        if max_drawdown_pct >= self.risk_config.get("max_drawdown_halt_pct", 10.0):
            return False, f"CIRCUIT_BREAKER: Max drawdown halt ({max_drawdown_pct:.2f}%)"

        return True, "OK"

    def _calculate_kelly_fraction(self) -> float:
        """
        Calculates the Kelly Criterion fraction to determine optimal risk.
        Kelly % = (win_rate * odds - (1 - win_rate)) / odds
        Where odds = average_win / average_loss.
        
        Returns:
            float: Recommended risk fraction (un-clamped).
        """
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

    def calculate_scaled_risk(self, current_balance: float, current_equity: float = None, 
                              symbol: Optional[str] = None, session: Optional[str] = None) -> float:
        """
        Main entry point for risk calculation.
        Applies a cascading reduction series:
        Kelly -> Asymmetric (SMA) -> Session -> Drawdown.
        
        Args:
            current_balance (float): Current account balance.
            current_equity (Optional[float]): Current equity.
            session (Optional[str]): Current session.
            
        Returns:
            float: Scaled risk percentage (e.g., 1.5 for 1.5%).
        """
        if self.initial_balance is None:
            self.initial_balance = current_balance
            self.day_start_balance = current_balance
            self.current_max_balance = current_balance
            self.daily_equity_history = [current_balance]
        
        equity = current_equity if current_equity is not None else current_balance

        # 1. Dynamic Kelly Risk
        kelly = self._calculate_kelly_fraction()
        risk_pct = (kelly * 0.25) * 100.0 # Quarter-Kelly
        
        # [FIX] Removed floor at base_risk_pct to allow Kelly to scale down properly
        # if risk_pct < self.base_risk_pct:
        #     risk_pct = self.base_risk_pct
            
        # Clamp based on user's sanity limits if any (defaulting to a wider range than before)
        risk_pct = max(0.1, min(risk_pct, 10.0)) 
        
        # 2. Asymmetric Risk Scaling (Equity < 20-day SMA)
        scaling_enabled = self.risk_config.get("asymmetric_risk_scaling", True)
        if scaling_enabled and len(self.daily_equity_history) > 1:
            # Use a dynamic SMA that grows until it reaches the target period
            window = min(len(self.daily_equity_history), self.equity_sma_period)
            equity_sma = sum(self.daily_equity_history[-window:]) / window
            if equity < equity_sma:
                old_risk = risk_pct
                risk_pct *= 0.5
                if not self.silent:
                    logger.info(f"Asymmetric Risk Scaling: {old_risk:.2f}% -> {risk_pct:.2f}% (Equity ${equity:.2f} < SMA(w={window}) ${equity_sma:.2f})")

        # 3. Apply Session Multiplier
        if session:
            multiplier = 1.0
            # A. Check symbol-specific override
            if symbol:
                sym_sessions = self.full_config.get("symbols_config", {}).get(symbol, {}).get("sessions", {})
                if session in sym_sessions:
                    multiplier = sym_sessions[session].get("risk_multiplier", 1.0)
                else:
                    # B. Fallback to global session config
                    multiplier = self.session_cfg.get(session, {}).get("risk_multiplier", 1.0)
            else:
                # B. Map directly to global session config
                multiplier = self.session_cfg.get(session, {}).get("risk_multiplier", 1.0)
            
            risk_pct *= multiplier
            
        # 4. Handle Drawdown Tracking
        if current_balance > self.current_max_balance:
            self.current_max_balance = current_balance
            return risk_pct

        drawdown_pct = (self.current_max_balance - current_balance) / self.current_max_balance * 100
        
        if drawdown_pct >= self.max_drawdown_limit:
            if not self.silent: logger.critical(f"Max Drawdown Limit Reached: {drawdown_pct:.2f}%")
            return 0.0

        if not self.drawdown_scaling_enabled:
            return risk_pct

        # 5. Drawdown Scaling (Reduce risk linearly after 2% drawdown)
        if drawdown_pct > 2.0:
            scale = max(0.2, 1.0 - (drawdown_pct - 2.0) / (self.max_drawdown_limit - 2.0))
            scaled_risk = risk_pct * scale
            if not self.silent: logger.info(f"Risk Scaled: {risk_pct:.2f}% -> {scaled_risk:.2f}% (DD: {drawdown_pct:.2f}%)")
            return scaled_risk

        return risk_pct

    def record_daily_close(self, closing_equity: float):
        """Record the daily closing equity for SMA calculation."""
        self.daily_equity_history.append(closing_equity)
        if len(self.daily_equity_history) > 100:
            self.daily_equity_history = self.daily_equity_history[-100:]

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
        """Check if daily loss limit (e.g., -2%) is hit."""
        if daily_pnl_pct <= -self.max_daily_loss_limit:
            if not self.silent: logger.warning(f"Daily Loss Limit Hit: {daily_pnl_pct:.2f}%")
            return True
        return False
