import logging
import numpy as np
from collections import deque
from datetime import datetime, date
from typing import Dict, Any, Optional, Tuple

class RiskGuardian:
    """
    Institutional Risk Governance Guardian.
    Enforces strict ATR-based sizing, dynamic scaling, and multi-layer circuit breakers.
    Independently runnable and testable.
    """

    def __init__(self, config: Dict[str, Any], broker_clock=None):
        self.config = config
        self.broker_clock = broker_clock
        
        # Extract configurations with safe defaults
        risk_cfg = config.get("risk_governance", {})
        
        self.initial_balance = float(config.get("backtest", {}).get("initial_balance", 1000.0))
        self.risk_per_trade_pct = float(risk_cfg.get("risk_per_trade_pct", 0.5))
        self.max_daily_loss_pct = float(risk_cfg.get("max_daily_loss_pct", 5.0))
        self.max_drawdown_halt_pct = float(risk_cfg.get("max_drawdown_halt_pct", 20.0))
        
        # State Tracking
        self.daily_loss = 0.0
        self.consecutive_losses = 0
        self.equity_history = deque(maxlen=200)  # Bounded to prevent memory leaks
        self.error_count = 0
        self.max_equity = self.initial_balance
        self.kill_switch_active = False
        self.silent = False
        
        self.logger = logging.getLogger("trading_bot.risk")

    def validate_signal(self, 
                        signal: Any, 
                        balance: float, 
                        market_data: Any,
                        symbol_info: Dict[str, Any]) -> bool:
        """
        Institutional Pre-Auction Signal Validation (Step 12: Step 5).
        Verifies if a signal is tradable based on current risk/governance 
        before it enters the Portfolio Auction.
        """
        if self.kill_switch_active:
            return False
            
        # Determine temporary SL distance for sizing check
        try:
            # We need to simulate the SL calculation to check for 'Death Lot' (too small)
            # This is a DRY violation but necessary for decoupling pre-auction
            # Better: pass the SL distance if already calculated.
            # For now, we assume the strategy logic is accessible or SL is attached.
            sl_price = signal.stop_loss if hasattr(signal, 'stop_loss') and signal.stop_loss != 0 else 0
            if sl_price == 0:
                self.logger.error("[RISK] Signal REJECTED: Missing or zero Stop Loss.")
                return False # STRICT ENFORCEMENT: No SL, No Trade
                
            sl_dist = abs(market_data.current_price - sl_price)
            lot = self.calculate_lot_size(balance, sl_dist, symbol_info, current_price=market_data.current_price)
            
            return lot > 0
        except Exception as e:
            self.logger.error(f"[RISK] Signal validation failed due to error: {e}", exc_info=True)
            return False

    def calculate_lot_size(self, 
                           balance: float, 
                           stop_loss_dist: float, 
                           symbol_info: Dict[str, Any],
                           last_pnl: float = 0.0,
                           current_price: float = 1.0) -> float:
        """
        Institutional Position Sizing: risk_amount / stop_loss_dist
        HARD CONSTRAINTS (Step 13):
        - No Martingale (No size increase after loss)
        - No Doubling after loss
        """
        if stop_loss_dist <= 0 or self.kill_switch_active:
            return 0.0

        # Anti-Martingale / Anti-Doubling (Hard Guard)
        risk_pct = self.risk_per_trade_pct
        if self.consecutive_losses > 0:
            # If we are in a losing streak, we ONLY reduce or keep risk fixed.
            # We strictly FORBID increasing risk_pct above its base.
            risk_pct = min(self.risk_per_trade_pct, risk_pct * 0.5 if self.consecutive_losses >= 3 else risk_pct)
            
            # Additional No-Doubling Guard: If last PnL was negative, current lot MUST be <= last lot
            # (Note: This is handled naturally by the fixed/reduced risk_pct)
        else:
            # Growth Booster (Stable Growth logic)
            if len(self.equity_history) > 20 and self.consecutive_losses == 0:
                risk_pct = min(1.0, risk_pct * 1.1)

        risk_amount = balance * (risk_pct / 100.0)
        
        point = symbol_info.get('point', 0.00001)
        tick_value = symbol_info.get('tick_value', 1.0)
        
        points_dist = stop_loss_dist / point if point > 0 else 0.0
        
        # FIX: Ensure the entire denominator is evaluated against zero before division
        denominator = points_dist * tick_value
        if denominator > 0:
            raw_lot = risk_amount / denominator
        else:
            raw_lot = 0.0
        
        return self._normalize_lots(raw_lot, symbol_info, current_price)

    def check_governance(self, current_balance: float, current_equity: float, slippage: float = 0.0, is_error: bool = False, open_positions: int = 0) -> Tuple[bool, str]:
        """Global safety check (Kill Switch, Equity Protection, and Parallel Thresholds)"""
        # Institutional Parallel Threshold (Step 13)
        # Defaulting to 4 to allow Trend, Breakout, MeanReversion, and Liquidity to trade together.
        max_parallel_positions = self.config.get("risk_governance", {}).get("max_parallel_strategies", 4)
        if open_positions >= max_parallel_positions:
            return False, f"MAX_PARALLEL_STRATEGIES_REACHED ({max_parallel_positions})"

        if self.kill_switch_active:
            return False, "KILL_SWITCH_ACTIVE"

        if is_error:
            self.error_count += 1

        # 1. Kill Switch logic (Step 5.5)
        daily_dd = (self.daily_loss / current_balance) * 100 if current_balance > 0 else 0
        limit = float(self.config.get("risk_governance", {}).get("max_daily_loss_pct", 5.0))
        if daily_dd >= limit or self.error_count > 10 or slippage > 50:
            self.kill_switch_active = True
            return False, "KILL_SWITCH_TRIGGERED"

        # 2. Equity Protection (Step 5.4)
        # Only block if equity is materially below the MA50 (>3% drawdown from MA)
        # This prevents minor dips from permanently blocking trading
        # FIX: self.equity_history.append(current_equity) removed to prevent high-frequency corruption.
        
        if len(self.equity_history) > 50:
            ma_equity = np.mean(list(self.equity_history)[-50:])
            equity_gap_pct = ((ma_equity - current_equity) / ma_equity) * 100 if ma_equity > 0 else 0
            if current_equity < ma_equity and equity_gap_pct > 3.0:
                return False, f"EQUITY_BELOW_MA50_PROTECTION (Gap: {equity_gap_pct:.1f}%)"
        
        # 3. Total Drawdown Check
        if current_equity > self.max_equity: self.max_equity = current_equity
        total_dd = ((self.max_equity - current_equity) / self.max_equity) * 100 if self.max_equity > 0 else 0
        if total_dd >= self.max_drawdown_halt_pct:
            if not self.kill_switch_active:
                self.kill_switch_active = True
                self.logger.critical(f"MAX DRAWDOWN {total_dd:.1f}% REACHED! INITIATING EMERGENCY FLATTEN.")
                return False, "EMERGENCY_FLATTEN_REQUIRED"
            return False, f"MAX_DRAWDOWN_REACHED ({total_dd:.1f}%)"

        return True, "OK"

    def record_trade_result(self, pnl: float, current_equity: float = None):
        """Updates internal risk state after trade closure."""
        # FIX: Append to equity history ONLY on trade result recording to maintain fidelity
        if current_equity is not None:
            self.equity_history.append(current_equity)

        if pnl < 0:
            self.daily_loss += abs(pnl)
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0

    def reset_daily(self, new_balance: float):
        """Synchronizes balance and resets daily risk counters."""
        self.daily_loss = 0.0
        self.initial_balance = new_balance

    def _normalize_lots(self, lot: float, sym: Dict[str, Any], current_price: float = 1.0) -> float:
        min_lot = sym.get('min_lot', 0.01)
        max_lot = sym.get('max_lot', 10.0)
        step = sym.get('lot_step', 0.01)
        
        # 1. Institutional 'Lot Floor'
        if lot < min_lot:
            return 0.0
            
        # 2. Minimum Notional Guard (Audit Fix)
        # Prevents rejected trades based on hardcoded defaults
        contract_size = sym.get('contract_size', 100000.0)
        min_notional = self.config.get("risk_governance", {}).get("min_notional_value", 0.0)
        
        if min_notional > 0:
            # FIX: Include price multiplier to derive true USD-equivalent notional volume
            notional = lot * contract_size * current_price
            if notional < min_notional:
                self.logger.warning(f"Trade REJECTED: Notional {notional:.2f} < Min {min_notional:.2f}")
                return 0.0
        
        normalized = round(lot / step) * step
        return min(max_lot, normalized)

if __name__ == "__main__":
    # Independent Test Logic
    logging.basicConfig(level=logging.INFO)
    test_config = {
        "risk_governance": {"risk_per_trade_pct": 1.0, "max_daily_loss_pct": 2.0},
        "backtest": {"initial_balance": 10000.0}
    }
    guardian = RiskGuardian(test_config)
    
    print("\n--- RiskGuardian Standalone Test ---")
    
    # Test 1: Sizing
    sym = {"point": 0.01, "tick_value": 1.0, "spread_pips": 50, "commission_per_lot": 7, "min_lot": 0.01, "lot_step": 0.01}
    lots = guardian.calculate_lot_size(10000, 2.50, sym) # 250 points SL
    print(f"Calculated Lots (SL=2.50): {lots}")
    
    # Test 2: Circuit Breaker
    guardian.daily_loss = 250 # 2.5% loss
    allowed, reason = guardian.check_governance(10000, 9750)
    print(f"Gov Check (2.5% loss): Allowed={allowed}, Reason={reason}")
    
    # Test 3: Recovery
    guardian.reset_daily(10000)
    allowed, reason = guardian.check_governance(10000, 10000)
    print(f"Gov Check (After Reset): Allowed={allowed}, Reason={reason}")
