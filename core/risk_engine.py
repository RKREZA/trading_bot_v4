import logging
import numpy as np
from datetime import datetime, date
from collections import deque

class RiskEngine:
    """
    V4 Institutional Risk Governance Engine.
    Mandatory ATR-based sizing, dynamic scaling, and equity protection.
    """

    def __init__(self, config: dict):
        self.config = config

        backtest_cfg = config.get("backtest", {}) if isinstance(config.get("backtest", {}), dict) else {}
        risk_cfg = config.get("risk_governance", {}) if isinstance(config.get("risk_governance", {}), dict) else {}

        self.initial_balance = float(
            backtest_cfg.get("initial_balance", config.get("initial_balance", 1000.0))
        )
        self.risk_per_trade_pct = float(
            risk_cfg.get("risk_per_trade_pct", config.get("risk_per_trade_pct", 1.0))
        )
        self.max_daily_loss_pct = float(
            risk_cfg.get("max_daily_loss_pct", config.get("max_daily_loss_pct", 5.0))
        )
        self.max_total_drawdown_pct = float(
            risk_cfg.get("max_drawdown_halt_pct", config.get("max_total_drawdown_pct", 20.0))
        )

        # Institutional Cost Filter
        self.max_cost_ratio = float(config.get("max_cost_ratio", 0.75)) # 75% max risk-to-cost ratio
        self.min_account_buffer = float(config.get("min_account_buffer", 0.15)) 
        
        self.daily_loss = 0.0
        self.daily_trades = 0
        self.consecutive_losses = 0
        self.equity_history = deque(maxlen=200)
        self.last_reset_date = date.today()
        self.kill_switch_active = False
        
        # Institutional Minimum Notional (e.g., $1000 standard)
        self.min_notional_value = float(config.get("risk_governance", {}).get("min_notional_value", 1000.0))
        
        self.logger = logging.getLogger("trading_bot.risk_engine")

    def calculate_lot_size(self, balance: float, stop_loss_distance: float, point: float, tick_value: float, symbol: str, spread_points: float = 0.0, commission_per_lot: float = 0.0) -> float:
        if stop_loss_distance <= 0:
            return 0.0

        # 1. Dynamic Risk Scaling (Account Hardening)
        effective_risk_pct = self.risk_per_trade_pct
        
        # Balance-aware scaling
        if balance < 1500:
            effective_risk_pct *= 0.75  # Slightly less aggressive than before to allow $5 risk
            
        # Drawdown scaling
        if self.consecutive_losses >= 3:
            effective_risk_pct *= 0.5
            self.logger.warning(f"Risk reduced to {effective_risk_pct}% due to losing streak.")

        risk_amount = balance * (effective_risk_pct / 100.0)
        
        # 2. Stop Loss Floor (Institutional Quality Filter)
        # We enforce a minimum SL distance of 3.0x current spread to survive noise.
        min_sl_dist = spread_points * point * 3.0
        actual_sl_dist = max(stop_loss_distance, min_sl_dist)
        
        # 3. Fixed Cost Analysis (Institutional Filter)
        # Lookup symbol-specific cost ratio (Don't hardcode - Reality Check)
        symbol_cfg = self.config.get("symbols_config", {}).get(symbol, {})
        allowed_ratio = float(symbol_cfg.get("max_cost_ratio", self.max_cost_ratio))
        
        # Calculate lots such that (SL distance in points * tick_value) matches risk_amount
        points_dist = (actual_sl_dist / point) if point > 0 else actual_sl_dist
        potential_lot = risk_amount / (points_dist * tick_value) if points_dist > 0 else 0.0
        
        # Cost = (Spread in points * tick_value * lot) + (lot * commission)
        fixed_cost = (spread_points * potential_lot * tick_value) + (potential_lot * commission_per_lot)
        
        if risk_amount > 0 and (fixed_cost / risk_amount) > allowed_ratio:
            self.logger.warning(f"Trade rejected on {symbol}: Cost-to-Risk ratio {fixed_cost/risk_amount:.2f} exceeds limit {allowed_ratio:.2f}")
            return 0.0

        return self._apply_broker_constraints(potential_lot, symbol)

    def check_circuit_breakers(self, current_balance: float, current_equity: float) -> tuple[bool, str]:
        if self.kill_switch_active:
            return False, "Kill Switch Active"

        daily_dd = (self.daily_loss / current_balance) * 100 if current_balance > 0 else 0
        if daily_dd >= self.max_daily_loss_pct:
            return False, f"Daily DD Threshold Reached: {daily_dd:.2f}%"

        total_dd = ((self.initial_balance - current_equity) / self.initial_balance) * 100 if self.initial_balance > 0 else 0
        if total_dd >= self.max_total_drawdown_pct:
            self.kill_switch_active = True
            return False, f"Max Total DD Reached: {total_dd:.2f}%"

        if len(self.equity_history) >= 50:
            ma_equity = np.mean(self.equity_history[-50:])
            equity_gap_pct = ((ma_equity - current_equity) / ma_equity) * 100 if ma_equity > 0 else 0
            if current_equity < ma_equity and equity_gap_pct > 3.0:
                return False, f"Equity below 50-period MA (Gap: {equity_gap_pct:.1f}%)"

        return True, "OK"

    def update_history(self, pnl: float, equity: float):
        self.equity_history.append(equity)
        if pnl < 0:
            self.daily_loss += abs(pnl)
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0
            
    def get_magic_number(self, strategy_id: str) -> int:
        """
        Dynamically derives a unique magic number for a strategy instance.
        Base: magic_number from config (default 234000).
        Offset: hash of strategy_id to ensure persistence.
        """
        base_magic = int(self.config.get("magic_number", 234000))
        # Simple deterministic offset based on strategy ID hash
        import hashlib
        sid_hash = int(hashlib.md5(strategy_id.encode()).hexdigest(), 16) % 1000
        return base_magic + sid_hash

    def reset_daily(self, balance: float):
        self.daily_loss = 0.0
        self.daily_trades = 0
        self.last_reset_date = date.today()

    def _apply_broker_constraints(self, lot: float, symbol: str) -> float:
        sym_cfg = self.config.get("symbols_config", {}).get(symbol, {})
        min_lot = float(sym_cfg.get("min_lot", 0.01))
        max_lot = float(sym_cfg.get("max_lot", 100.0))
        step = float(sym_cfg.get("lot_step", 0.01))
        
        # 1. Institutional 'Liquidity Clamp' (Rule 5.1 Hardened)
        # Strictly enforced cap to prevent extreme exposure.
        clamp_cap = float(sym_cfg.get("max_liquidity_lot", 25.0))
        final_lot = min(lot, clamp_cap)
        
        # 2. Minimum Notional Guard
        # Rejects trades that are too small to be institutionally viable or below broker minimum.
        if final_lot < min_lot:
            self.logger.warning(f"Trade REJECTED for {symbol}: Lot {final_lot:.4f} is below broker minimum {min_lot}.")
            return 0.0
            
        # Get actual contract size from config (Audit Fix)
        # Avoids overestimating notional for Gold/Indices
        contract_size = float(sym_cfg.get("contract_size", 100000.0))
        
        # We assume 1.0 lot = contract_size of base currency.
        notional_estimate = final_lot * contract_size
        if notional_estimate < self.min_notional_value:
             self.logger.warning(f"Trade REJECTED for {symbol}: Estimated notional ${notional_estimate:,.2f} is below threshold ${self.min_notional_value:,.2f}.")
             return 0.0
        
        normalized = round(final_lot / step) * step
        return min(max_lot, normalized)
