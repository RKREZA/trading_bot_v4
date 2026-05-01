import logging
import json
import os
import time
from collections import deque
from typing import Dict, Any, Optional, Tuple, List
from ..common.types import VolatilityStatus
from ..common.events import event_bus, RiskBreachEvent

class RiskGuardian:
    """
    V6-LOCKED: Institutional Risk Governance Guardian.
    Enforces strict Exposure Netting 2.0, Vol-Adjusted Sizing, and Mismatch Circuit Breakers.
    """
    
    # Rule 1.2: Volatility-Adjusted Multipliers
    VOL_MULTIPLIERS = {
        "MAJOR_FX": 1.0,  # EURUSD, GBPUSD
        "JPY_EXT": 0.8,   # USDJPY, GBPJPY
        "XAUUSDm": 0.5,   # Gold
        "INDICES": 0.5,   # High Beta
    }

    # Default Symbol → Basket Mapping (overridden by config["risk_governance"]["basket_map"])
    DEFAULT_BASKET_MAP = {
        "EURUSD": "MAJOR_FX", "GBPUSD": "MAJOR_FX", "AUDUSD": "MAJOR_FX",
        "USDCHF": "MAJOR_FX", "USDCAD": "MAJOR_FX", "NZDUSD": "MAJOR_FX",
        "USDJPY": "JPY_EXT", "GBPJPY": "JPY_EXT", "EURJPY": "JPY_EXT",
        "AUDJPY": "JPY_EXT", "CADJPY": "JPY_EXT", "NZDJPY": "JPY_EXT",
        "XAUUSDm": "XAUUSDm", "XAUUSD": "XAUUSDm",
        "DE30": "INDICES", "US30": "INDICES", "NAS100": "INDICES",
        "US500": "INDICES", "UK100": "INDICES",
    }

    def __init__(self, config: Dict[str, Any], broker_clock=None):
        self.config = config
        self.broker_clock = broker_clock
        
        # Institutional Integrity: Resilient Config (Audit PASS #5 Fix)
        # We use safe defaults to prevent startup crashes while logging critical warnings.
        logger = logging.getLogger("trading_bot.risk")
        risk_governance = config.get("risk_governance", {})
        if not risk_governance:
             logger.critical("CRITICAL CONFIG WARNING: 'risk_governance' section missing! Using safe defaults.")

        # Institutional Safe Defaults
        self.risk_per_trade_pct = float(risk_governance.get("risk_per_trade_pct", 0.5))
        self.max_daily_loss_pct = float(risk_governance.get("max_daily_loss_pct", 5.0))
        self.max_drawdown_halt_pct = float(risk_governance.get("max_drawdown_halt_pct", 20.0))
        
        # Exposure Limits
        self.max_net_exposure_long = int(risk_governance.get("max_net_exposure_long", 3))
        self.max_net_exposure_short = int(risk_governance.get("max_net_exposure_short", 3))
        
        if "risk_per_trade_pct" not in risk_governance:
            logger.warning(f"Risk parameter 'risk_per_trade_pct' missing. Defaulting to safe {self.risk_per_trade_pct}%.")
        
        # Mode Detection
        bt_cfg = config.get("backtest", {})
        self._mode = "live"
        if bt_cfg.get("enabled") or config.get("backtest_mode"):
            self._mode = "backtest"
        
        # State Tracking
        # Institutional Calibration Priority: Search for initial_balance_per_strategy first, then standard initial_balance
        bt_cfg = config.get("backtest", {})
        self.initial_balance = float(bt_cfg.get("initial_balance_per_strategy", 
                                     bt_cfg.get("initial_balance", 
                                     config.get("initial_balance", 1000.0))))
        self.daily_loss = 0.0
        self.consecutive_losses = 0
        self.equity_history = deque(maxlen=200)
        self.error_count = 0
        self.max_equity = self.initial_balance
        self.kill_switch_active = False
        self.current_portfolio_equity = 0.0
        self.silent = False
        
        # Strategy-Level Circuit Breakers (Step 24)
        self.strategy_performance = {} # format: {sid: deque([ (timestamp, pnl_pct), ... ])}
        self.strategy_status = {} # format: {sid: "OK" | "HALTED"}
        # INSTITUTIONAL DRY: Path driven from config, not hardcoded
        self.health_file = config.get("paths", {}).get("strategy_health_file", "config/strategy_health.json")
        self.logger = logging.getLogger("trading_bot.risk")
        self._load_health_state()

        # Configurable basket map: merge defaults with config overrides
        self.basket_map = dict(self.DEFAULT_BASKET_MAP)
        config_baskets = config.get("risk_governance", {}).get("basket_map", {})
        self.basket_map.update(config_baskets)

        # Auto-cooldown period for circuit breakers (hours, 0 = manual reset only)
        self.circuit_breaker_cooldown_hours = float(
            config.get("risk_governance", {}).get("circuit_breaker_cooldown_hours", 24.0)
        )

        # Initialize Institutional Telemetry (Telegram)
        try:
            from core.notifications.telegram_alerter import TelegramAlerter
            self.alerter = TelegramAlerter()
        except ImportError:
            self.logger.warning("[RISK] TelegramAlerter could not be imported. Alerts acting as no-op.")
            class DummyAlerter:
                def send_emergency_alert(self, *args, **kwargs): pass
                def send_risk_alert(self, *args, **kwargs): pass
            self.alerter = DummyAlerter()

    def validate_signal(self, 
                        signal: Any, 
                        balance: float, 
                        market_data: Any,
                        symbol_info: Dict[str, Any],
                        sl_dist: float = 0.0) -> bool:
        """
        Institutional Pre-Auction Signal Validation (Step 12: Step 5).
        Verifies if a signal is tradable based on current risk/governance 
        before it enters the Portfolio Auction.
        
        DRY REFACTOR: Accepts optional `sl_dist` parameter. If the calling
        site has already computed stop-loss distance, pass it directly to
        avoid redundant SL derivation. Falls back to signal.stop_loss if
        sl_dist is not provided.
        """
        if self.kill_switch_active:
            return False
            
        try:
            # DRY: Use pre-computed sl_dist if provided; otherwise derive from signal
            if sl_dist <= 0:
                sl_price = signal.stop_loss if hasattr(signal, 'stop_loss') and signal.stop_loss != 0 else 0
                if sl_price == 0:
                    self.logger.error("[RISK] Signal REJECTED: Missing or zero Stop Loss.")
                    return False  # STRICT ENFORCEMENT: No SL, No Trade
                sl_dist = abs(market_data.current_price - sl_price)
                
            lot = self.calculate_lot_size(balance, sl_dist, symbol_info, current_price=market_data.current_price)
            return lot > 0
        except Exception as e:
            self.logger.error(f"[RISK] Signal validation failed due to error: {e}", exc_info=True)
            return False

    def get_magic_number(self, strategy_id: str) -> int:
        """
        Dynamically derives a unique magic number for a strategy instance.
        Base: magic_number from config (default 234000).
        Offset: SHA256 hash of strategy_id — low collision probability even for
        large strategy registries (replaces MD5 which had ~1/1000 collision risk).
        """
        base_magic = int(self.config.get("magic_number", 234000))
        import hashlib
        sid_hash = int(hashlib.sha256(strategy_id.encode()).hexdigest(), 16) % 1000
        return base_magic + sid_hash

    def calculate_lot_size(self, 
                           balance: float, 
                           stop_loss_dist: float, 
                           symbol_info: Dict[str, Any],
                           current_price: float = 1.0,
                           volatility_status: Optional[VolatilityStatus] = None) -> float:
        """
        Institutional Position Sizing: risk_amount / stop_loss_dist
        Includes Volatility Scaling and Drawdown-Aware De-scaling.
        """
        # [ Calibration Hardening ]: Force minimum SL distance (10 ticks) to avoid zero-lot math
        point = symbol_info.get('point', 0.01)
        stop_loss_dist = max(stop_loss_dist, point * 10)

        if self.kill_switch_active:
            return 0.0        # 1. Base Risk Calculation
        risk_pct = self.risk_per_trade_pct
        
        # 1a. REGIME VOLATILITY SCALING (Centralized from Gater/Orchestrator)
        if volatility_status == VolatilityStatus.HIGH:
            risk_pct = risk_pct * 0.5
            if not self.silent:
                self.logger.info("[RISK] Volatility Scaling: HIGH volatility detected. Risk halved.")

        # 1b. Equity Drawdown De-scaling (INSTITUTIONAL ENFORCEMENT)
        # Smoothly ramps down position sizing as drawdown increases past 4%,
        # reaching zero risk at max_drawdown_halt_pct.
        eval_equity = self.current_portfolio_equity if self.current_portfolio_equity > 0 else balance
        if self.max_equity > eval_equity:
            drawdown = ((self.max_equity - eval_equity) / self.max_equity) * 100
            if drawdown > 4.0:
                limit = float(self.config.get("risk_governance", {}).get("max_drawdown_halt_pct", 8.0))
                room = limit - 4.0
                penalty = max(0, 1.0 - (drawdown - 4.0) / room) if room > 0 else 0
                risk_pct = risk_pct * penalty
                if not self.silent:
                    self.logger.info(f"[RISK] A+ Vault Scaling: {drawdown:.2f}% DD. Risk throttled: {risk_pct:.3f}% (Penalty: {penalty:.2f})")

        # 1c. Anti-Martingale: Progressive scaling — 5% reduction per consecutive loss
        if self.consecutive_losses > 0:
            decay = 0.95 ** self.consecutive_losses
            risk_pct = risk_pct * decay

        risk_amount = balance * (risk_pct / 100.0)
        
        # Institutional Precision: Use symbol's actual point, default to 0.00001 ONLY if missing from sym_info
        point = symbol_info.get('point', 0.00001)
        tick_value = symbol_info.get('tick_value', symbol_info.get('trade_tick_value', 1.0))
        
        points_dist = stop_loss_dist / point if point > 0 else 0.0
        
        denominator = points_dist * tick_value
        raw_lot = (risk_amount / denominator) if denominator > 0 else 0.0

        # Apply Institutional Exposure-Based Ceiling (Phase 2 Refactor)
        # Instead of regime confidence, we use a global safety cap based on current account drawdown.
        # Max cap is the configured _phase1_lot_ceiling, but it degrades as DD increases.
        base_ceiling = float(self.config.get("execution", {}).get("phase1_lot_ceiling", 
                            self.config.get("risk_governance", {}).get("phase1_lot_ceiling", 50.0)))
        eval_equity = self.current_portfolio_equity if self.current_portfolio_equity > 0 else balance
        if self.max_equity > eval_equity:
            drawdown = ((self.max_equity - eval_equity) / self.max_equity) * 100
            # Ceiling starts shrinking at 2% DD
            if drawdown > 2.0:
                limit = self.max_drawdown_halt_pct
                dd_penalty = max(0.1, 1.0 - (drawdown / limit)) 
                base_ceiling *= dd_penalty

        raw_lot = min(raw_lot, base_ceiling)

        return self._normalize_lots(raw_lot, symbol_info, current_price)

    def check_governance(self, 
                         current_balance: float, 
                         current_equity: float, 
                         positions: List[Dict[str, Any]] = None) -> Tuple[bool, str]:
        """
        Global safety check (Exposure Netting 2.0, Kill Switch, Equity Protection).
        Rule 1.1: 2.0 lots per $10k equity (Vol-Adjusted)
        Rule 1.3: Total Gross Exposure <= 8.0 lots per $10k
        """
        if self.kill_switch_active:
            # INSTITUTIONAL ENFORCEMENT: Kill-switch is absolute. No bypass.
            # Any active kill-switch MUST halt all trading operations immediately.
            return False, "KILL_SWITCH_ACTIVE"

        # 1. Total Drawdown Check (A+ Stricter Halts)
        if current_equity > self.max_equity: 
            self.max_equity = current_equity
        self.current_portfolio_equity = current_equity
            
        total_dd = ((self.max_equity - current_equity) / self.max_equity) * 100 if self.max_equity > 0 else 0
        if total_dd >= self.max_drawdown_halt_pct:
            self.kill_switch_active = True
            msg = f"MAX DRAWDOWN {total_dd:.1f}% REACHED! EMERGENCY HALT. MaxEq: {self.max_equity:.2f}, CurrEq: {current_equity:.2f}"
            self.logger.critical(msg)
            if hasattr(self, 'alerter'):
                self.alerter.send_emergency_alert(msg)
            self._emit_risk_breach("MAX_DRAWDOWN", "CRITICAL", msg, total_dd, self.max_drawdown_halt_pct)
            return False, f"MAX_DRAWDOWN_REACHED ({total_dd:.1f}%)"

        # 2. Exposure Netting 2.0 Enforcement
        if positions:
            equity_10k_units = current_equity / 10000.0
            total_gross = 0.0
            ccy_exposure = {} # {basket: total_lots}
            net_exposure = {}  # {basket: net_lots} for hedging detection
            
            for p in positions:
                symbol = p.get('symbol', 'UNKNOWN').upper()
                lots = p.get('volume', 0.0)
                direction = p.get('type', p.get('direction', 'BUY'))
                signed_lots = lots if str(direction).upper() in ('BUY', '0') else -lots
                total_gross += lots
                
                # CONFIG-DRIVEN basket classification (replaces naive string matching)
                basket = self.basket_map.get(symbol, self._infer_basket(symbol))
                
                ccy_exposure[basket] = ccy_exposure.get(basket, 0.0) + lots
                net_exposure[basket] = net_exposure.get(basket, 0.0) + signed_lots
                
                # Limit Check per Basket
                limit = 2.0 * equity_10k_units * self.VOL_MULTIPLIERS.get(basket, 1.0)
                if ccy_exposure[basket] > limit:
                    self.logger.warning(f"[RISK] Basket {basket} breach: {ccy_exposure[basket]:.2f} > {limit:.2f}")
                    self._emit_risk_breach("EXPOSURE_NETTING", "WARNING",
                                           f"Basket {basket}: {ccy_exposure[basket]:.2f} > {limit:.2f}",
                                           ccy_exposure[basket], limit)
                    return False, f"EXPOSURE_NETTING_BREACH ({basket})"

            global_cap = 8.0 * equity_10k_units
            if total_gross > global_cap:
                self.logger.warning(f"[RISK] Global Exposure breach: {total_gross:.2f} > {global_cap:.2f}")
                self._emit_risk_breach("GLOBAL_EXPOSURE", "WARNING",
                                       f"Gross {total_gross:.2f} > cap {global_cap:.2f}",
                                       total_gross, global_cap)
                return False, f"GLOBAL_EXPOSURE_CAP_REACHED"

        return True, "OK"

    def detect_mismatch(self, 
                        internal_positions: List[Dict[str, Any]], 
                        mt5_positions: List[Any]) -> Tuple[bool, str, bool]:
        """
        Rule 2: Notional & Directional Conflict Trigger.
        Returns (mismatch_detected, reason, immediate_flatten_required)
        """
        # Directional Conflict Check (MANDATORY: IMMEDIATE FLATTEN)
        # We check symbol by symbol
        int_map = {p['symbol']: p for p in internal_positions}
        mt5_map = {p.symbol: p for p in mt5_positions}
        
        for symbol in set(list(int_map.keys()) + list(mt5_map.keys())):
            ip = int_map.get(symbol)
            mp = mt5_map.get(symbol)
            
            if ip and mp:
                # Rule 2.2: Directional Conflict
                id_dir = "BUY" if "BUY" in ip.get('type_text', '').upper() else "SELL"
                md_dir = "BUY" if mp.type == 0 else "SELL" # 0=BUY, 1=SELL
                if id_dir != md_dir:
                    return True, f"DIRECTIONAL_CONFLICT on {symbol}", True
                
                # Rule 2.1: Notional Mismatch (> 0.5 lots)
                if abs(ip.get('volume', 0) - mp.volume) > 0.5:
                    return True, f"NOTIONAL_MISMATCH on {symbol} Delta > 0.5", True
            
            # Orphaned Position (External manual interference or desync)
            elif not ip and mp:
                if mp.volume > 0.1: # Tolerance for micro-residual
                    return True, f"GHOST_POSITION on {symbol}", False # Soft sync first
            elif ip and not mp:
                return True, f"MISSING_POSITION on {symbol}", False # Soft sync first
                
        return False, "", False

    def _infer_basket(self, symbol: str) -> str:
        """Fallback basket inference for symbols not in the explicit basket_map."""
        sym = symbol.upper()
        if "XAU" in sym or "GOLD" in sym:
            return "XAUUSDm"
        elif "JPY" in sym:
            return "JPY_EXT"
        elif any(idx in sym for idx in ["DE30", "US30", "NAS100", "US500", "UK100"]):
            return "INDICES"
        return "MAJOR_FX"

    def check_strategy_governance(self, strategy_id: str) -> Tuple[bool, str]:
        """
        Institutional Strategy-Specific Circuit Breaker.
        Checks if a strategy has exceeded its 48-hour trailing loss limit.
        
        HARDENED: Auto-cooldown after configurable period (default 24h).
        If cooldown_hours > 0, halted strategies auto-recover after the cooldown.
        """
        if self.strategy_status.get(strategy_id) == "HALTED":
            # AUTO-COOLDOWN: Check if enough time has passed since halt
            if self.circuit_breaker_cooldown_hours > 0:
                halt_time = self.strategy_status.get(f"{strategy_id}_halt_time", 0)
                elapsed_hours = (time.time() - halt_time) / 3600.0
                if elapsed_hours >= self.circuit_breaker_cooldown_hours:
                    self.strategy_status[strategy_id] = "OK"
                    del self.strategy_status[f"{strategy_id}_halt_time"]
                    self._save_health_state()
                    self.logger.info(f"CIRCUIT BREAKER AUTO-RECOVERED: Strategy {strategy_id} unsuspended after {elapsed_hours:.1f}h cooldown.")
                else:
                    remaining = self.circuit_breaker_cooldown_hours - elapsed_hours
                    return False, f"STRATEGY_HALTED (Auto-recovery in {remaining:.1f}h)"
            else:
                return False, "STRATEGY_HALTED (Manual Reset Required)"

        # Calculate Rolling 48h Loss
        perf_history = self.strategy_performance.get(strategy_id, [])
        if not perf_history:
            return True, "OK"

        now = time.time()
        cutoff = now - (48 * 3600)
        
        # Clean old history
        recent_perf = [p for p in perf_history if p[0] > cutoff]
        self.strategy_performance[strategy_id] = recent_perf
        
        total_pnl_pct = sum(p[1] for p in recent_perf)
        
        limit = self.config.get("risk_governance", {}).get("strategy_loss_halt_pct", 3.0)
        if total_pnl_pct <= -limit:
            self.strategy_status[strategy_id] = "HALTED"
            self.strategy_status[f"{strategy_id}_halt_time"] = time.time()  # Track halt timestamp
            self._save_health_state()
            msg = f"CIRCUIT BREAKER: Strategy {strategy_id} HALTED! Trailing 48h PnL: {total_pnl_pct:.2f}%"
            self.logger.critical(msg)
            if hasattr(self, 'alerter'):
                self.alerter.send_risk_alert(f"Strategy Halted: {strategy_id}", f"Trailing 48h PnL: {total_pnl_pct:.2f}%")
            self._emit_risk_breach("CIRCUIT_BREAKER", "CRITICAL", msg, total_pnl_pct, -limit)
            return False, f"CIRCUIT_BREAKER_TRIGGERED ({total_pnl_pct:.2f}%)"

        return True, "OK"

    def _emit_risk_breach(self, breach_type: str, severity: str, message: str,
                          current_value: float, threshold: float) -> None:
        try:
            event_bus.publish_sync(RiskBreachEvent(
                breach_type=breach_type,
                severity=severity,
                message=message,
                current_value=current_value,
                threshold=threshold,
            ))
        except Exception:
            self.logger.debug("EventBus not running, skipping risk breach event")

    def record_strategy_result(self, strategy_id: str, pnl_abs: float, alloc_balance: float):
        """Records strategy-specific result and calculates relative PnL%."""
        if alloc_balance <= 0: return
        
        pnl_pct = (pnl_abs / alloc_balance) * 100
        if strategy_id not in self.strategy_performance:
            self.strategy_performance[strategy_id] = []
            
        self.strategy_performance[strategy_id].append((time.time(), pnl_pct))
        self._save_health_state()

    def _save_health_state(self):
        import sqlite3
        db_path = self.health_file.replace('.json', '.db')
        try:
            os.makedirs(os.path.dirname(db_path), exist_ok=True)
            state = {
                "performance": {sid: list(p) for sid, p in self.strategy_performance.items()},
                "status": self.strategy_status
            }
            conn = sqlite3.connect(db_path, timeout=10.0)
            # WAL mode: allows concurrent readers while the writer is active,
            # preventing 'database is locked' from the Telegram alerter thread.
            conn.execute("PRAGMA journal_mode=WAL")
            cursor = conn.cursor()
            cursor.execute('''CREATE TABLE IF NOT EXISTS risk_state (id INTEGER PRIMARY KEY, state_json TEXT)''')
            cursor.execute('''INSERT OR REPLACE INTO risk_state (id, state_json) VALUES (1, ?)''', (json.dumps(state),))
            conn.commit()
            conn.close()
        except Exception as e:
            self.logger.error(f"Failed to save health state: {e}")

    def _load_health_state(self):
        import sqlite3
        db_path = self.health_file.replace('.json', '.db')
        if os.path.exists(db_path):
            try:
                conn = sqlite3.connect(db_path, timeout=10.0)
                conn.execute("PRAGMA journal_mode=WAL")
                cursor = conn.cursor()
                cursor.execute('''CREATE TABLE IF NOT EXISTS risk_state (id INTEGER PRIMARY KEY, state_json TEXT)''')
                cursor.execute('''SELECT state_json FROM risk_state WHERE id=1''')
                row = cursor.fetchone()
                conn.close()
                if row:
                    state = json.loads(row[0])
                    self.strategy_performance = {sid: list(p) for sid, p in state.get("performance", {}).items()}
                    self.strategy_status = state.get("status", {})
                    self.logger.info("Loaded strategy health state from SQLite persistence.")
            except Exception as e:
                self.logger.error(f"Failed to load SQLite health state: {e}")

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
        """Synchronizes balance and resets daily risk counters including consecutive loss streak."""
        self.daily_loss = 0.0
        self.initial_balance = new_balance
        # Reset consecutive losses at day boundary: a streak that spans midnight should
        # not penalize the new day's position sizing.
        self.consecutive_losses = 0

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
