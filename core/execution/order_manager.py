import logging
import time
import random
import os
import numpy as np
from collections import deque
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List
from core.common.types import TradeSignal, ExecutionIntent, MarketSnapshot, ExecutionOutcome
from core.execution.stochastic_kernel import StochasticKernel
from core.common.exceptions import CriticalRiskViolationError

class OrderManager:
    """
    V5-INSIGNIA Unified Execution Engine (v6-LIVE).
    Handles both LIVE MT5 execution and high-fidelity SIMULATION.
    Unifies the execution pipeline using the V5 Stochastic Kernel.
    """

    def __init__(self, config: Dict[str, Any], connection=None):
        self.config = config
        self.connection = connection # MT5Connection if live
        exe_cfg = config.get("execution", {})
        
        self.latency_ms = int(exe_cfg.get("latency_ms", 150))
        self.max_spread_pts = float(exe_cfg.get("max_spread_points", 500.0))
        
        # Deterministic RNG for reproducibility (Institutional requirement)
        self.deterministic = config.get("backtest", {}).get("deterministic", False)
        self.seed = config.get("backtest", {}).get("random_seed", 42)
        self._rng = random.Random(self.seed if self.deterministic else None)
        
        self.logger = logging.getLogger("trading_bot.execution")
        
        # Rule 3.1: ShadowFill Metrics & Audit
        # INSTITUTIONAL DRY: Path driven from config, not hardcoded
        self.audit_log = config.get("paths", {}).get("shadow_fill_audit", "logs/shadow_fill_audit.csv")
        self._ensure_audit_log()
        self.slippage_diffs = []
        self.latency_diffs = []
        self.recent_spreads = deque(maxlen=100) # For Z-Score
        self.degradation_factor = 1.0 # 1.0 = Normal, 0.5 = Reduced
        
        # Rule 1.1: Unified Execution Kernel
        self.kernel = StochasticKernel(global_seed=self.seed)

    def _ensure_audit_log(self):
        if not os.path.exists("logs"): os.makedirs("logs")
        if not os.path.exists(self.audit_log):
            with open(self.audit_log, "w") as f:
                # Expanded Forensic Header (Phase 1 Validation)
                f.write("timestamp,symbol,strategy_id,intent_hash,snapshot_id,snapshot_hash,bid,ask,spread,spread_zscore,regime,sim_fill,actual_fill,signed_drift,absolute_drift,sim_latency,actual_latency,outcome\n")

    def get_degraded_volume(self, volume: float) -> float:
        """Rule 3.2: Auto-Degradation Logic."""
        return volume * self.degradation_factor

    def execute_signal(self, 
                       signal: TradeSignal, 
                       symbol: str, 
                       price_data: Dict[str, float],
                       is_news_blocked: bool = False,
                       magic: int = None,
                       comment: str = "V5-INSIGNIA",
                       timestamp: float = None) -> Optional[Dict[str, Any]]:
        """
        Processes a TradeSignal with institutional realism (Spread, News, Latency).
        Routes to Live MT5 if connection is present, otherwise Simulates via Kernel.
        """
        if signal.direction == "NONE":
            return None

        # 1. News Blockade
        if is_news_blocked:
            self.logger.warning(f"Execution REJECTED: News Event Active for {symbol}")
            return None

        # 2. Institutional Live Path
        if self.connection and not self.config.get("backtest", {}).get("enabled", False):
            # --- PHASE 1 HARD BLOCK (NON-BYPASSABLE) ---
            lot_to_execute = self.get_degraded_volume(getattr(signal, 'volume', 0.01))
            
            # Enforce minimum lot size (MT5 broker minimum)
            min_lot = self.config.get("symbols_config", {}).get(symbol, {}).get("min_lot", 0.01)
            if lot_to_execute < min_lot:
                self.logger.info(f"Lot {lot_to_execute} below min {min_lot}, setting to min")
                lot_to_execute = min_lot
            
            if lot_to_execute > 0.05:
                # INSTITUTIONAL: Raise exception for graceful shutdown instead of sys.exit().
                # This allows the orchestrator to close connections, flush logs, and
                # execute emergency flatten before process termination.
                self.logger.critical(f"PHASE 1 SAFETY VIOLATION: Intent {lot_to_execute} > 0.05! Raising CriticalRiskViolationError.")
                raise CriticalRiskViolationError(
                    lot_size=lot_to_execute,
                    max_allowed=0.05,
                    symbol=symbol,
                    strategy_id=getattr(signal, 'strategy_id', 'V6_LIVE'),
                    detail=f"PHASE 1 LOT VIOLATION: {lot_to_execute:.4f} > 0.05 on {symbol}"
                )

            # 2.0: Generate Parallel Shadow (Simulated) Fill for Audit
            intent = ExecutionIntent(
                symbol=symbol,
                direction=signal.direction,
                volume=getattr(signal, 'volume', 0.01),
                stop_loss=signal.stop_loss,
                take_profit=signal.take_profit,
                strategy_id=getattr(signal, 'strategy_id', "V6_LIVE"),
                setup_timestamp=time.time()
            )
            snapshot = MarketSnapshot(
                timestamp=time.time(),
                bid=price_data.get('bid'),
                ask=price_data.get('ask'),
                spread=price_data.get('ask') - price_data.get('bid'),
                point=price_data.get('point', 0.00001),
                dfs=price_data.get('dfs', 1.0),
                volatility=price_data.get('volatility', 'NORMAL')
            )
            sim_outcome = self.kernel.execute(intent, snapshot)

            # Live execution via MT5Connection with Institutional Retry Queue
            start_time = time.time()
            max_retries = 3
            backoff = 0.5
            
            # Resolve magic for position deduplication
            eval_magic = magic if magic is not None else self.config.get("magic_number", 234000)
            
            # PRE-FLIGHT: Snapshot active position tickets for this magic/symbol
            pre_positions = self.connection.get_positions(symbol=symbol)
            pre_tickets = {p.ticket for p in pre_positions if p.magic == eval_magic} if pre_positions else set()
            
            result = None
            for attempt in range(max_retries):
                result = self.connection.place_order(
                    symbol=symbol,
                    signal=signal,
                    lot_size=lot_to_execute,
                    magic=magic,
                    comment=comment
                )
                
                if result and not result.get("is_error", False):
                    break
                    
                err_msg = result.get("error", "Unknown MT5 Rejection") if result else "None/Timeout"
                self.logger.warning(f"Execute attempt {attempt+1}/{max_retries} failed on {symbol}: {err_msg}")
                
                # INSTITUTIONAL ORPHAN HOOK: Check if terminal processed it despite returning failure
                time.sleep(0.5) # Give MT5 brief window to sync state
                post_positions = self.connection.get_positions(symbol=symbol)
                if post_positions:
                    post_tickets = [p for p in post_positions if p.magic == eval_magic and p.ticket not in pre_tickets]
                    if post_tickets:
                        new_pos = post_tickets[0]
                        self.logger.warning(f"ORPHAN INTERCEPT: Ticket {new_pos.ticket} found despite timeout. Deduplicating.")
                        result = {
                            "is_error": False,
                            "ticket": new_pos.ticket,
                            "volume": new_pos.volume,
                            "price": new_pos.price_open
                        }
                        break
                        
                if attempt < max_retries - 1:
                    time.sleep(backoff)
                    backoff *= 2.0
            
            # Rule 3: Forensic Reconciliation & Audit (Phase 1 Hardening)
            if result and not result.get("is_error", False):
                latency = (time.time() - start_time) * 1000.0
                actual_fill = result.get("price")
                point = price_data.get('point', 0.00001)
                
                # 3.1: Drift Calculations (Signed, Absolute, Tail)
                signed_drift = (actual_fill - sim_outcome.fill_price) / point if point > 0 else 0
                absolute_drift = abs(signed_drift)
                latency_diff = abs(latency - sim_outcome.actual_latency_ms)
                
                # 3.2: Spread Z-Score
                current_spread = snapshot.spread / point
                self.recent_spreads.append(current_spread)
                mean_s = np.mean(self.recent_spreads)
                std_s = np.std(self.recent_spreads) if len(self.recent_spreads) > 1 else 1.0
                spread_z = (current_spread - mean_s) / (std_s if std_s > 0 else 1.0)
                
                # 3.3: Execution Regime Tagging
                dt_now = datetime.now(timezone.utc)
                is_rollover = 21 <= dt_now.hour <= 22 # 21:55-22:05 typical
                regime = "ROLLOVER" if is_rollover else f"{snapshot.volatility}_VOL"
                
                self.slippage_diffs.append(absolute_drift)
                if len(self.slippage_diffs) > 50: self.slippage_diffs.pop(0)
                
                # Rule 3.2: Auto-Degradation Trigger (Threshold: 0.2 pip drift on P95)
                p95_drift = np.percentile(self.slippage_diffs, 95) if self.slippage_diffs else 0
                if p95_drift > 0.2:
                    self.degradation_factor = 0.5
                    self.logger.warning(f"[SHADOW] High Slippage Drift Detected ({p95_drift:.2f} pips). AUTO-DEGRADING.")
                else:
                    self.degradation_factor = 1.0

                # 3.4: Log Forensic Data (Phase 1 Mandatory Fields)
                with open(self.audit_log, "a") as f:
                    f.write(f"{time.time()},{symbol},{intent.strategy_id},{intent.intent_hash},"
                            f"{snapshot.snapshot_id},{snapshot.snapshot_id},{snapshot.bid},{snapshot.ask},"
                            f"{current_spread:.2f},{spread_z:.2f},{regime},{sim_outcome.fill_price},"
                            f"{actual_fill},{signed_drift:.4f},{absolute_drift:.4f},"
                            f"{sim_outcome.actual_latency_ms:.1f},{latency:.1f},SUCCESS\n")
                
                result["outcome"] = sim_outcome # Attach for trace
                return result
                    
            return None

        # 3. Unified Simulation Path (Rule 1.1: Hardware Parity)
        # We prepare the SNAPSHOT and INTENT to pass to the Grade-A+ Kernel
        intent = ExecutionIntent(
            symbol=symbol,
            direction=signal.direction,
            volume=getattr(signal, 'volume', 0.01),
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
            strategy_id=getattr(signal, 'strategy_id', "V6_LIVE"),
            setup_timestamp=timestamp or time.time()
        )
        
        # Prepare Snapshot (Simulated)
        snapshot = MarketSnapshot(
            timestamp=timestamp or time.time(),
            bid=price_data.get('bid'),
            ask=price_data.get('ask'),
            spread=price_data.get('ask') - price_data.get('bid'),
            point=price_data.get('point', 0.00001),
            dfs=price_data.get('dfs', 1.0),
            volatility=price_data.get('volatility', 'NORMAL'),
            metadata=price_data.get('metadata', {})
        )
        
        # Kernel Execution (Standardized)
        outcome = self.kernel.execute(intent, snapshot)
        
        return {
            "ticket": self._rng.randint(1000000, 9999999), 
            "symbol": symbol,
            "direction": signal.direction,
            "fill_price": outcome.fill_price,
            "actual_slippage_pips": outcome.actual_slippage_pips,
            "sl": intent.stop_loss,
            "tp": intent.take_profit,
            "lots": intent.volume,
            "timestamp": outcome.timestamp,
            "execution_drag": outcome.execution_drag,
            "outcome": outcome, # Preserve for Audit
            "is_error": False
        }

    def simulate_exit(self, ticket: int, exit_type: str, price: float, point: float, direction: str = "BUY", volume: float = 0.01, exit_time: float = None) -> Dict[str, Any]:
        """
        Simulates an exit event (SL/TP) utilizing the robust StochasticKernel.
        SLs are modeled as full market orders (impact + slippage).
        TPs are modeled as limit orders (bypassing liquidity latency/impact spikes).
        """
        from core.common.types import ExecutionIntent, MarketSnapshot
        
        sim_time = exit_time if exit_time is not None else time.time()
        exit_dir = "SELL" if direction == "BUY" else "BUY"
        
        # Determine pseudo-metadata for realistic stochastic friction
        base_slip = 1.0 if exit_type == "sl" else 0.1
        obi_val = self._rng.uniform(-0.5, 0.5) if exit_type == "sl" else 0.0 # TPs ignore OBI friction
        
        intent = ExecutionIntent(
            symbol="EXIT_SIM",
            direction=exit_dir,
            volume=volume,
            stop_loss=0.0,
            take_profit=0.0,
            strategy_id=f"EXIT_{exit_type.upper()}",
            setup_timestamp=sim_time
        )
        
        snapshot = MarketSnapshot(
            timestamp=sim_time,
            bid=price,
            ask=price,
            spread=point * 1.0, 
            point=point,
            dfs=1.0,
            volatility="HIGH" if exit_type == "sl" else "NORMAL",
            metadata={"base_slippage_points": base_slip, "obi": obi_val}
        )
        
        outcome = self.kernel.execute(intent, snapshot)
        
        return {
            "ticket": ticket,
            "exit_price": outcome.fill_price,
            "exit_type": exit_type,
            "exit_time": outcome.timestamp,
            "execution_drag": outcome.execution_drag,
            "microstructure_loss": outcome.microstructure_loss
        }

if __name__ == "__main__":
    # Institutional Standalone Test
    logging.basicConfig(level=logging.INFO)
    test_config = {
        "execution": {"latency_ms": 100, "max_spread_points": 500},
        "backtest": {"deterministic": True, "random_seed": 42}
    }
    manager = OrderManager(test_config)
    print("V6-LIVE OrderManager initialized successfully.")
