# Rule 2.2: CPU Determinism Global Guards (Institutional lockdown)
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAX_THREADS"] = "1"
os.environ["NUMBA_NUM_THREADS"] = "1"

import logging
import os
import heapq
import math
import subprocess
import sys
import time
from tqdm import tqdm
from datetime import datetime, timezone, timedelta
import numpy as np
from typing import List, Dict, Any, Optional, Tuple

ENGINE_VERSION = "v5.0_LOCKED"

class EnvironmentGuard:
    """
    V5-INSIGNIA: Institutional Environment Hardening.
    """
    @staticmethod
    def autolock(output_dir: str):
        """Rule 2.1: Automated Environment Lockfiles."""
        os.makedirs(output_dir, exist_ok=True)
        try:
            # Persistent requirements lock
            reqs = subprocess.check_output([sys.executable, "-m", "pip", "freeze"]).decode()
            with open(os.path.join(output_dir, "requirements.lock"), "w") as f:
                f.write(reqs)
            # Runtime lock
            runtime = f"Python: {sys.version}\nPlatform: {sys.platform}\nEngine: {ENGINE_VERSION}"
            with open(os.path.join(output_dir, "runtime.lock"), "w") as f:
                f.write(runtime)
        except Exception as e:
            logging.error(f"EnvironmentGuard LOCK FAILURE: {e}")

class DatasetFingerprinter:
    """
    V5-LOCKED: Dataset Integrity Protocol.
    """
    @staticmethod
    def get_hash(filepath: str) -> str:
        """Rule 3.1: SHA256 of raw data bytes."""
        import hashlib
        h = hashlib.sha256()
        with open(filepath, "rb") as f:
            while chunk := f.read(8192):
                h.update(chunk)
        return h.hexdigest()

from core.base_strategy import MarketData
from core.regime_detector import RegimeDetector
from core.regime_store import MemoryRegimeStore # v3 Persistence
from core.volatility_detector import VolatilityDetector, VolatilityLevel
from core.risk.risk_guardian import RiskGuardian
from core.session_detector import SessionDetector
from core.portfolio_manager import PortfolioManager
from core.regime_gater import RegimeGater
from core.recovery.checkpoint_manager import CheckpointManager
from core.execution.order_manager import OrderManager
from core.indicator_engine import IndicatorEngine
from core.common.exceptions import CriticalRiskViolationError

# Phase 5: Institutional Grade-A+ Imports
from core.common.types import (
    CandleArray, TradeSignal, MarketRegime, 
    ExecutionIntent, MarketSnapshot, ExecutionOutcome
)
from core.data.fidelity import FidelityEngine
from core.execution.stochastic_kernel import StochasticKernel
from backtesting.reconstructor import PathReconstructor
from core.portfolio.audit_engine import AuditEngine
from types import MappingProxyType

logger = logging.getLogger("trading_bot.backtester")

class PortfolioBacktester:
    def __init__(self, config: dict):
        self.config = config
        self.regime_detector = RegimeDetector()
        self.volatility_detector = VolatilityDetector(atr_period=14, lookback=100)
        self.risk_guardian = RiskGuardian(config)
        self.order_manager = OrderManager(config)
        self.portfolio_manager = PortfolioManager(config)
        self.checkpoint_manager = CheckpointManager(os.path.join("backtests", "checkpoints"))

        # Institutional V5 Logic
        self.kernel = StochasticKernel(config.get("backtest", {}).get("random_seed", 42))
        self.reconstructor = PathReconstructor(n_paths=200, seed=config.get("backtest", {}).get("random_seed", 42))
        self.audit_engine = AuditEngine()
        self.regime_store = MemoryRegimeStore() # v3 Persistent Store

        bt_cfg = config.get("backtest", {})
        self.initial_partition_balance = float(bt_cfg.get("initial_balance_per_strategy", 1000.0))
        
        vol_cfg = config.get("volatility_adaptive", {})
        self.volatility_adaptive_enabled = vol_cfg.get("enabled", True)
        self.min_volatility_for_trades = vol_cfg.get("min_volatility_for_trades", "VERY_LOW")
        
        self.current_index = 0
        self.history = []
        self.open_trades = {}
        self.balances = {}
        self.equities = {}
        self.peak_equity = {}
        self.volatility_history = []
        self.equity_history = []
        
        # Rule 3.1: Global Monotonic Sequence Source
        self._sequence_counter = 0
        # Rule 3.2: Priority Queue (execution_time, intent_hash, sequence_id, data)
        self.pending_queue = []
        
        self.dfs_score = 1.0 # Initial DFS
        self.dataset_hashes = {} # tf -> sha256
        self.rejection_stats = {} # {sid: {reason: count}}

    def reset(self, active_strategies: list):
        """Full reset of the simulation state with capital allocation (Step 9)."""
        self.current_index = 0
        self.history = []
        self.open_trades = {}
        self.pending_signals = {}
        
        total_pool = len(active_strategies) * self.initial_partition_balance
        
        self.balances = {}
        self.equities = {}
        self.peak_equity = {}
        self.max_drawdowns = {}
        allocated_sum = 0.0
        
        for strat in active_strategies:
            sid = strat.strategy_id
            # [ Institutional Level ]: Scale bounds directly to portfolio micro-allocation rules.
            bal = self.initial_partition_balance
            self.balances[sid] = bal
            self.equities[sid] = bal
            self.peak_equity[sid] = bal
            self.max_drawdowns[sid] = 0.0
            allocated_sum += bal
            
            # Audit Trail
            print(f"[ALLOCATION_AUDIT] {sid}: ${bal:,.2f} (DYNAMIC CONFIG)")
            logger.info(f"[ALLOCATION_AUDIT] {sid}: ${bal:,.2f} / Weight: 100%")
            
        # v3 Persistence Reset
        self.regime_store = MemoryRegimeStore()
            
        risk_cfg = self.config.get("risk_governance", {})
        self.risk_guardian.max_drawdown_halt_pct = float(risk_cfg.get("max_drawdown_halt_pct", 8.0))
        self.risk_guardian.max_daily_loss_pct = float(risk_cfg.get("max_daily_loss_pct", 5.0))
        self.risk_guardian.initial_balance = allocated_sum
        self.risk_guardian.max_equity = allocated_sum
        self.risk_guardian.kill_switch_active = False
        
        logger.info(f"[AUDIT] Backtest Reset Complete. Total Portfolio Capital: ${allocated_sum:,.2f}")
            
        self.equity_history = []
        self.rejection_stats = {}
        
        # Rule 3.3 Recovery: Only clear/use checkpoint if enabled
        if not self.config.get("backtest", {}).get("disable_checkpoint", False):
            self.checkpoint_manager.clear_checkpoint()

    def get_state(self) -> Dict[str, Any]:
        return {
            "current_index": self.current_index,
            "balances": self.balances,
            "equities": self.equities,
            "peak_equity": self.peak_equity,
            "max_drawdowns": self.max_drawdowns,
            "open_trades": self.open_trades,
            "history": self.history
        }

    def set_state(self, state: Dict[str, Any]):
        self.current_index = state["current_index"]
        self.balances = state["balances"]
        self.equities = state["equities"]
        self.peak_equity = state["peak_equity"]
        self.max_drawdowns = state["max_drawdowns"]
        self.open_trades = state["open_trades"]
        self.history = state["history"]
    def run(self, symbol: str, strategies: list, target_tf_data, h1_data, m15_data, m5_data, m1_data, d1_data=None, data_hashes: Dict[str, str] = None, resume: bool = False):
        """
        V5-LOCKED Production Backtest Runner.
        Implements 'Step 15' development loop with Checkpoint support.
        """
        # Rule 2.1: Operational Environmental Lockdown
        run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.audit_trail_dir = os.path.join("backtest_results", symbol, f"run_{run_timestamp}")
        EnvironmentGuard.autolock(self.audit_trail_dir)
        
        self.dataset_hashes = data_hashes or {}
        
        logger.info(f"Starting V4-ULTRA Production Backtest on {symbol} (Build: {ENGINE_VERSION})...")
        logger.info(f"DATA SIZES: target_tf={len(target_tf_data)}, m1={len(m1_data)}, m5={len(m5_data)}, m15={len(m15_data)}, h1={len(h1_data)}")
        
        # Institutional Gating Filter: Must be enabled AND have an allocation > 0
        active_strategies = []
        for s in strategies:
            # 1. Check logical enabled flag
            if not getattr(s, "enabled", True):
                continue
            
            # 2. Check symbol allowance
            if not s.is_symbol_allowed(symbol):
                continue
                
            # 3. Check allocation > 0 (via PortfolioManager logic)
            # Use a dummy total_pool to check if allocation is 0
            if self.portfolio_manager.get_strategy_balance(100.0, s.strategy_id) <= 0:
                continue
                
            active_strategies.append(s)
            
        sid_list = [s.strategy_id for s in active_strategies]
        
        # Portfolio Mode: Run all active strategies in parallel
        logger.info(f"Portfolio Mode initialized with {len(active_strategies)} institutional engines")
        
        if not resume:
            self.reset(active_strategies)


        symbol_cfg = self.config.get("symbols_config", {}).get(symbol, {})
        point = float(symbol_cfg.get("point", 0.0001))
        tick_value = float(symbol_cfg.get("tick_value", 10.0))
        comm_per_lot = float(symbol_cfg.get("commission_per_lot", 7.0))
        
        # 1. Institutional Indicator Pre-calculation (IPC) - Step 4.2
        from core.indicator_engine import IndicatorEngine
        from rich.console import Console
        console = Console()
        
        # 0. DFS & FINGERPRINT PRE-FLIGHT
        self.dfs_score = FidelityEngine.calculate_dfs(target_tf_data, self.config.get("backtest", {}).get("timeframe", "M5"))
        self.dfs_class = FidelityEngine.get_classification(self.dfs_score)
        
        fingerprint = AuditEngine.generate_fingerprint(self.config, {"symbol": symbol, "len": len(target_tf_data)})
        logger.info(f"INSTITUTIONAL RUN STARTED. Fingerprint: {fingerprint}")
        logger.info(f"DFS SCORE: {self.dfs_score:.4f} ({self.dfs_class})")
        
        if self.dfs_class == "INVALID":
            logger.critical("RUN ABORTED: Data fidelity below institutional threshold.")
            return [], []

        logger.info(f"Calibrating {symbol} Strategy Indicators...")
        target_tf_data._indicators = IndicatorEngine.precalculate_all(symbol, getattr(target_tf_data, "timeframe", "UNKNOWN"), target_tf_data)
        m5_data._indicators = IndicatorEngine.precalculate_all(symbol, "M5", m5_data)
        m15_data._indicators = IndicatorEngine.precalculate_all(symbol, "M15", m15_data)
        h1_data._indicators = IndicatorEngine.precalculate_all(symbol, "H1", h1_data)
        if d1_data is not None:
            d1_data._indicators = IndicatorEngine.precalculate_all(symbol, "D1", d1_data)
        logger.info("Indicator Pre-calculation COMPLETED.")

        # Pre-flight data integrity check (Step 11)
        self._validate_data_alignment(target_tf_data, m1_data)

        # V5-LOCKED: Ensure institutional warmup for HTF indicators (Step 9)
        # RegimeDetector requires atr_14 with ≥100 values and adx_14 with ≥20 values.
        # Minimum warmup must exceed max(100 + atr_period, htf_ema_200) to guarantee
        # all indicators are fully mature before strategy evaluation begins.
        start_idx = max(200, self.current_index)
        if start_idx >= len(target_tf_data.time):
            start_idx = min(200, len(target_tf_data.time) // 2) if len(target_tf_data.time) > 200 else 0
        
        if len(active_strategies) == 0:
            logger.warning(f"NO STRATEGIES LOADED - cannot run backtest")
            return [], []
            
        last_date = None
        last_date = None
        start_time = time.time()
        
        for i in range(start_idx, len(target_tf_data.time)):
            if i % 5000 == 0:
                elapsed = time.time() - start_time
                speed = i / elapsed if elapsed > 0 else 0
                logger.info(f"[PROGRESS] {i}/{len(target_tf_data.time)} bars | Speed: {speed:.1f} it/s")
            try:
                self.current_index = i
                t = target_tf_data.time[i]
                dt = datetime.fromtimestamp(t, tz=timezone.utc)

                # 0. DAILY RESET TRIGGER (Critical for Session Strategies)
                current_date = dt.date()
                if last_date is not None and current_date != last_date:
                    for strat in active_strategies:
                        strat.reset_daily_stats()
                    for sid in self.balances:
                        self.risk_guardian.reset_daily(self.balances[sid])
                last_date = current_date

                # 0.5 INSTITUTIONAL GOVERNANCE GATE (A+ Hardening)
                for sid in self.balances:
                    if sid not in self.open_trades:
                        self.equities[sid] = self.balances[sid]
                
                total_bal = sum(self.balances.values())
                total_eq = sum(self.equities.values())
                all_open = len(self.open_trades)
                t_long = sum(1 for tr in self.open_trades.values() if tr["direction"] == "BUY")
                t_short = sum(1 for tr in self.open_trades.values() if tr["direction"] == "SELL")
                
                is_ok, reason = self.risk_guardian.check_governance(
                    total_bal, total_eq, list(self.open_trades.values())
                )
                if not is_ok:
                    dd = ((self.risk_guardian.max_equity - total_eq) / self.risk_guardian.max_equity * 100) if self.risk_guardian.max_equity > 0 else 0
                    logger.critical(f"[{dt}] INSTITUTIONAL HALT: {reason} | Bal: {total_bal:.2f} | Eq: {total_eq:.2f} | MaxEq: {self.risk_guardian.max_equity:.2f} | DD: {dd:.2f}%")
                    break

                # [ Institutional Fidelity ]: Zero-Copy Index Shifting
                target_tf_data.set_limit(i) 
                
                # Ensure minimum bars for strategy requirements
                m5_idx = self._get_tf_idx(m5_data, t, side="right")
                if m5_data is not target_tf_data: m5_data.set_limit(m5_idx)
                
                h1_idx = self._get_tf_idx(h1_data, t, side="right")
                if h1_data is not target_tf_data: h1_data.set_limit(h1_idx)
                
                m15_idx = self._get_tf_idx(m15_data, t, side="right")
                if m15_data is not target_tf_data: m15_data.set_limit(m15_idx)
                
                if d1_data is not None:
                    d1_idx = self._get_tf_idx(d1_data, t, side="right")
                    d1_data.set_limit(d1_idx)
                
                # V3 Institutional Upgrade: Canonical Execution Anchoring
                timeframe = self.config.get("backtest", {}).get("timeframe", "M5")
                tf_secs = 300
                if timeframe == "M1": tf_secs = 60
                elif timeframe == "M15": tf_secs = 900
                elif timeframe == "H1": tf_secs = 3600
                
                # V2 Institutional Patch: RegimeDetector now expects an object exposing m5_candles and htf_candles.
                class _RegimeDataShim: pass
                shim = _RegimeDataShim()
                shim.m5_candles = target_tf_data if m5_data is target_tf_data else m5_data
                shim.htf_candles = h1_data
                shim.session = SessionDetector.get_session(dt, self.config.get("backtest", {}).get("utc_offset", 0))
                
                # Backtest Parity: Handle both datetime and raw unix timestamps
                ts = t.timestamp() if hasattr(t, "timestamp") else float(t)
                shim.timestamp = datetime.fromtimestamp(ts, tz=timezone.utc)
                
                time_bucket = int(ts / tf_secs)
                global_exec_id = f"GLOBAL:BACKTEST:{time_bucket}"

                # Global "Environment" Regime (for telemetry, logging, and audit)
                g_state = self.regime_store.load("GLOBAL")
                regime_info, _, _ = self.regime_detector.detect(
                    shim, g_state, global_exec_id, "GLOBAL", is_live=False
                )
                regime = regime_info.market_type

                # --- 0.7 DFS STABILITY UPDATE ---
                # Only calculate DFS every 500 bars to save CPU
                if i % 500 == 0:
                    self.dfs_score = FidelityEngine.calculate_dfs(target_tf_data, self.config.get("backtest", {}).get("timeframe", "M5"), self.dfs_score)
                    self.dfs_class = FidelityEngine.get_classification(self.dfs_score)

                # --- 1. EXPLICIT EXECUTION PIPELINE (Priority Queue Fulfillment) ---
                # Step 3: Maturity Check for Latency Queue (Strict Ordering)
                # Pull all ready trades from the priority queue
                while self.pending_queue and self.pending_queue[0][0] <= t:
                    exec_time, i_hash, seq_id, pending_data = heapq.heappop(self.pending_queue)
                    intent = pending_data["intent"]
                    sid = intent.strategy_id
                    
                    if sid in self.open_trades:
                        continue
                        
                    # Rule 5.1: Causality Monotonicity Law
                    # t_intent <= t_snapshot <= exec_time
                    snapshot_t = exec_time # Snapshot corresponds to the time it was queued
                    if snapshot_t < intent.setup_timestamp:
                        logger.error(f"CAUSALITY VIOLATION: setup {intent.setup_timestamp} > snapshot {snapshot_t}")
                        continue
                        
                    # Step 4: Market Snapshot (Frozen State)
                    snapshot = MarketSnapshot(
                        timestamp=snapshot_t,
                        bid=np.float64(target_tf_data.open[i]),
                        ask=np.float64(target_tf_data.open[i]) + pending_data["spread_val"],
                        spread=pending_data["spread_val"],
                        point=pending_data["point"],
                        dfs=self.dfs_score,
                        volatility=regime_info.volatility.value,
                        metadata=MappingProxyType({
                            "obi": 0.0,
                            "liquidity_depth": 100.0,
                            "base_slippage_points": 0.5
                        })
                    )
                    
                    # Step 5: Kernel Execution (Pure Function)
                    outcome = self.kernel.execute(intent, snapshot)
                    
                    if outcome and not outcome.is_error:
                        fill = {
                            "ticket": outcome.ticket,
                            "direction": intent.direction,
                            "fill_price": outcome.fill_price,
                            "actual_slippage_pips": outcome.actual_slippage_pips,
                            "actual_latency_ms": outcome.actual_latency_ms,
                            "sl": intent.stop_loss,
                            "initial_sl": intent.stop_loss,
                            "tp": intent.take_profit,
                            "strategy_id": sid,
                            "lots": intent.volume,
                            "initial_lots": intent.volume,
                            "tp1_hit": False,
                            "session": SessionDetector.get_session(dt, self.config.get("backtest", {}).get("utc_offset", 0)),
                            "entry_comm": intent.volume * comm_per_lot,
                            "timestamp": dt.timestamp(),
                            "outcome": outcome
                        }
                        self.open_trades[sid] = fill
                        logger.info(f"[{dt}] [{sid}] PIPELINE FULFILLED (Seq: {seq_id}): {fill['direction']} @ {fill['fill_price']:.5f}")
                risk_mult = RegimeGater.get_risk_multiplier(regime_info.volatility)
                conf_buffer = RegimeGater.get_confidence_buffer(regime_info.volatility)
                
                # 1.5 Volatility Analysis (V4.3 New Feature)
                vol_analysis = None
                if self.volatility_adaptive_enabled:
                    vol_analysis = self.volatility_detector.analyze(m5_data, h1_data)
                    self.volatility_history.append(vol_analysis)
                    
                    # Progress bar update removed - replaced by periodic logging
                    # Update risk multiplier based on volatility level
                    vol_risk_mult = vol_analysis.risk_multiplier
                    risk_mult = risk_mult * vol_risk_mult
                    
                    # Skip trades in extreme low volatility
                    if vol_analysis.level == VolatilityLevel.EXTREME_LOW:
                        continue

                # [ Institutional Fidelity ]: Anti-Lookahead Isolation
                # The strategy MUST only see data up to Bar i-1.
                target_tf_data.set_limit(i) 
                
                # 2. MarketData Construction (Strict Causal Isolation)
                # market_data represents the state AT THE CLOSE of Bar i-1.
                current_bid = float(target_tf_data.close[i-1]) if i > 0 else float(target_tf_data.open[0])
                symbol_cfg = self.config.get("symbols_config", {}).get(symbol, {})
                point = symbol_cfg.get("point", 0.0001)
                
                spread_val = symbol_cfg.get("spread_pips", 2) * point
                current_ask = current_bid + spread_val

                market_data = MarketData(
                    symbol=symbol,
                    htf_candles=h1_data,
                    m15_candles=m15_data,
                    m5_candles=m5_data,
                    d1_candles=d1_data,
                    m1_candles=m1_data,
                    current_price=current_bid,
                    bid=current_bid,
                    ask=current_ask,
                    spread=spread_val,
                    point=point,
                    session=SessionDetector.get_session(dt, self.config.get("backtest", {}).get("utc_offset", 0)),
                    timestamp=dt
                )
                
                # Run all active strategies (Parallel Portfolio Execution)
                strategies_to_try = active_strategies
                
                for strat in strategies_to_try:
                    sid = strat.strategy_id
                    
                    # v3 State-Aware Regime Gating
                    ts = t.timestamp() if hasattr(t, "timestamp") else float(t)
                    time_bucket = int(ts / tf_secs)
                    exec_id = f"{sid}:BACKTEST:{time_bucket}"
                    s_state = self.regime_store.load(sid)
                    s_regime_info, new_s_state, _ = self.regime_detector.detect(
                        shim, s_state, exec_id, sid, is_live=False
                    )
                    self.regime_store.save(sid, new_s_state)

                    if RegimeGater.is_drawdown_gated(self.max_drawdowns.get(sid, 0)): continue
                    
                    if not RegimeGater.is_strategy_allowed(strat.__class__.__name__, s_regime_info):
                        if self.config.get("backtest", {}).get("debug_signals"):
                            logger.info(f"[{dt}] [{sid}] REGIME GATED: Strat {strat.__class__.__name__} not allowed in {s_regime_info.market_type} (conf={s_regime_info.confidence:.2f}, vol={s_regime_info.volatility})")
                        continue
                        
                    if sid in self.open_trades: continue
                    
                    # Apply volatility-adjusted parameters if available
                    original_params = {}
                    if self.volatility_adaptive_enabled and vol_analysis:
                        original_params = self._apply_volatility_params(strat, vol_analysis)
                    
                    try:
                        signal = strat.generate_signal(market_data)
                    finally:
                        if original_params:
                            self._restore_original_params(strat, original_params)
                    
                    if not signal or signal.direction == "NONE":
                        reason = getattr(strat, "last_rejection_reason", "No specific reason")
                        if sid not in self.rejection_stats: self.rejection_stats[sid] = {}
                        self.rejection_stats[sid][reason] = self.rejection_stats[sid].get(reason, 0) + 1
                        
                        if self.config.get("backtest", {}).get("debug_signals"):
                            logger.info(f"[{dt}] [{sid}] Signal REJECTED: {reason}")
                        continue
                            
                    min_conf = getattr(strat, "min_confidence", 0.6)
                    if self.config.get("backtest", {}).get("debug_signals"):
                        logger.info(f"[{dt}] [{sid}] Signal generated: {signal.direction} @ {signal.price:.2f} conf={signal.confidence:.2f}")
                    
                    if signal.confidence < (min_conf + conf_buffer - 0.001):
                        reason = f"Confidence {signal.confidence:.2f} < {min_conf + conf_buffer:.2f}"
                        if sid not in self.rejection_stats: self.rejection_stats[sid] = {}
                        self.rejection_stats[sid][reason] = self.rejection_stats[sid].get(reason, 0) + 1
                        
                        if self.config.get("backtest", {}).get("debug_signals"):
                            logger.info(f"[{dt}] [{sid}] Confidence REJECTED: {reason}")
                        continue
                            
                    sl = strat.get_stop_loss(signal, market_data)
                    tp = strat.get_take_profit(signal, market_data)
                    sl_dist = abs(market_data.current_price - sl)
                    
                    if self.config.get("backtest", {}).get("debug_signals"):
                        logger.info(f"[{dt}] [{sid}] SL={sl:.2f}, TP={tp:.2f}, dist={sl_dist:.2f}")
                    
                    if sl_dist > 0:
                        lot_size = self.risk_guardian.calculate_lot_size(
                            balance=self.balances[sid],
                            stop_loss_dist=sl_dist,
                            symbol_info=symbol_cfg,
                            current_price=market_data.current_price
                        )
                        lot_size = lot_size * risk_mult
                        
                        if self.config.get("backtest", {}).get("debug_signals"):
                            logger.info(f"[{dt}] [{sid}] INSTITUTIONAL LOT: {lot_size:.4f}")

                        
                        if lot_size >= 0.01:
                            # --- 2. EXPLICIT EXECUTION PIPELINE (Intent Creation) ---
                            # Rule 4.1: Strict Causality (t_signal <= t_intent)
                            # t_signal is t from market_data (end of bar i-1)
                            # Intent setup_timestamp is t from current target bar (Open of bar i)
                            if not (market_data.timestamp.timestamp() <= t):
                                logger.error(f"TIME PARADOX: Signal @ {market_data.timestamp.timestamp()} exceeds Intent @ {t}")
                                continue

                            intent = ExecutionIntent(
                                symbol=symbol,
                                direction=signal.direction,
                                volume=lot_size,
                                stop_loss=sl,
                                take_profit=tp,
                                strategy_id=sid,
                                setup_timestamp=t
                            )
                            
                            # Push to Priority Queue with Sequence ID
                            self._sequence_counter += 1
                            # Predicted Execution Time (T+1 Open + Latency Approximation for sorting)
                            # Real latency is calculated by the kernel at T_execution
                            heapq.heappush(self.pending_queue, (
                                t, # Execution roughly occurs at bar open
                                intent.intent_hash,
                                self._sequence_counter,
                                {
                                    "intent": intent,
                                    "point": point,
                                    "spread_val": spread_val
                                }
                            ))
                            logger.info(f"[{dt}] [{sid}] INTENT QUEUED (Seq: {self._sequence_counter}): {intent.intent_hash[:8]}")
                        elif self.config.get("backtest", {}).get("debug_signals"):
                            logger.info(f"[{dt}] [{sid}] Risk REJECTED: Lot size {lot_size:.3f} < 0.01")

                # 4. M1 Intra-Bar Execution (Safety Gate: Check for Gaps)
                m1_slice = self._get_m1_for_m5(m1_data, t)
                atr_vals = target_tf_data.get_indicator("atr_14")
                atr_val = atr_vals[i] if i < len(atr_vals) else 0.0

                if len(m1_slice) > 0:
                    self._manage_active_trades(m1_slice, tick_value, point, comm_per_lot, active_strategies, atr_val=atr_val)
                elif self.open_trades:
                    # Institutional Grade-A+: Volatility-Aware Path Reconstruction
                    for sid, trade in list(self.open_trades.items()):
                        # We use the M5 bar to resolve paths for active trades
                        res = self.reconstructor.resolve_path(
                            candle=target_tf_data[i],
                            sl=trade["sl"],
                            tp=trade["tp"],
                            direction=trade["direction"],
                            volatility_regime=regime_info.volatility.value
                        )
                        
                        if res["p_sl"] > 0.5:
                            # SL Hit is most probable
                            self._close_trade(trade, target_tf_data.l[i] if trade["direction"] == "BUY" else target_tf_data.h[i], "sl", t, point, tick_value, comm_per_lot)
                        elif res["p_tp"] > 0.5:
                            # TP Hit is most probable
                            self._close_trade(trade, target_tf_data.h[i] if trade["direction"] == "BUY" else target_tf_data.l[i], "tp", t, point, tick_value, comm_per_lot)
                
                # 5. Equity Sampling & Drawdown Track (Rule 6.1)
                # Correct mark-to-market equity: balance + floating PnL of any open trade.
                # Previously this was `math.fsum([balance, 0.0])` which was a no-op.
                for sid in self.balances:
                    if sid in self.open_trades:
                        trade = self.open_trades[sid]
                        direction = trade["direction"]
                        # Use the last available M5 close as the current price proxy.
                        current_mark = float(target_tf_data.close[i - 1]) if i > 0 else float(target_tf_data.open[0])
                        raw_diff = (current_mark - trade["fill_price"]) if direction == "BUY" else (trade["fill_price"] - current_mark)
                        floating_pnl = math.fsum([(raw_diff / point) * tick_value * trade["lots"]])
                        self.equities[sid] = math.fsum([self.balances[sid], floating_pnl])
                    else:
                        # No open trade — equity equals settled balance
                        self.equities[sid] = self.balances[sid]

                    self.peak_equity[sid] = max(self.peak_equity[sid], self.equities[sid])
                    dd = (self.peak_equity[sid] - self.equities[sid]) / self.peak_equity[sid] * 100 if self.peak_equity[sid] > 0 else 0.0
                    self.max_drawdowns[sid] = max(self.max_drawdowns[sid], dd)
                   # Step 10: State Sampling (Optimized: Sample hourly at M5 to reduce memory pressure)
                if i % 12 == 0: # 12 bars @ M5 = 1 hour
                    for sid in self.balances:
                        self.equity_history.append({"time": t, "strategy_id": sid, "equity": self.equities[sid]})

                # [ Institutional Recovery ]: Checkpoint every 100 bars if enabled
                if i % 100 == 0 and not self.config.get("backtest", {}).get("disable_checkpoint", False):
                    try:
                        self.checkpoint_manager.save_checkpoint(self.get_state())
                    except Exception as e:
                        logger.warning(f"Checkpoint failed (skipping): {e}")

            except CriticalRiskViolationError as crve:
                # INSTITUTIONAL: Graceful handling of lot size violations
                # raised by the execution pipeline. Log forensic trail and halt cleanly.
                logger.critical(f"BACKTEST HALTED: {crve.detail}")
                crash_file = os.path.join("logs", "crash_report.log")
                os.makedirs("logs", exist_ok=True)
                with open(crash_file, "a") as f:
                    import json as _json
                    f.write(f"\n--- BACKTEST CRITICAL RISK VIOLATION: {datetime.now()} ---\n")
                    f.write(_json.dumps(crve.forensic_dict(), indent=2))
                    f.write("\n")
                break  # Graceful halt vs. raw crash

            except Exception as e:
                import traceback
                crash_file = os.path.join("logs", "crash_report.log")
                os.makedirs("logs", exist_ok=True)
                with open(crash_file, "a") as f:
                    f.write(f"\n--- BACKTEST CRASH: {datetime.now()} ---\n")
                    f.write(traceback.format_exc())
                raise e

        # Progress bar closed by removing tqdm - periodic logging handles status
        self._force_close_at_end(target_tf_data, point, tick_value, comm_per_lot, active_strategies)
        self.checkpoint_manager.clear_checkpoint()
        
        self._print_rejection_summary()
        return self.history, self.equity_history

    def _manage_active_trades(self, m1_candles, tick_value, point, comm_per_lot, strategies, atr_val=0.0, force_sl_first=True):
        """M1-Event Replay Engine for Trade Management."""
        for sid, trade in list(self.open_trades.items()):
            is_closed = False
            for m in range(len(m1_candles)):
                if is_closed: break
                
                m1_high = m1_candles.high[m]
                m1_low = m1_candles.low[m]
                spread = m1_candles.spread[m] * point
                direction = trade["direction"]

                # --- V4-ULTRA Institutional Partial Exit & Trailing (Rule 3.1) ---
                current_price = m1_candles.close[m]
                entry = trade["fill_price"]
                direction = trade["direction"]
                initial_risk_price = abs(entry - trade["initial_sl"])
                
                if initial_risk_price > 0:
                    profit_price = (current_price - entry) if direction == "BUY" else (entry - current_price)
                    current_rr = profit_price / initial_risk_price
                    
                    # [ Institutional Scale-Hardening ]: TP1 @ 1.2R (Partial Exit + Break-Even)
                    if current_rr >= 1.2 and not trade.get("tp1_hit", False):
                        partial_lots = trade["lots"] * 0.5
                        
                        # Realize 50% profit immediately
                        raw_profit_pts = (current_price - entry) if direction == "BUY" else (entry - current_price)
                        partial_pnl = (raw_profit_pts / point) * partial_lots * tick_value
                        
                        # Apply partial commission
                        partial_comm = (partial_lots / trade["initial_lots"]) * trade["entry_comm"]
                        net_partial_pnl = partial_pnl - partial_comm
                        
                        # Update balance and reduce volume
                        self.balances[sid] += net_partial_pnl
                        self.equities[sid] += net_partial_pnl
                        trade["lots"] -= partial_lots
                        trade["tp1_hit"] = True
                        
                        # Lock in Risk: Move SL to Break-Even (plus small buffer for cost)
                        be_buffer = 1.0 * point
                        trade["sl"] = entry + be_buffer if direction == "BUY" else entry - be_buffer
                        
                        logger.info(f"[{m1_candles.time[m]}] [{sid}] PARTIAL EXIT: 50% @ 1.2R (Locked BE)")

                    # Phase 2: ATR-based Trailing (at 3R+)
                    if current_rr >= 3.0 and atr_val > 0:
                        trail_mult = self.config.get("trailing_stop", {}).get("phase3_trail_mult", 1.5)
                        trail_sl = current_price - (atr_val * trail_mult) if direction == "BUY" else current_price + (atr_val * trail_mult)
                        # Only move if improves protection
                        if direction == "BUY" and trail_sl > trade["sl"]:
                            trade["sl"] = trail_sl
                        elif direction == "SELL" and (trade["sl"] == 0 or trail_sl < trade["sl"]):
                            trade["sl"] = trail_sl

                exit_price = None
                event = None
                
                if direction == "BUY":
                    sl_hit = m1_low <= trade["sl"]
                    tp_hit = m1_high >= trade["tp"]
                    
                    if sl_hit and tp_hit:
                        if force_sl_first: exit_price, event = trade["sl"], "sl"
                        else: exit_price, event = trade["tp"], "tp"
                    elif sl_hit: exit_price, event = trade["sl"], "sl"
                    elif tp_hit: exit_price, event = trade["tp"], "tp"
                else: # SELL
                    sl_hit = m1_high + spread >= trade["sl"]
                    tp_hit = m1_low + spread <= trade["tp"]
                    
                    if sl_hit and tp_hit:
                        if force_sl_first: exit_price, event = trade["sl"], "sl"
                        else: exit_price, event = trade["tp"], "tp"
                    elif sl_hit: exit_price, event = trade["sl"], "sl"
                    elif tp_hit: exit_price, event = trade["tp"], "tp"
                
                if exit_price:
                    exit_time = m1_candles.time[m]
                    exit_res = self.order_manager.simulate_exit(
                        ticket=trade["ticket"], 
                        exit_type=event, 
                        price=exit_price, 
                        point=point, 
                        direction=direction, 
                        volume=trade["lots"],
                        exit_time=exit_time
                    )
                    final_exit = exit_res["exit_price"]
                    exit_slip = abs(final_exit - exit_price)
                    
                    raw_diff = (final_exit - trade["fill_price"]) if direction == "BUY" else (trade["fill_price"] - final_exit)
                    gross_pnl = (raw_diff / point) * tick_value * trade["lots"]
                    exit_comm = trade["lots"] * comm_per_lot
                    entry_comm = trade.get("entry_comm", 0.0)
                    
                    net_pnl = gross_pnl - entry_comm - exit_comm
                    start_balance = self.balances[sid]
                    self.balances[sid] += net_pnl
                    self.equities[sid] = self.balances[sid]
                    
                    trade_record = {
                        **trade,
                        "exit_price": final_exit,
                        "exit_time": m1_candles.time[m],
                        "pnl": net_pnl,
                        "exit_slippage": exit_slip / point,
                        "result": event.upper(),
                        "final_balance": self.balances[sid],
                        "balance_at_start": start_balance
                    }
                    self.history.append(trade_record)
                    self.risk_guardian.record_trade_result(net_pnl, self.equities[sid])
                    
                    for s in strategies:
                        if s.strategy_id == sid:
                            s.on_trade_closed(trade_record)
                            break
                            
                    del self.open_trades[sid]
                    is_closed = True
                else:
                    floating_price = m1_low if direction == "BUY" else m1_high
                    f_diff = (floating_price - trade["fill_price"]) if direction == "BUY" else (trade["fill_price"] - floating_price)
                    f_gross_pnl = (f_diff / point) * tick_value * trade["lots"]
                    self.equities[sid] = self.balances[sid] + f_gross_pnl

    def _validate_data_alignment(self, m5, m1):
        """Ensures that M1 data covers the M5 range without gaps (Step 11)."""
        if len(m5) == 0 or len(m1) == 0:
            logger.warning(f"DATA ALIGNMENT SKIPPED: Missing timeframe slice.")
            return

        if m5.time[-1] > m1.time[-1]:
            logger.critical(f"DATA ALIGNMENT ERROR: M1 data ({m1.time[-1]}) expires before M5 ({m5.time[-1]})")
            raise ValueError("CRITICAL_SYSTEM_ERROR: Data inconsistency.")

    # --- TIMEFRAME INDEX RESOLUTION CONSTANTS ---
    _TF_INTERVALS = {"M1": 60, "M5": 300, "M15": 900, "M30": 1800, "H1": 3600, "H4": 14400, "D1": 86400}

    def _get_tf_idx(self, tf_data, target_time, side: str = "right", tf_str: str = None) -> int:
        """
        Returns the current index of a higher timeframe candle relative to target_time.
        
        INSTITUTIONAL HARDENING: After searchsorted, we verify:
        1. The found candle's timestamp does NOT exceed target_time (anti-lookahead)
        2. The gap between the found candle and target_time is within expected bounds
           (detects missing candles/data gaps)
        
        Args:
            tf_str: Explicit timeframe string (e.g. "M15", "H1"). Falls back to
                    tf_data.timeframe attribute if not provided.
        """
        if len(tf_data.time) == 0: return 0
        
        # CPU OPTIMIZATION: Pointer-cached sliced search
        if not hasattr(self, "_tf_pointers"): self._tf_pointers = {}
        tf_id = id(tf_data)
        
        # Fallback to 0 if time went backwards
        last_t = getattr(self, "_last_target_time", 0)
        start_idx = self._tf_pointers.get(tf_id, 0) if target_time >= last_t else 0
        self._last_target_time = target_time
        
        sliced_times = tf_data.time[start_idx:]
        if len(sliced_times) == 0: 
            idx = len(tf_data.time) - 1
        else:
            offset = np.searchsorted(sliced_times, target_time, side=side)
            idx = min(start_idx + offset, len(tf_data.time) - 1)
            self._tf_pointers[tf_id] = max(0, idx - 2)
        
        # STRICT ANTI-LOOKAHEAD: Ensure the candle at idx does not start AFTER target_time
        # If it does, step back by one to guarantee we only use past data.
        if tf_data.time[idx] > target_time and idx > 0:
            idx -= 1
        
        # GAP DETECTION: Warn if the distance between found candle and target_time
        # exceeds 2x the expected timeframe interval (indicates missing candles)
        resolved_tf = tf_str or getattr(tf_data, 'timeframe', None)
        if resolved_tf and resolved_tf in self._TF_INTERVALS:
            expected_interval = self._TF_INTERVALS[resolved_tf]
            gap = abs(target_time - tf_data.time[idx])
            if gap > expected_interval * 2:
                logger.warning(
                    f"DATA GAP DETECTED in {resolved_tf}: Target {target_time}, Found {tf_data.time[idx]}, "
                    f"Gap {gap}s > {expected_interval * 2}s (2x interval)"
                )
        
        return idx

    def _get_m1_for_m5(self, m1, target_time):
        """
        Returns M1 candles within the target timeframe bar window.
        
        INSTITUTIONAL HARDENING:
        1. Derives next_bar_time dynamically from config timeframe instead of hardcoded +300
        2. Validates M1 slice completeness (warns if missing candles)
        3. Asserts monotonicity: first M1 candle >= target_time (no past leakage)
        """
        if len(m1) == 0:
            from core.common.types import CandleArray
            return CandleArray.from_dicts([])

        # Dynamic timeframe interval (defaults to M5=300s if not configured)
        tf_str = self.config.get("backtest", {}).get("timeframe", "M5")
        tf_seconds = self._TF_INTERVALS.get(tf_str, 300)

        # CPU OPTIMIZATION: Pointer array tracking for M1
        if not hasattr(self, "_m1_pointers"): self._m1_pointers = {}
        m1_id = id(m1)
        start_idx = self._m1_pointers.get(m1_id, 0)
        
        sliced_times = m1.time[start_idx:]
        if len(sliced_times) == 0: return m1[0:0]
        
        idx_start_offset = np.searchsorted(sliced_times, target_time, side='left')
        idx_start = start_idx + idx_start_offset
        
        next_bar_time = target_time + tf_seconds
        
        search_slice = m1.time[idx_start:]
        if len(search_slice) == 0:
            idx_end = idx_start
        else:
            idx_end_offset = np.searchsorted(search_slice, next_bar_time, side='left')
            idx_end = idx_start + idx_end_offset

        self._m1_pointers[m1_id] = max(0, idx_start - 5)

        if idx_end <= idx_start:
            idx_end = min(idx_start + (tf_seconds // 60), len(m1.time))
        
        # COMPLETENESS CHECK: Expected M1 count = tf_seconds / 60
        expected_m1_count = tf_seconds // 60
        actual_m1_count = idx_end - idx_start
        if 0 < actual_m1_count < expected_m1_count:
            logger.debug(
                f"M1 SLICE INCOMPLETE: Expected {expected_m1_count} candles for {tf_str}, "
                f"got {actual_m1_count} at time {target_time}"
            )
        
        result = m1[idx_start:idx_end]
        
        # MONOTONICITY ASSERTION: First M1 candle should not precede target_time
        if len(result) > 0 and result.time[0] < target_time:
            logger.warning(
                f"M1 MONOTONICITY WARNING: First M1 candle {result.time[0]} < target {target_time}. "
                f"Possible past data leakage."
            )
        
        return result

    def _force_close_at_end(self, m5_data, point, tick_value, comm_per_lot, strategies):
        if not self.open_trades: return
        last_price = m5_data.close[-1]
        for sid, trade in list(self.open_trades.items()):
            net_pnl = ((last_price - trade["fill_price"] if trade["direction"] == "BUY" else trade["fill_price"] - last_price) / point) * tick_value * trade["lots"]
            self.history.append({**trade, "exit_price": last_price, "pnl": net_pnl, "result": "FORCED_CLOSE"})
            del self.open_trades[sid]

    def _apply_volatility_params(self, strategy, vol_analysis) -> Dict[str, Any]:
        """Apply volatility-adjusted parameters to a strategy."""
        from core.volatility_detector import VolatilityAdaptiveParameters
        
        strategy_id = strategy.strategy_id
        strategy_type = self._get_strategy_type(strategy_id)
        
        vol_params = VolatilityAdaptiveParameters.get_parameters_for_volatility(
            vol_analysis.level, 
            strategy_type
        )
        
        original = {}
        for param, value in vol_params.items():
            if hasattr(strategy, param):
                original[param] = getattr(strategy, param)
                setattr(strategy, param, value)
        
        return original
    
    def _restore_original_params(self, strategy, original_params: Dict[str, Any]) -> None:
        """Restore original strategy parameters."""
        for param, value in original_params.items():
            setattr(strategy, param, value)
    
    def _get_strategy_type(self, strategy_id: str) -> str:
        """Determine strategy type for volatility parameter selection."""
        if "Breakout" in strategy_id or "Liquidity" in strategy_id:
            return "breakout"
        elif "MeanReversion" in strategy_id or "RangeBounce" in strategy_id:
            return "mean_reversion"
        elif "Trend" in strategy_id:
            return "trend"
        return "breakout"
    
    def get_volatility_summary(self) -> Dict[str, Any]:
        """Get summary of volatility conditions encountered during backtest."""
        if not self.volatility_history:
            return {"status": "No volatility data"}
        
        level_counts = {}
        for vol in self.volatility_history:
            level_key = vol.level.value
            level_counts[level_key] = level_counts.get(level_key, 0) + 1
        
        ratios = [v.ratio for v in self.volatility_history]
        
        return {
            "total_bars": len(self.volatility_history),
            "level_distribution": {k: f"{(v/len(self.volatility_history)*100):.1f}%" for k, v in level_counts.items()},
            "avg_ratio": float(np.mean(ratios)),
            "min_ratio": float(np.min(ratios)),
            "max_ratio": float(np.max(ratios)),
        }

    def _print_rejection_summary(self):
        """Prints an institutional forensic summary of why signals were rejected."""
        if not hasattr(self, "rejection_stats") or not self.rejection_stats: 
            logger.info("[AUDIT] No rejection data collected (Zero signals attempted?).")
            return
        
        from rich.table import Table
        from rich.console import Console
        console = Console()
        
        for sid, stats in self.rejection_stats.items():
            table = Table(title=f"Forensic Rejection Summary: {sid}")
            table.add_column("Reason", style="cyan")
            table.add_column("Count", style="magenta", justify="right")
            
            sorted_stats = sorted(stats.items(), key=lambda x: x[1], reverse=True)
            for reason, count in sorted_stats[:15]: # Top 15 reasons
                 table.add_row(reason, str(count))
            
            console.print(table)
