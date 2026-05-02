import logging
import time
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone
from dataclasses import replace
import threading

from core.strategy_runtime import StrategyRuntime
from core.portfolio_manager import PortfolioManager
from core.base_strategy import MarketData
from core.common.types import TradeSignal
from core.execution.order_manager import OrderManager

logger = logging.getLogger("trading_bot.orchestrator")

class StrategyOrchestrator:
    """
    Coordinates multiple StrategyRuntimes and manages the overall signal workflow.
    Interfaces with PortfolioManager to resolve signal conflicts and ensure 
    proper capital allocation between strategies.
    Institutional Grade: Strict decoupled communication.
    """

    def __init__(self, 
                 runtimes: List[StrategyRuntime], 
                 config: dict, 
                 order_manager: OrderManager,
                 position_manager, 
                 notification_manager,
                 broker_clock,
                 news_filter,
                 regime_store=None):
        self.runtimes = runtimes
        self.config = config
        self.order_manager = order_manager
        self.connection = order_manager.connection # Underlying connection
        self.position_manager = position_manager
        self.notification_manager = notification_manager
        self.broker_clock = broker_clock
        self.news_filter = news_filter
        self.regime_store = regime_store
        
        self.portfolio_manager = PortfolioManager(self.config)
        self.last_cycle_time: Optional[datetime] = None
        self.last_analysis: Dict[str, Any] = {}
        self._analysis_lock = threading.Lock() 

        self._open_tickets = set()
        self._scaled_tickets = set() # Track tickets that have already taken partial profits
        self._tickets_lock = threading.Lock()
        # original_sl registry: {ticket: original_sl_price} — used by manage_partials for
        # correct initial risk calculation even after SL has been moved to break-even.
        self._original_sl: Dict[int, float] = {}

        # Phase 2: Professional Execution Ceiling
        self._phase1_lot_ceiling = float(
            self.config.get("execution", {}).get("phase1_lot_ceiling", 50.0)
        )

        # Singleton RegimeDetector — avoids re-instantiating on every cycle.
        from core.regime_detector import RegimeDetector
        self._regime_detector = RegimeDetector()

        # ── LIVE WFO GATE ────────────────────────────────────────────────
        # On startup, check each strategy's last WFO outcome.
        # Any strategy whose most recent window was REJECTED is suspended
        # immediately via its risk guardian's kill-switch + CRITICAL alert.
        # This prevents a strategy that failed walk-forward from trading live.
        self._apply_wfo_gate()

        # Asynchronous Trailing Stop Management Removed (Institutional Hardening)
        # Trailing stops and partials are now handled synchronously in execute_cycle


    def resolve_block(self, block_name: str, session: Optional[str] = None, symbol: Optional[str] = None) -> Dict[str, Any]:
        """
        Universal session-aware block resolution for Orchestrator components.
        Supports per-pair overrides from DB via config["pair_options"].
        """
        block = dict(self.config.get(block_name, {}))
        if not block:
            block = {}

        if symbol:
            pair_options = self.config.get("pair_options", {})
            for key, opts in pair_options.items():
                if key.startswith(f"{symbol}:") and block_name in opts:
                    block = {**block, **opts[block_name]}
                    break

        if session and "session_overrides" in block:
            overrides = block["session_overrides"].get(session, {})
            if overrides:
                merged = block.copy()
                merged.update(overrides)
                return merged
        return block

    def _apply_wfo_gate(self):
        """
        Queries the WFO gate database and suspends any strategy whose
        most recent WFO window was classified as REJECTED.
        Runs once at startup before the first trading cycle.
        """
        from backtesting.walk_forward import WalkForwardValidator
        suspended = []
        for runtime in self.runtimes:
            sid = runtime.strategy.strategy_id
            robustness = WalkForwardValidator.get_last_wfo_robustness(sid, self.config)
            if robustness == "REJECTED":
                # Activate kill-switch on this strategy's risk guardian
                runtime.risk_guardian.kill_switch_active = True
                suspended.append(sid)
                logger.critical(
                    f"[WFO GATE] Strategy '{sid}' SUSPENDED: last WFO window was REJECTED. "
                    f"Re-run WFO to clear. Trading disabled for this strategy."
                )
                try:
                    self.notification_manager.send_risk_alert(
                        reason="WFO Gate: Strategy Suspended",
                        details=f"{sid} failed last walk-forward window (REJECTED). Trading disabled."
                    )
                except Exception:
                    pass  # Never block startup on notification failure
            elif robustness in ("MARGINAL",):
                logger.warning(
                    f"[WFO GATE] Strategy '{sid}' last WFO window was MARGINAL. "
                    f"Reduce position sizing."
                )
        if suspended:
            logger.critical(f"[WFO GATE] {len(suspended)} strateg(ies) suspended: {suspended}")
        else:
            logger.info(f"[WFO GATE] All {len(self.runtimes)} strategies passed WFO gate check.")

    def execute_cycle(self, symbol: str, market_data: MarketData, account_snapshot: dict, is_news_blocked: bool = False) -> Dict[str, Any]:
        """
        STRICT 8-STEP INSTITUTIONAL EXECUTION FLOW (Step 12).
        ====================================================
        Returns a 'Pulse Report' for dashboard telemetry.
        """
        exec_payloads = []
        symbol_info = self.connection.get_symbol_info(symbol) if hasattr(self.connection, 'get_symbol_info') else {}

        # 0. Synchronize Ticket Registry (Step 12: Integrity Check)
        self._sync_ticket_registry()
        
        # 1. ANALYZE MARKET (Hierarchical Multi-Timeframe)
        
        # 2. CANONICAL EXECUTION ANCHORING (v3 Hardening)
        # timeframe_seconds: default 300 for M5
        timeframe_seconds = 300 
        target_tf = self.config.get("backtest", {}).get("timeframe", "M5")
        if target_tf == "M1": timeframe_seconds = 60
        elif target_tf == "M15": timeframe_seconds = 900
        elif target_tf == "H1": timeframe_seconds = 3600
        
        ts_val = market_data.timestamp.timestamp()
        time_bucket = int(ts_val / timeframe_seconds)
        
        # Representative Regime for Cycle Telemetry (using GLOBAL ID)
        global_exec_id = f"GLOBAL:{market_data.session}:{time_bucket}"
        
        # If no store, we use a transient memory store as fallback
        if not self.regime_store:
            from core.regime_store import MemoryRegimeStore
            self.regime_store = MemoryRegimeStore()
            
        global_state = self.regime_store.load("GLOBAL")
        pulse_regime_info, _, _ = self._regime_detector.detect(
            market_data, global_state, global_exec_id, "GLOBAL", is_live=True
        )

        blocking_event = self.news_filter.is_blocked(symbol, ts_val)
        
        # Pulse Telemetry Initialization
        upcoming_obj = self.news_filter.get_upcoming_events(ts_val, 24)
        for ev in upcoming_obj:
            # Shift from UTC to Local System Time for Unified UI alignment
            ev['time'] = datetime.fromtimestamp(ev['timestamp'], tz=timezone.utc).astimezone().strftime("%I:%M %p")
            
        pulse_report = {
            "regime": pulse_regime_info,
            "strategies": {},
            "execution": [],
            "news_blocked": blocking_event,
            "upcoming_news": [e["title"] for e in upcoming_obj],
            "upcoming_news_obj": upcoming_obj,
            "timestamp": market_data.timestamp.astimezone().strftime("%H:%M:%S"),
            "open_tickets": list(self._open_tickets) # Dashboard visibility
        }
        
        # Update shared analysis state for background threads (Atomic Swap)
        # We create a new dict reference to minimize lock duration and contention.
        with self._analysis_lock:
            new_analysis = dict(self.last_analysis)
            new_analysis[symbol] = {
                'bid': market_data.current_price,
                'ask': market_data.current_price + (symbol_info.get("spread", 0) * symbol_info.get("point", 0)),
                'atr': pulse_regime_info.atr,
                'timestamp': time.time()
            }
            self.last_analysis = new_analysis

        # Hard Block Check: Skip if blocked by news
        if blocking_event or is_news_blocked:
            logger.info(f"Cycle Skip: {symbol} is blocked by news ({blocking_event})")
            return pulse_report
        
        # 2.5 Fetch upcoming news for AI Vetting
        news_events = self.news_filter.get_upcoming_events(market_data.timestamp.timestamp(), window_hours=24)
        
        # 3. ACTIVATE RELEVANT STRATEGIES (Regime Gating)
        from core.regime_gater import RegimeGater  # noqa: PLC0415 — kept here for lazy-import safety
        active_runtimes = []
        
        # Thread-safe snapshot of runtimes for this execution cycle
        with self._analysis_lock: 
            runtimes_snapshot = list(self.runtimes)
            
        for r in runtimes_snapshot:
            if not r.strategy.is_symbol_allowed(symbol):
                continue
            
            # --- Per-Strategy REGIME DETERMINISM (v3 Spec) ---
            sid = r.strategy_id
            exec_id = f"{sid}:{market_data.session}:{time_bucket}"
            
            # 1. Load State
            state = self.regime_store.load(sid)
            # 2. Detect with State Injection
            r_regime_info, new_state, trace = self._regime_detector.detect(
                market_data, state, exec_id, sid, is_live=True
            )
            # 3. Persist New State
            self.regime_store.save(sid, new_state)
            
            # Diagnostic: Session Gating
            allowed_sessions = r.strategy.config.get("allowed_sessions", [])
            if allowed_sessions and market_data.session not in allowed_sessions:
                if self.config.get("backtest", {}).get("debug_signals"):
                    logger.debug(f"[{symbol}] [{sid}] Skipped: Session {market_data.session} not in {allowed_sessions}")
                continue

            if RegimeGater.is_strategy_allowed(r.strategy.__class__.__name__, r_regime_info):
                # Inject strategy-specific regime into execution context
                active_runtimes.append((r, r_regime_info))
        
        # 4. GENERATE SIGNALS (Institutional Parallel Execution)
        raw_signals = {}
        import concurrent.futures
        
        def _execute_runtime(args):
            runtime, r_regime_info = args
            sid = runtime.strategy_id
            
            # [ Institutional Fidelity ]: Inject per-strategy regime into MarketData
            strat_md = replace(market_data, regime=r_regime_info.market_type.value)
            
            sig = runtime.execute_cycle(strat_md)
            metrics = runtime.strategy.get_metrics(strat_md)
            thresholds = runtime.strategy.get_thresholds()
            
            if sig and sig.direction != "NONE":
                sl = runtime.strategy.get_stop_loss(sig, market_data)
                sig = replace(sig, stop_loss=sl)
                
                return sid, sig, metrics, thresholds
                    
            return sid, sig, metrics, thresholds

        if active_runtimes:
            with concurrent.futures.ThreadPoolExecutor(max_workers=min(4, len(active_runtimes))) as executor:
                futures = {executor.submit(_execute_runtime, args): args[0] for args in active_runtimes}
                for future in concurrent.futures.as_completed(futures):
                    try:
                        sid, sig, metrics, thresholds = future.result()
                        pulse_report["strategies"][sid] = {
                            "signal": sig if sig else "NONE",
                            "metrics": metrics,
                            "thresholds": thresholds,
                            "fidelity": sig.confidence if hasattr(sig, "confidence") else 1.0
                        }
                        if sig and sig.direction != "NONE":
                            raw_signals[sid] = sig
                    except Exception as e:
                        logger.error(f"Execution failed for strategy {futures[future].strategy_id}: {e}")

        if not raw_signals:
            return pulse_report

        # 5. RISK ENGINE VALIDATES TRADE (Institutional Pillar 4: Zero Latency Ledger)
        current_balance = account_snapshot.get('balance', 1000.0)
        current_equity = account_snapshot.get('equity', current_balance)
        
        risk_guardian = self.runtimes[0].risk_guardian if self.runtimes else None
        validated_signals = {}
        
        if risk_guardian:
            # 5.1 Global circuit breaker (Exposure Netting 2.0)
            # Gather live positions for exposure counting in a list of dicts format
            all_open_pos = self.position_manager.get_open_positions() if hasattr(self.position_manager, 'get_open_positions') else []
            pos_dicts = []
            for p in all_open_pos:
                pos_dicts.append({
                    "symbol": p.symbol,
                    "volume": p.volume,
                    "type": p.type,
                    "profit": p.profit
                })
            
            allowed, reason = risk_guardian.check_governance(
                current_balance, 
                current_equity, 
                positions=pos_dicts
            )
            if not allowed:
                logger.warning(f"Flow HALTED: {reason}")
                return pulse_report
            
            # 5.2 Individual signal vetting & HARD CONSTRAINTS (Step 13)
            for sid, sig in raw_signals.items():
                runtime = next((r for r in self.runtimes if r.strategy_id == sid), None)
                if not runtime: continue
                
                strat_risk_guardian = runtime.risk_guardian
                
                # S-ID MAGIC Logic (Rule 2.2 Hardening)
                # Derived from sid to ensure persistence across sessions
                magic = strat_risk_guardian.get_magic_number(sid)
                
                if hasattr(self.position_manager, 'get_positions_by_magic'):
                    open_pos = self.position_manager.get_positions_by_magic(magic, symbol)
                elif hasattr(self.position_manager, 'get_positions_by_strategy'):
                    open_pos = self.position_manager.get_positions_by_strategy(sid, symbol)
                else:
                    open_pos = []

                if len(open_pos) >= 1:
                    logger.warning(f"[RISK] Rejecting {symbol}: Strategy {sid} already has active exposure (Magic: {magic})")
                    continue
                
                # 5.3 Circuit Breaker Check (Step 24)
                strat_allowed, strat_reason = strat_risk_guardian.check_strategy_governance(sid)
                if not strat_allowed:
                    logger.warning(f"[CIRCUIT BREAKER] Strategy {sid} rejected: {strat_reason}")
                    continue

                # Sizing & Margin Validation
                if strat_risk_guardian.validate_signal(sig, current_balance, market_data, symbol_info):
                    validated_signals[sid] = sig
                else:
                    logger.debug(f"[{sid}] Signal REJECTED at Risk Validation")

        if not validated_signals:
            return pulse_report

        # 6. PORTFOLIO MANAGER AUDITS SIGNALS
        # In V5-INSIGNIA Parallel mode, we approve all non-conflicting edges.
        approved_signals = self.portfolio_manager.resolve_signals(validated_signals)
        if not approved_signals:
            return pulse_report
            
        # 7. EXECUTION ENGINE FLOW (Iterate over all approved strategies)
        for sid, sig in approved_signals:
            runtime = next((r for r in self.runtimes if r.strategy_id == sid), None)
            if not runtime:
                continue
                
            # Final TP Calculation (using replace for frozen dataclass)
            tp = runtime.strategy.get_take_profit(sig, market_data)
            sig = replace(sig, take_profit=tp)

            # ── INSTITUTIONAL LOT SIZING ─────────────────────────────────────
            # Compute risk-based lot from the strategy's own risk guardian.
            # This correctly uses risk_per_trade_pct against the current balance
            # and stop-loss distance — never overrides with a fixed value.
            sl_dist = abs(sig.price - sig.stop_loss) if sig.stop_loss and sig.price else 0.0
            if sl_dist <= 0:
                logger.warning(f"[{sid}] SL distance is zero — cannot size lot. Skipping.")
                continue

            risk_based_lot = runtime.risk_guardian.calculate_lot_size(
                balance=current_balance,
                stop_loss_dist=sl_dist,
                symbol_info=symbol_info,
                current_price=market_data.current_price,
                volatility_status=pulse_regime_info.volatility
            )

            # Institutional Integrity: All risk scaling and safety caps are now
            # managed inside RiskGuardian to prevent non-linear compounding risk.
            final_lot = risk_based_lot
            
            sym_min_lot = symbol_info.get("min_lot", 0.01) if symbol_info else 0.01
            final_lot = max(final_lot, sym_min_lot)  # Enforce broker minimum

            if final_lot < sym_min_lot:
                logger.warning(f"[{sid}] Risk-based lot {risk_based_lot:.4f} below min {sym_min_lot}. Skipping.")
                continue

            sig = replace(sig, volume=final_lot)
            
            if sig.volume > 0:
                # 8. UNIFIED EXECUTION BRIDGE (Audit Bug #7 Fix)
                magic = runtime.risk_guardian.get_magic_number(sid)
                
                # Construct price_data for OrderManager simulation fallback
                price_data = {
                    "bid": market_data.current_price,
                    "ask": market_data.current_price + (symbol_info.get("spread", 0) * symbol_info.get("point", 0.0001)),
                    "point": symbol_info.get("point", 0.0001)
                }
                
                execution_result = self.order_manager.execute_signal(
                    signal=sig,
                    symbol=symbol,
                    price_data=price_data,
                    is_news_blocked=False, # Already checked at start of cycle
                    magic=magic,
                    comment=f"V5-PBO"
                )
                
                # 9. PERFORMANCE TRACKER LOGS RESULT
                if execution_result and not execution_result.get("is_error", False):
                    # Strategy-specific feedback
                    runtime.risk_guardian.check_governance(
                        current_balance, current_equity
                    )
                    pulse_report["execution"].append(execution_result)

                    # Store Ticket for Awareness (Audit Bug #2 Fix)
                    ticket = execution_result.get("ticket")
                    if ticket:
                        with self._tickets_lock:
                            self._open_tickets.add(ticket)
                            # Register original SL for accurate initial-risk tracking in partials
                            self._original_sl[ticket] = sig.stop_loss
                            logger.info(f"[{sid}] Ticket {ticket} added to Orchestrator Registry (orig_sl={sig.stop_loss:.5f}).")
            
        # 6. POST-CYCLE MANAGEMENT (Trailing Stops & Partials)
        # Institutional Hardening: Synchronous execution ensures zero race conditions.
        regime_info = pulse_report.get("regime")
        if regime_info:
            self.manage_trailing_stops(
                symbol, 
                market_data.current_price, 
                market_data.current_price + (symbol_info.get("spread", 0) * symbol_info.get("point", 0.0001)), 
                regime_info.atr, 
                None, 
                market_data.session
            )
            self.manage_partials(
                symbol, 
                market_data.current_price, 
                market_data.current_price + (symbol_info.get("spread", 0) * symbol_info.get("point", 0.0001)),
                market_data.session
            )
            
        return pulse_report

    def _sync_ticket_registry(self):
        """Synchronizes internal ticket registry with live MT5 state."""
        try:
            live_pos = self.position_manager.get_open_positions()
            live_tickets = {p.ticket for p in live_pos}
            with self._tickets_lock:
                closed_tickets = self._open_tickets - live_tickets
                self._open_tickets = self._open_tickets.intersection(live_tickets)
                for t in closed_tickets:
                    self._original_sl.pop(t, None)
        except Exception as e:
            logger.debug(f"Ticket Sync failed: {e}")

    def on_trade_closed(self, trade_record: dict):
        """Propagates trade closure to the relevant strategy runtime."""
        sid = trade_record.get("strategy_id")
        ticket = trade_record.get("ticket")
        
        # 1. Cleanup Risk Registry (Phase 1 Stability)
        if ticket:
            with self._tickets_lock:
                removed_sl = self._original_sl.pop(ticket, None)
                if removed_sl:
                    logger.debug(f"[REGISTRY] Cleared orig_sl for {ticket} on closure.")
                if ticket in self._scaled_tickets:
                    self._scaled_tickets.remove(ticket)

        # 2. Propagate to strategies
        with self._analysis_lock:
            runtimes_snapshot = list(self.runtimes)
            
        for runtime in runtimes_snapshot:
            if runtime.strategy_id == sid:
                # Record strategy-specific performance for Circuit Breakers
                pnl = trade_record.get("net_pnl", 0.0)
                strat_bal = self.portfolio_manager.get_strategy_balance(self.connection.get_balance(), sid)
                runtime.risk_guardian.record_strategy_result(sid, pnl, strat_bal)
                
                runtime.on_trade_closed(trade_record)
                break

    def manage_trailing_stops(self, symbol: str, bid: float, ask: float, atr: float, last_candle: Any, session: str):
        """
        Institutional Trailing Stop Logic.
        Moves SL based on R:R thresholds and ATR-based trailing.
        """
        conf = self.resolve_block("trailing_stop", session, symbol)
        if not conf.get("enabled", False):
            return

        rr_threshold = conf.get("phase1_rr_threshold", 1.5)
        
        # Delegate to PositionManager for MT5 interaction
        positions = self.position_manager.get_open_positions(symbol)
        
        for pos in positions:
            # Calculate current R:R
            entry = pos.price_open
            current_sl = pos.sl
            direction = "BUY" if pos.type == 0 else "SELL"
            current_price = bid if direction == "BUY" else ask
            
            # Initial Risk
            initial_risk = abs(entry - current_sl)
            if initial_risk == 0: continue
            
            # Current Profit in Points
            profit_points = (current_price - entry) if direction == "BUY" else (entry - current_price)
            current_rr = profit_points / initial_risk
            
            new_sl = None
            
            # Phase 1: Move to Break-Even (at 1.5R)
            if current_rr >= rr_threshold and abs(current_sl - entry) > (initial_risk * 0.1):
                be_offset = initial_risk * conf.get("phase2_be_offset_pct", 0.1)
                new_sl = entry + be_offset if direction == "BUY" else entry - be_offset
                logger.info(f"Trailing: Moving {pos.ticket} to Break-Even.")

            # Phase 2: ATR-based Trailing (at 3R+)
            if current_rr >= 3.0:
                trail_sl = current_price - (atr * conf.get("phase3_trail_mult", 1.5)) if direction == "BUY" else current_price + (atr * conf.get("phase3_trail_mult", 1.5))
                # Only move SL if it improves protection
                if direction == "BUY" and trail_sl > current_sl:
                    new_sl = trail_sl
                elif direction == "SELL" and (current_sl == 0 or trail_sl < current_sl):
                    new_sl = trail_sl
            
            if new_sl:
                self.connection.modify_sl_tp(pos.ticket, symbol, new_sl, pos.tp)

    def manage_partials(self, symbol: str, bid: float, ask: float, session: str):
        """
        Handles partial profit taking scaling (Institutional Grade).
        Closes a percentage of volume when the initial R:R target is hit.
        """
        conf = self.resolve_block("partial_profit", session, symbol)
        if not conf.get("enabled", False):
            return

        rr_target = conf.get("phase1_rr_target", 1.5)
        close_pct = conf.get("phase1_close_pct", 50)
        
        positions = self.position_manager.get_open_positions(symbol)
        
        for pos in positions:
            # Skip if already scaled
            with self._tickets_lock:
                if pos.ticket in self._scaled_tickets:
                    continue

            # Calculate current R:R
            entry = pos.price_open
            direction = "BUY" if pos.type == 0 else "SELL"
            current_price = bid if direction == "BUY" else ask

            # Initial Risk: prefer tracked original SL over live (possibly moved) SL.
            # This prevents incorrect R:R after a break-even move.
            with self._tickets_lock:
                orig_sl = self._original_sl.get(pos.ticket, pos.sl)
            initial_risk = abs(entry - orig_sl)
            if initial_risk == 0: continue
            
            # Current Profit in Points
            profit_points = (current_price - entry) if direction == "BUY" else (entry - current_price)
            current_rr = profit_points / initial_risk if initial_risk > 0 else 0
            
            if current_rr >= rr_target:
                close_vol = pos.volume * (close_pct / 100.0)
                # Institutional Floor: Ensure we don't scale below min lot
                sym_info = self.connection.get_symbol_info(symbol)
                min_lot = sym_info.get("min_lot", 0.01)
                
                if close_vol >= min_lot and (pos.volume - close_vol) >= min_lot:
                    logger.info(f"[PARTIAL] Target hit for {pos.ticket} ({symbol}) at {current_rr:.2f}R. Scaling out {close_vol} lots.")
                    success = self.connection.close_position_partial(pos.ticket, close_vol)
                    
                    if success:
                        with self._tickets_lock:
                            self._scaled_tickets.add(pos.ticket)
                        
                        # Move to Break-Even if configured
                        if conf.get("move_to_be_at_partial", True):
                            logger.info(f"[PARTIAL] Moving {pos.ticket} to Break-Even after scale-out.")
                            be_sl = entry + (initial_risk * 0.1) if direction == "BUY" else entry - (initial_risk * 0.1)
                            self.connection.modify_sl_tp(pos.ticket, symbol, be_sl, pos.tp)

    def reset_daily(self, new_balance: float):
        """Propagates daily reset and balance sync to all strategy runtimes."""
        for runtime in self.runtimes:
            runtime.reset_daily(new_balance)

    def close_before_news(self, current_time: float):
        """
        Institutional Risk Reduction.
        Flattens positions for currencies with upcoming high-impact news.
        """
        targets = self.news_filter.get_auto_close_targets(current_time)
        if not targets:
            return
            
        logger.info(f"News: Proactive Risk Reduction for currencies: {targets}")
        
        # Flatten positions for affected currencies
        all_positions = self.position_manager.get_open_positions()
        for pos in all_positions:
            symbol = pos.symbol
            # Check if any currency in the symbol is in targets
            is_affected = any(curr in symbol for curr in targets)
            if is_affected:
                logger.warning(f"[NEWS CLOSE] Flattening {symbol} (Ticket: {pos.ticket}) before high-impact event.")
                self.connection.close_position(pos.ticket, symbol)
                if self.notification_manager:
                    self.notification_manager.send_alert(f"NEWS BLOCK: Proactive closure of {symbol} @ {pos.price_open}")

    def flatten_all_positions(self, symbol: str = None):
        """
        Emergency Liquidation.
        Closes all open positions immediately (globally if no symbol specified).
        """
        target = symbol if symbol else "ALL SYMBOLS"
        logger.critical(f"EMERGENCY FLATTEN: Closing all positions for {target}")
        all_positions = self.position_manager.get_open_positions(symbol)
        for pos in all_positions:
            success = self.connection.close_position(pos.ticket, pos.symbol)
            if success:
                logger.info(f"Emergency closed position {pos.ticket} ({pos.symbol})")
            else:
                logger.error(f"Failed to emergency close position {pos.ticket} ({pos.symbol})")
        
    def __repr__(self):
        return f"<StrategyOrchestrator(runtimes={len(self.runtimes)})>"
