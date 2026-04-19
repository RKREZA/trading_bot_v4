import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone
from dataclasses import replace
import threading

from core.strategy_runtime import StrategyRuntime
from core.portfolio_manager import PortfolioManager
from core.base_strategy import MarketData
from core.common.types import TradeSignal
from core.execution.order_manager import OrderManager
from core.ai.predictor import AIPredictor

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
                 news_filter):
        self.runtimes = runtimes
        self.config = config
        self.order_manager = order_manager
        self.connection = order_manager.connection # Underlying connection
        self.position_manager = position_manager
        self.notification_manager = notification_manager
        self.broker_clock = broker_clock
        self.news_filter = news_filter
        
        self.portfolio_manager = PortfolioManager(self.config)
        self.ai_predictor = AIPredictor(self.config)
        self.last_cycle_time: Optional[datetime] = None
        self.last_analysis: Dict[str, Any] = {}
        self._analysis_lock = threading.Lock() 

        self._open_tickets = set()
        self._scaled_tickets = set() # Track tickets that have already taken partial profits
        self._tickets_lock = threading.Lock()

        # Asynchronous Trailing Stop Management (Rule 3.1)
        self._stop_thread = threading.Thread(target=self._trailing_stop_loop, daemon=True)
        self._stop_thread.start()

    def execute_cycle(self, symbol: str, market_data: MarketData, account_snapshot: dict, is_news_blocked: bool = False) -> Dict[str, Any]:
        """
        STRICT 8-STEP INSTITUTIONAL EXECUTION FLOW (Step 12).
        ====================================================
        Returns a 'Pulse Report' for dashboard telemetry.
        """
        exec_payloads = []
        symbol_info = self.connection.get_symbol_info(symbol) if hasattr(self.connection, 'get_symbol_info') else {}

        # 1. LOAD: Handled via market_data arrival
        
        # 2. DETECT REGIME
        from core.regime_detector import RegimeDetector
        regime_info = RegimeDetector().detect(market_data.m5_candles)
        market_type = regime_info.market_type
        volatility = regime_info.volatility
        
        ts = market_data.timestamp.timestamp()
        blocking_event = self.news_filter.is_blocked(symbol, ts)
        
        # Pulse Telemetry Initialization
        upcoming_obj = self.news_filter.get_upcoming_events(ts, 24)
        for ev in upcoming_obj:
            # Shift from UTC to Local System Time for Unified UI alignment
            ev['time'] = datetime.fromtimestamp(ev['timestamp'], tz=timezone.utc).astimezone().strftime("%I:%M %p")
            
        pulse_report = {
            "regime": regime_info,
            "strategies": {},
            "execution": [],
            "news_blocked": blocking_event,
            "upcoming_news": [e["title"] for e in upcoming_obj],
            "upcoming_news_obj": upcoming_obj,
            "timestamp": market_data.timestamp.astimezone().strftime("%H:%M:%S"),
            "open_tickets": list(self._open_tickets) # Dashboard visibility
        }
        
        # Update cache for async services (Rule 3.1) - Thread Safe
        with self._analysis_lock:
            self.last_analysis[symbol] = {
                "bid": market_data.current_price, # Simplified, should use tick if available
                "ask": market_data.current_price + (symbol_info.get("spread", 0) * symbol_info.get("point", 0)),
                "atr": regime_info.atr,
                "timestamp": ts
            }

        # Hard Block Check
        if blocking_event or is_news_blocked:
            logger.info(f"Cycle Skip: {symbol} is blocked by news ({blocking_event})")
            return pulse_report
        
        # 2.5 Fetch upcoming news for AI Vetting
        news_events = self.news_filter.get_upcoming_events(market_data.timestamp.timestamp(), window_hours=24)
        
        # 3. ACTIVATE RELEVANT STRATEGIES (Regime Gating)
        from core.regime_gater import RegimeGater
        active_runtimes = []
        for r in self.runtimes:
            if not r.strategy.is_symbol_allowed(symbol):
                continue
            
            # Diagnostic: Session Gating
            allowed_sessions = r.strategy.config.get("allowed_sessions", [])
            if allowed_sessions and market_data.session not in allowed_sessions:
                if self.config.get("backtest", {}).get("debug_signals"):
                    logger.debug(f"[{symbol}] [{r.strategy_id}] Skipped: Session {market_data.session} not in {allowed_sessions}")
                continue

            if RegimeGater.is_strategy_allowed(r.strategy.__class__.__name__, market_type):
                active_runtimes.append(r)
        
        # 4. GENERATE SIGNALS (Institutional Parallel Execution)
        raw_signals = {}
        import concurrent.futures
        
        def _execute_runtime(runtime):
            sid = runtime.strategy_id
            sig = runtime.execute_cycle(market_data)
            metrics = runtime.strategy.get_metrics(market_data)
            thresholds = runtime.strategy.get_thresholds()
            
            if sig and sig.direction != "NONE":
                sl = runtime.strategy.get_stop_loss(sig, market_data)
                sig = replace(sig, stop_loss=sl)
                
                # ML LAYER: Immutable DTO Priority 1 Evaluation (Statistical + Macro Reasoning)
                # Pass m5_candles instead of market_data (FeatureEngineer expects CandleArray)
                candles_for_ai = market_data.m5_candles if market_data.m5_candles is not None else market_data.htf_candles
                filtered_sig = self.ai_predictor.filter_signal(
                    sig, 
                    candles_for_ai, 
                    abs(sig.price - sl) if hasattr(sig, 'price') else sl,
                    news_events=news_events
                )
                
                if filtered_sig.approved:
                    return sid, sig, metrics, thresholds
                else:
                    return sid, None, metrics, thresholds
                    
            return sid, sig, metrics, thresholds

        if active_runtimes:
            with concurrent.futures.ThreadPoolExecutor(max_workers=min(4, len(active_runtimes))) as executor:
                futures = {executor.submit(_execute_runtime, r): r for r in active_runtimes}
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
            
            # Fixed lot size override (0.05 = user requested max)
            fixed_lot = 0.05
            sym_min_lot = symbol_info.get("min_lot", 0.01) if symbol_info else 0.01
            if fixed_lot < sym_min_lot:
                fixed_lot = sym_min_lot
            
            sig = replace(sig, volume=fixed_lot)
            
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
                            logger.info(f"[{sid}] Ticket {ticket} added to Orchestrator Registry.")
            
        return pulse_report

    def on_trade_closed(self, trade_record: dict):
        """Propagates trade closure to the relevant strategy runtime."""
        sid = trade_record.get("strategy_id")
        for runtime in self.runtimes:
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
        if not self.config.get("trailing_stop", {}).get("enabled", False):
            return

        conf = self.config["trailing_stop"]
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

    def manage_partials(self, symbol: str, bid: float, ask: float):
        """
        Handles partial profit taking scaling (Institutional Grade).
        Closes a percentage of volume when the initial R:R target is hit.
        """
        if not self.config.get("partial_profit", {}).get("enabled", False):
            return

        conf = self.config["partial_profit"]
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
            current_sl = pos.sl
            direction = "BUY" if pos.type == 0 else "SELL"
            current_price = bid if direction == "BUY" else ask
            
            # Initial Risk (distance from entry to original SL)
            # Note: We use pos.sl which might have been moved. 
            # Ideally we want original_sl. For now, we estimate from entry.
            initial_risk = abs(entry - current_sl)
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
        
    def _trailing_stop_loop(self):
        """Background thread for high-frequency stop management."""
        import time
        logger.info("Trailing Stop Service started.")
        while True:
            try:
                # 1. Synchronize Registry with Live Positions
                try:
                    live_pos = self.position_manager.get_open_positions()
                    live_tickets = {p.ticket for p in live_pos}
                    with self._tickets_lock:
                        # Prune tickets no longer in MT5
                        self._open_tickets = self._open_tickets.intersection(live_tickets)
                except Exception as e:
                    logger.debug(f"Ticket Sync failed: {e}")

                # Use a snapshot of the analysis dictionary to avoid ConcurrentModificationException
                with self._analysis_lock:
                    analysis_snapshot = dict(self.last_analysis)

                if not analysis_snapshot:
                    time.sleep(0.5)
                    continue

                for symbol, data in analysis_snapshot.items():
                    # 1. Trailing Stop Management
                    self.manage_trailing_stops(
                        symbol, 
                        data['bid'], 
                        data['ask'], 
                        data['atr'], 
                        None, 
                        "GLOBAL"
                    )
                    
                    # 2. Partials Management
                    self.manage_partials(symbol, data['bid'], data['ask'])

                time.sleep(0.2) # 5Hz update rate
            except Exception as e:
                logger.error(f"Trailing Stop Thread Error: {e}")
                time.sleep(1)

    def __repr__(self):
        return f"<StrategyOrchestrator(runtimes={len(self.runtimes)})>"
