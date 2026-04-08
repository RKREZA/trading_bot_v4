import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

from core.strategy_runtime import StrategyRuntime
from core.portfolio_manager import PortfolioManager
from core.base_strategy import MarketData
from core.common.types import TradeSignal

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
                 connection,
                 position_manager, 
                 notification_manager,
                 broker_clock):
        self.runtimes = runtimes
        self.config = config
        self.connection = connection
        self.position_manager = position_manager
        self.notification_manager = notification_manager
        self.broker_clock = broker_clock
        
        from core.news_filter import InstitutionalNewsFilter
        self.news_filter = InstitutionalNewsFilter(self.config)
        
        self.portfolio_manager = PortfolioManager(self.config)
        self.last_cycle_time: Optional[datetime] = None
        self.last_analysis: Dict[str, Any] = {}

        # Asynchronous Trailing Stop Management (Rule 3.1)
        import threading
        self._stop_thread = threading.Thread(target=self._trailing_stop_loop, daemon=True)
        self._stop_thread.start()

    def execute_cycle(self, symbol: str, market_data: MarketData, is_news_blocked: bool = False) -> Dict[str, Any]:
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
        pulse_report = {
            "regime": regime_info,
            "strategies": {},
            "execution": [],
            "news_blocked": blocking_event,
            "upcoming_news": [e["title"] for e in self.news_filter.get_upcoming_events(ts, 4)],
            "timestamp": market_data.timestamp.strftime("%H:%M:%S")
        }
        
        # Update cache for async services (Rule 3.1)
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
        
        # 3. ACTIVATE RELEVANT STRATEGIES (Regime Gating)
        from core.regime_gater import RegimeGater
        active_runtimes = [
            r for r in self.runtimes 
            if r.strategy.is_symbol_allowed(symbol) and RegimeGater.is_strategy_allowed(r.strategy.__class__.__name__, market_type)
        ]
        
        # 4. GENERATE SIGNALS (Independent Generation)
        raw_signals = {}
        for runtime in active_runtimes:
            sid = runtime.strategy_id
            sig = runtime.strategy.generate_signal(market_data)
            
            # Telemetry: Record Full Signal (including Reasons)
            pulse_report["strategies"][sid] = sig if sig else "NONE"
            
            if sig and sig.direction != "NONE":
                # Attach SL for risk validation
                sig.stop_loss = runtime.strategy.get_stop_loss(sig, market_data)
                raw_signals[sid] = sig

        if not raw_signals:
            return pulse_report

        # 5. RISK ENGINE VALIDATES TRADE
        current_balance = self.connection.get_balance() if hasattr(self.connection, 'get_balance') else 1000.0
        current_equity = self.connection.get_equity() if hasattr(self.connection, 'get_equity') else current_balance
        
        risk_guardian = self.runtimes[0].risk_guardian if self.runtimes else None
        validated_signals = {}
        
        if risk_guardian:
            # 5.1 Global circuit breaker
            allowed, reason = risk_guardian.check_governance(current_balance, current_equity)
            if not allowed:
                logger.warning(f"Flow HALTED: {reason}")
                return pulse_report
            
            # 5.2 Individual signal vetting & HARD CONSTRAINTS (Step 13)
            for sid, sig in raw_signals.items():
                # S-ID MAGIC Logic (Rule 2.2 Hardening)
                # Derived from sid to ensure persistence across sessions
                magic = self.runtimes[0].risk_guardian.get_magic_number(sid) if self.runtimes else 234000
                
                if hasattr(self.position_manager, 'get_positions_by_magic'):
                    open_pos = self.position_manager.get_positions_by_magic(magic, symbol)
                elif hasattr(self.position_manager, 'get_positions_by_strategy'):
                    open_pos = self.position_manager.get_positions_by_strategy(sid, symbol)
                else:
                    open_pos = []

                if len(open_pos) >= 1:
                    logger.warning(f"[RISK] Rejecting {symbol}: Strategy {sid} already has active exposure (Magic: {magic})")
                    continue
                
                # Sizing & Margin Validation
                if risk_guardian.validate_signal(sig, current_balance, market_data, symbol_info):
                    validated_signals[sid] = sig
                else:
                    logger.debug(f"[{sid}] Signal REJECTED at Risk Validation")

        if not validated_signals:
            return pulse_report

        # 6. PORTFOLIO MANAGER AUDITS SIGNALS
        # In V4-ULTRA Parallel mode, we approve all non-conflicting edges.
        approved_signals = self.portfolio_manager.resolve_signals(validated_signals)
        if not approved_signals:
            return pulse_report
            
        # 7. EXECUTION ENGINE FLOW (Iterate over all approved strategies)
        for sid, sig in approved_signals:
            runtime = next((r for r in self.runtimes if r.strategy_id == sid), None)
            if not runtime:
                continue
                
            # Final TP Calculation
            sig.take_profit = runtime.strategy.get_take_profit(sig, market_data)
            
            # Final Lot Calculation (Partitioned Strategy Balance)
            strat_balance = self.portfolio_manager.get_strategy_balance(current_balance, sid)
            sl_dist = abs(market_data.current_price - sig.stop_loss)
            sig.volume = risk_guardian.calculate_lot_size(strat_balance, sl_dist, symbol_info)
            
            if sig.volume > 0:
                # 8. LIVE EXECUTION BRIDGE (with Strategy-Specific Magic)
                magic = runtime.risk_guardian.get_magic_number(sid)
                execution_result = self.connection.place_order(
                    symbol, 
                    sig, 
                    sig.volume,
                    comment=f"V4 {sid.upper()} PARALLEL",
                    magic=magic
                )
                
                # 9. PERFORMANCE TRACKER LOGS RESULT
                if execution_result:
                    # Strategy-specific feedback
                    runtime.risk_guardian.check_governance(
                        current_balance, current_equity, 
                        slippage=execution_result.get("actual_slippage_pips", 0),
                        is_error=execution_result.get("is_error", False)
                    )
                    pulse_report["execution"].append(execution_result)
            
        return pulse_report

    def on_trade_closed(self, trade_record: dict):
        """Propagates trade closure to the relevant strategy runtime."""
        sid = trade_record.get("strategy_id")
        for runtime in self.runtimes:
            if runtime.strategy_id == sid:
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
        """Handles partial profit taking targets."""
        # Simple implementation: Close 50% at 2:1 RR if configured
        pass

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

    def _trailing_stop_loop(self):
        """Background thread for high-frequency stop management."""
        import time
        logger.info("Trailing Stop Service started.")
        while True:
            try:
                # We need Bid/Ask/ATR per symbol. 
                # This threaded version requires a way to get latest state.
                # For now, we use the last_analysis state if available or wait for next tick data.
                if not self.last_analysis:
                    time.sleep(0.5)
                    continue

                for symbol, data in self.last_analysis.items():
                    # We only manage symbols that have open positions
                    self.manage_trailing_stops(
                        symbol, 
                        data['bid'], 
                        data['ask'], 
                        data['atr'], 
                        None, 
                        "GLOBAL"
                    )
                time.sleep(0.2) # 5Hz update rate
            except Exception as e:
                logger.error(f"Trailing Stop Thread Error: {e}")
                time.sleep(1)

    def __repr__(self):
        return f"<StrategyOrchestrator(runtimes={len(self.runtimes)})>"
