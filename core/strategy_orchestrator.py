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
        
        self.portfolio_manager = PortfolioManager(self.config)
        self.last_cycle_time: Optional[datetime] = None
        self.last_analysis: Dict[str, Any] = {}

    def execute_cycle(self, symbol: str, market_data: MarketData, is_news_blocked: bool = False) -> List[Dict[str, Any]]:
        """
        STRICT 8-STEP INSTITUTIONAL EXECUTION FLOW (Step 12).
        ====================================================
        """
        exec_payloads = []
        symbol_info = self.connection.get_symbol_info(symbol) if hasattr(self.connection, 'get_symbol_info') else {}

        # 1. LOAD: Handled via market_data arrival
        
        # 2. DETECT REGIME
        from core.regime_detector import RegimeDetector
        regime_info = RegimeDetector().detect(market_data.m5_candles)
        regime = regime_info.type
        
        # 3. ACTIVATE RELEVANT STRATEGIES (Regime Gating)
        from core.regime_gater import RegimeGater
        active_runtimes = [
            r for r in self.runtimes 
            if r.is_symbol_allowed(symbol) and RegimeGater.is_strategy_allowed(r.strategy_id, regime)
        ]
        
        # 4. GENERATE SIGNALS (Independent Generation)
        raw_signals = {}
        for runtime in active_runtimes:
            sig = runtime.strategy.generate_signal(market_data)
            if sig and sig.direction != "NONE":
                # Attach SL for risk validation
                sig.stop_loss = runtime.strategy.get_stop_loss(sig, market_data)
                raw_signals[runtime.strategy_id] = sig

        if not raw_signals:
            return []

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
                return []
            
            # 5.2 Individual signal vetting & HARD CONSTRAINTS (Step 13)
            for sid, sig in raw_signals.items():
                # NO-GRID Check (Rule 13.2)
                open_pos = self.position_manager.get_positions_by_strategy(sid, symbol)
                if len(open_pos) >= 1:
                    logger.debug(f"[{sid}] Signal REJECTED: Anti-Grid Constraint (Already has position)")
                    continue
                
                # Sizing & Margin Validation
                if risk_guardian.validate_signal(sig, current_balance, market_data, symbol_info):
                    validated_signals[sid] = sig
                else:
                    logger.debug(f"[{sid}] Signal REJECTED at Risk Validation")

        if not validated_signals:
            return []

        # 6. PORTFOLIO MANAGER RESOLVES CONFLICTS
        resolution = self.portfolio_manager.resolve_signals(validated_signals)
        if not resolution:
            return []

        winner_sid, winning_sig = resolution
        winner_runtime = next((r for r in self.runtimes if r.strategy_id == winner_sid), None)

        # 7. EXECUTION ENGINE PLACES TRADE
        if winner_runtime:
            winning_sig.take_profit = winner_runtime.strategy.get_take_profit(winning_sig, market_data)
            
            # Final Lot Calculation (Partitioned)
            strat_balance = self.portfolio_manager.get_strategy_balance(current_balance, winner_sid)
            sl_dist = abs(market_data.current_price - winning_sig.stop_loss)
            winning_sig.volume = risk_guardian.calculate_lot_size(strat_balance, sl_dist, symbol_info)
            
            if winning_sig.volume > 0:
                execution_result = winner_runtime.order_manager.execute_signal(
                    winning_sig, symbol, 
                    {'bid': market_data.current_price, 'ask': market_data.current_price, 'point': symbol_info.get('point', 0.00001)},
                    is_news_blocked=is_news_blocked
                )
                
                # 8. PERFORMANCE TRACKER LOGS RESULT (Step 8)
                if execution_result:
                    # Individual strategy feedback
                    winner_runtime.risk_guardian.check_governance(
                        current_balance, current_equity, 
                        slippage=execution_result.get("actual_slippage_pips", 0),
                        is_error=execution_result.get("is_error", False)
                    )
                    exec_payloads.append(execution_result)
            
        return exec_payloads

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

    def __repr__(self):
        return f"<StrategyOrchestrator(runtimes={len(self.runtimes)})>"
