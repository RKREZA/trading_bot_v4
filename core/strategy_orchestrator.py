"""
TRADING BOT V3 — Strategy Orchestrator
Decoupled multi-strategy dispatcher that feeds the same market data
to all strategy runtimes and collects signals independently.

Design:
    - Market data is shared (read-only, frozen)
    - Strategy state is fully isolated (per-runtime)
    - Orders are tagged with strategy_id for attribution
    - No cross-strategy signal influence
"""

import logging
import time
import threading
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from .base_strategy import MarketData, TaggedSignal
from .strategy_runtime import StrategyRuntime
from .order_tagger import OrderTagger
from .lot_calculator import LotCalculator
from .strategy_engine import StrategyEngine

try:
    import MetaTrader5 as mt5
except ImportError:
    mt5 = None

logger = logging.getLogger("trading_bot.orchestrator")

# Minimum SL distance in price units to prevent undefined-risk trades
_MIN_SL_DISTANCE_POINTS = 50


class StrategyOrchestrator:
    """
    Central execution engine for the multi-strategy framework.
    
    Responsibilities:
        1. Feed identical market data to all strategy runtimes
        2. Collect signals independently (no cross-contamination)
        3. Apply per-strategy risk checks
        4. Execute orders tagged with strategy_id
        5. Route trade events back to the owning strategy
    
    Extensibility:
        Adding a new strategy requires ZERO changes to this class.
        Simply add a new StrategyRuntime to the runtimes list.
    """

    def __init__(
        self,
        runtimes: List[StrategyRuntime],
        config: dict,
        connection: Any,
        position_manager: Any,
        notification_manager: Any = None,
    ):
        """
        Args:
            runtimes: List of StrategyRuntime instances (one per strategy)
            config: Global bot configuration
            connection: MT5Connection instance
            position_manager: PositionManager instance 
            notification_manager: Optional NotificationManager
        """
        self.runtimes = runtimes
        self.config = config
        self.connection = connection
        self.position_manager = position_manager
        self.notification_manager = notification_manager
        
        # Spread filter state (shared monitoring, not strategy state)
        self.spread_history: List[float] = []
        
        # Preprocessing cache
        self._last_preprocessed_time: int = 0
        self._cached_pre_ctx: Optional[dict] = None
        self._preprocessing_engine = StrategyEngine(config, silent=True)
        
        # Research mode
        self.research_mode = config.get("research_mode", False)
        
        # Last analysis for dashboard
        self.last_analysis: Dict[str, Any] = {}

    def execute_cycle(
        self,
        symbol: str,
        h1: Any, m15: Any, m5: Any, d1: Any,
        current_price: float,
        session: str,
    ) -> List[TaggedSignal]:
        """
        Execute one full cycle: feed data → collect signals → risk check → execute.
        
        Args:
            symbol: Trading symbol
            h1, m15, m5, d1: Multi-timeframe CandleArrays
            current_price: Latest known price
            session: Current trading session
            
        Returns:
            List of TaggedSignals that resulted in executed orders
        """
        executed_signals = []
        
        # 0. Spread Filter (shared read-only check, not strategy state)
        if not self._check_spread(symbol):
            return executed_signals

        start_time = time.time()
        
        # 1. Preprocessing (shared computation, read-only result)
        current_m5_time = int(m5.time[-1]) if len(m5) > 0 else 0
        if current_m5_time != self._last_preprocessed_time:
            self._cached_pre_ctx = self._preprocessing_engine.preprocess_history(h1, m15, m5, m5)
            self._last_preprocessed_time = current_m5_time

        pre_ctx = self._cached_pre_ctx or {}
        latest_meta = pre_ctx.get("m5", [{}])[-1] if pre_ctx.get("m5") else {}

        # Build frozen market data (shared read-only across all strategies)
        market_data = MarketData(
            symbol=symbol,
            htf_candles=h1,
            m15_candles=m15,
            m5_candles=m5,
            d1_candles=d1,
            current_price=current_price,
            session=session,
            timestamp=datetime.now(timezone.utc),
            preprocessed=latest_meta,
        )

        # Update dashboard analysis cache
        self.last_analysis = {
            "trend": latest_meta.get("m_bias", "NEUTRAL"),
            "regime": "MULTI_STRATEGY",
            "bias": latest_meta.get("m_bias", "NEUTRAL"),
            "in_demand": latest_meta.get("in_htf_demand", False),
            "in_supply": latest_meta.get("in_htf_supply", False),
            "vol_sma": latest_meta.get("vol_sma", 0.0),
            "current_vol": m5.tick_volume[-1] if len(m5) > 0 else 0,
        }

        # 2. Account snapshot (shared read-only)
        acc = self.connection.get_account_snapshot()
        current_balance = acc.get("balance", 0.0)
        current_equity = acc.get("equity", 0.0)

        # 3. Per-strategy signal generation and execution
        for runtime in self.runtimes:
            if not runtime.enabled:
                continue

            try:
                tagged = self._process_runtime(
                    runtime, market_data, symbol,
                    current_balance, current_equity, session
                )
                if tagged:
                    executed_signals.append(tagged)
            except Exception as e:
                logger.error(
                    "[%s] Runtime error (recovering): %s",
                    runtime.strategy_id, e, exc_info=True
                )

        latency_ms = (time.time() - start_time) * 1000
        if executed_signals:
            logger.info(
                "Orchestrator cycle: %d signals executed in %.2fms",
                len(executed_signals), latency_ms
            )

        return executed_signals

    def _process_runtime(
        self,
        runtime: StrategyRuntime,
        market_data: MarketData,
        symbol: str,
        balance: float,
        equity: float,
        session: str,
    ) -> Optional[TaggedSignal]:
        """
        Process a single strategy runtime: generate signal → risk check → execute.
        Fully isolated — no cross-strategy state access.
        """
        # Per-strategy circuit breaker check
        allowed, cb_reason = runtime.check_risk(balance, equity, session)
        if not allowed:
            if self.research_mode:
                logger.warning(
                    "[%s] RESEARCH MODE: Bypassing circuit breaker (%s)",
                    runtime.strategy_id, cb_reason
                )
            else:
                return None

        # Generate signal
        tagged = runtime.generate_signal(market_data)
        if tagged is None:
            return None

        # Position limit: 1 open position per strategy per symbol
        if runtime.positions.has_open_position:
            logger.info(
                "[%s] Signal ignored — position already open",
                runtime.strategy_id
            )
            return None

        # Risk scaling
        risk_pct = runtime.calculate_risk_pct(balance, equity, session)
        if risk_pct <= 0.0:
            logger.warning("[%s] Risk scaling returned 0 — halting", runtime.strategy_id)
            return None

        # Execute the order
        return self._execute_tagged_order(tagged, runtime, symbol, balance, risk_pct)

    def _execute_tagged_order(
        self,
        tagged: TaggedSignal,
        runtime: StrategyRuntime,
        symbol: str,
        balance: float,
        risk_pct: float,
    ) -> Optional[TaggedSignal]:
        """
        Execute an order on MT5 with strategy attribution tagging.
        """
        signal = tagged.signal
        
        sym_info = self.connection.get_symbol_info(symbol)
        if not sym_info:
            return None

        # Validate SL distance
        point = sym_info.get("point", 0.01)
        sl_dist = abs(signal.entry_price - signal.stop_loss)
        sl_points = sl_dist / point if point > 0 else 0
        if sl_points < _MIN_SL_DISTANCE_POINTS:
            logger.warning(
                "[%s] SL distance too small (%.1f points). Rejected.",
                runtime.strategy_id, sl_points
            )
            return None

        # Lot sizing
        risk_dollar = balance * (risk_pct / 100.0)
        lot = LotCalculator.calculate(
            risk_amount=risk_dollar,
            sl_distance=sl_dist,
            tick_size=point,
            tick_value=sym_info.get("trade_tick_value", 1.0),
            volume_min=sym_info.get("volume_min", 0.01),
            volume_max=sym_info.get("volume_max", 100.0),
            volume_step=sym_info.get("volume_step", 0.01),
        )

        # Create tagged order comment
        comment = OrderTagger.create_comment(tagged.strategy_id, tagged.trade_id)

        logger.info(
            "[%s] Executing %s %s | Lot: %s | SL: %s | TP: %s",
            runtime.strategy_id, signal.direction, symbol,
            lot, signal.stop_loss, signal.take_profit
        )

        # Place order with strategy-tagged comment
        result = self.connection.place_order(
            symbol=symbol,
            signal=signal,
            lot_size=lot,
            comment=comment,
        )

        if result:
            ticket_id = result["ticket"]
            # Register position in the owning runtime ONLY
            runtime.on_trade_opened(ticket_id, {
                "session": signal.session,
                "best_price": signal.entry_price,
                "partial_closed_count": 0,
                "risk": sl_dist,
                "entry_time": time.time(),
                "trade_id": tagged.trade_id,
                "direction": signal.direction,
            })

            if self.notification_manager:
                self.notification_manager.notify_trade_open(
                    symbol=symbol, direction=signal.direction,
                    entry=signal.entry_price, lot=lot,
                    sl=signal.stop_loss, tp=signal.take_profit,
                )

            return tagged

        return None

    def _check_spread(self, symbol: str) -> bool:
        """Rolling spread filter. Shared read-only market check."""
        if mt5 is None:
            return True

        sym_info = self.connection.get_symbol_info(symbol)
        if not sym_info:
            return False

        try:
            with self.connection.MT5_LOCK:
                tick = mt5.symbol_info_tick(symbol)
        except Exception:
            return True

        if tick:
            current_spread = (tick.ask - tick.bid) / sym_info.get("point", 0.01)
            self.spread_history.append(current_spread)
            if len(self.spread_history) > 20:
                self.spread_history = self.spread_history[-20:]

            if len(self.spread_history) >= 20:
                sma = sum(self.spread_history) / 20
                mult = self.config.get("execution", {}).get("spread_filter_mult", 1.5)
                if not self.research_mode and current_spread > sma * mult:
                    logger.warning(
                        "SPREAD FILTER: Current %.1f > SMA %.1f (x%.1f). Blocked.",
                        current_spread, sma, mult
                    )
                    return False
        return True

    def detect_closed_trades(self, symbol: str) -> None:
        """
        Detect closed positions and route trade events to the owning strategy runtime.
        Uses order comment parsing for strategy attribution.
        """
        if mt5 is None:
            return

        try:
            magic = int(self.config.get("magic_number", 234000))
            with self.connection.MT5_LOCK:
                active = mt5.positions_get()
            live_tickets = {p.ticket for p in active if p.magic == magic} if active else set()

            for runtime in self.runtimes:
                closed = runtime.positions.reconcile(live_tickets)
                for ticket in closed:
                    self._process_closed_trade(ticket, runtime, symbol)

        except Exception as e:
            logger.error("Closed trade detection failed: %s", e)

    def _process_closed_trade(self, ticket: int, runtime: StrategyRuntime, symbol: str) -> None:
        """Process a single closed trade for a specific runtime."""
        try:
            with self.connection.MT5_LOCK:
                deals = mt5.history_deals_get(position=ticket)
            if deals:
                total_pnl = sum(d.profit + d.commission + d.swap for d in deals)
                trade_record = {
                    "ticket": ticket,
                    "pnl": round(total_pnl, 2),
                    "result": "WIN" if total_pnl >= 0 else "LOSS",
                    "strategy_id": runtime.strategy_id,
                }
                runtime.on_trade_closed(ticket, trade_record)

                if self.notification_manager:
                    self.notification_manager.notify_trade_close(
                        symbol=symbol,
                        direction="BUY",  # approximate
                        exit_price=0,
                        pnl=total_pnl,
                        result="WIN" if total_pnl >= 0 else "LOSS",
                    )
                logger.info(
                    "[%s] Trade closed: ticket=%d pnl=$%.2f",
                    runtime.strategy_id, ticket, total_pnl
                )
        except Exception as e:
            logger.error("Error processing closed ticket %d for %s: %s", ticket, runtime.strategy_id, e)

    def manage_trailing_stops(self, symbol: str, bid: float, ask: float, 
                               atr: float, last_candle: dict) -> None:
        """Per-strategy trailing stop management."""
        from .trailing_stop import TrailingStopManager

        if not self.config.get("trailing_stop_enabled", True):
            return

        magic = self.config.get("magic_number", 234000)
        positions = self.connection.get_positions(symbol)
        if not positions:
            return

        for pos in positions:
            if pos.magic != magic:
                continue

            ticket = pos.ticket
            is_buy = (pos.type == 0)
            current_price = bid if is_buy else ask

            # Find which runtime owns this ticket
            for runtime in self.runtimes:
                meta = runtime.positions.get_position(ticket)
                if meta is None:
                    continue

                # Update best price
                if (is_buy and current_price > meta.get("best_price", 0)) or \
                   (not is_buy and current_price < meta.get("best_price", float('inf'))):
                    runtime.positions.update_position(ticket, {"best_price": current_price})
                    meta["best_price"] = current_price

                risk = meta.get("risk", abs(pos.price_open - pos.sl) if pos.sl > 0 else 0)

                new_sl = TrailingStopManager.calculate_new_sl(
                    is_buy, pos.price_open, pos.sl, meta["best_price"],
                    atr, risk, self.config, last_candle
                )
                if new_sl and new_sl != pos.sl:
                    if (is_buy and new_sl > pos.sl) or (not is_buy and (pos.sl == 0 or new_sl < pos.sl)):
                        if self.connection.modify_sl_tp(ticket, symbol, new_sl, pos.tp):
                            logger.info(
                                "[%s] Trailing: ticket=%d → SL=%.2f",
                                runtime.strategy_id, ticket, new_sl
                            )
                break  # Ticket found, no need to check other runtimes

    def get_all_positions(self) -> Dict[str, list]:
        """Get positions grouped by strategy_id."""
        result = {}
        for runtime in self.runtimes:
            positions = runtime.positions.get_all_positions()
            result[runtime.strategy_id] = list(positions.values())
        return result

    def get_performance_summary(self) -> Dict[str, dict]:
        """Get performance summary for all strategies."""
        return {
            rt.strategy_id: rt.performance.get_summary()
            for rt in self.runtimes
        }

    def reset_daily(self, balance: float) -> None:
        """Reset daily stats for all runtimes."""
        for runtime in self.runtimes:
            runtime.reset_daily(balance)

    def get_states(self) -> Dict[str, dict]:
        """Serialize all runtime states for persistence."""
        return {
            rt.strategy_id: rt.get_state()
            for rt in self.runtimes
        }

    def load_states(self, states: Dict[str, dict]) -> None:
        """Restore runtime states from persistence."""
        for runtime in self.runtimes:
            state = states.get(runtime.strategy_id)
            if state:
                runtime.load_state(state)
