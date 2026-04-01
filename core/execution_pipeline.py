import logging
import threading
from typing import Dict, Any, Optional
from core.strategy_engine import StrategyEngine, TradeSignal
from core.ai_advisor import AIAdvisor
from core.risk_manager import RiskManager
from core.connection import MT5Connection, PositionManager
from core.notifications import NotificationManager
from core.lot_calculator import LotCalculator

logger = logging.getLogger("trading_bot.execution")

try:
    import MetaTrader5 as mt5
except ImportError:
    mt5 = None

class ExecutionPipeline:
    """Orchestrates signal generation, AI vetting, risk checks, and order execution."""
    
    def __init__(self, 
                 config: "BotConfig", 
                 connection: MT5Connection, 
                 position_manager: PositionManager,
                 strategy: StrategyEngine, 
                 ai_advisor: AIAdvisor, 
                 risk_manager: RiskManager,
                 notification_manager: NotificationManager,
                 position_meta: Dict[int, Any],
                 state_lock: threading.Lock):
        self.config = config
        self.connection = connection
        self.position_manager = position_manager
        self.strategy = strategy
        self.ai_advisor = ai_advisor
        self.risk_manager = risk_manager
        self.notification_manager = notification_manager
        self.position_meta = position_meta
        self.state_lock = state_lock
        self.spread_history = []
        
    def execute_cycle(self, symbol: str, m30: "CandleArray", h1: "CandleArray", h4: "CandleArray", m5: "CandleArray", d1: "CandleArray", current_price: float, session: str) -> bool:
        """Runs one full execution cycle for a symbol."""
        # 0. Risk Context & Circuit Breakers
        acc_info = self.connection.get_account_snapshot()
        current_balance = acc_info.get("balance", 0.0)
        current_equity = acc_info.get("equity", 0.0)
        
        # [FIX]: Trigger daily reset for Risk Manager at midnight in LIVE trading
        from datetime import datetime, timezone
        current_date = datetime.now(timezone.utc).date()
        if not hasattr(self, '_last_reset_date') or self._last_reset_date != current_date:
            self.risk_manager.reset_daily_stats(current_balance)
            self._last_reset_date = current_date
        
        daily_trades = self.strategy.daily_trades
        daily_losses = self.strategy.daily_losses
        # Use session-specific consecutive losses if available, otherwise 0
        con_losses = self.strategy.consecutive_losses.get(session, 0)
        
        allowed, cb_reason = self.risk_manager.check_circuit_breakers(
            current_balance=current_balance,
            current_equity=current_equity,
            daily_trades=daily_trades,
            daily_losses=daily_losses,
            consecutive_losses=con_losses
        )
        
        if not allowed:
            logger.warning(f"CIRCUIT BREAKER HALT: {cb_reason}")
            # We still run analyze so it can report trend/regime for dashboard, but it will return None signal
        
        # --- PHASE 11: ROLLING SPREAD FILTER ---
        symbol_info = self.connection.get_symbol_info(symbol)
        if not symbol_info: return False
        
        with self.connection.MT5_LOCK:
            tick = mt5.symbol_info_tick(symbol)
        
        if tick:
            current_spread = (tick.ask - tick.bid) / symbol_info.get("point", 0.01)
            self.spread_history.append(current_spread)
            if len(self.spread_history) > 20:
                self.spread_history = self.spread_history[-20:]
            
            if len(self.spread_history) >= 20:
                spread_sma = sum(self.spread_history) / 20
                if current_spread > spread_sma * 1.5:
                    logger.warning(f"SPREAD FILTER: Aborting trade. Current: {current_spread:.1f} | SMA: {spread_sma:.1f}")
                    return False

        # Start Latency Tracking
        import time
        start_time = time.time()
        
        # 1. Generate Signal
        signal, trend, regime = self.strategy.analyze(
            symbol=symbol,
            m30_candles=m30,
            h1_candles=h1,
            h4_candles=h4,
            m5_candles=m5,
            d1_candles=d1,
            current_price=current_price,
            session=session,
            circuit_breaker_safe=allowed
        )
        
        if not signal:
            return False
            
        # 2. Open Position Check
        if self.position_manager.count_open_positions(symbol) > 0:
            logger.info("Signal ignored — position already open for %s", symbol)
            return False

        # 4. AI Advisory (Veto) Check
        if hasattr(self.ai_advisor, 'enabled') and self.ai_advisor.enabled:
             signal_data_for_ai = {
                 "direction": signal.direction,
                 "reasons": signal.reasons,
                 "confluence": signal.confluence_score,
                 "regime": regime,
                 "trend": trend,
                 "price": current_price,
                 "session": session
             }
             logger.info("Requesting AI Advisory veto check...")
             # Unify backtest/live into filter_signal
             approved, prob, conf = self.ai_advisor.filter_signal_backtest(signal_data_for_ai)
             if not approved:
                 logger.info("Signal VETOED by AI. Confidence: %s", conf)
                 return False
             signal.confidence = conf
             signal.reasons.append("AI APPROVED")

        # 5. Risk Scaling (Passing Equity for SMA Check)
        account = self.connection.get_account_snapshot()
        current_balance = account.get("balance", 0.0)
        current_equity = account.get("equity", 0.0)
        risk_pct = self.risk_manager.calculate_scaled_risk(current_balance, current_equity=current_equity, session=session)
        
        if risk_pct <= 0.0:
            logger.warning("Risk scaling returned 0.0 — Halting trade.")
            return False

        # 6. Lot Size Calculation
        sym_info = self.connection.get_symbol_info(symbol)
        if not sym_info:
            return False
            
        risk_dollar = current_balance * (risk_pct / 100.0)
        sl_dist = abs(signal.entry_price - signal.stop_loss)
        
        lot = LotCalculator.calculate(
            risk_amount=risk_dollar,
            sl_distance=sl_dist,
            tick_size=sym_info.get("point", 0.01),
            tick_value=sym_info.get("trade_tick_value", 1.0),
            volume_min=sym_info.get("volume_min", 0.01),
            volume_max=sym_info.get("volume_max", 100.0),
            volume_step=sym_info.get("volume_step", 0.01)
        )
        
        # 7. Order Execution
        logger.info("Executing %s %s | Lot: %s | SL: %s | TP: %s", signal.direction, symbol, lot, signal.stop_loss, signal.take_profit)
        
        ticket = self.connection.place_order(
            symbol=symbol,
            signal=signal,
            lot_size=lot
        )
        
        latency_ms = (time.time() - start_time) * 1000
        logger.info(f"Execution Latency: {latency_ms:.2f}ms")
        
        if ticket:
            with self.state_lock:
                self.position_meta[ticket] = {
                    "ticket": ticket,
                    "session": session,
                    "best_price": current_price,
                    "partial_closed_count": 0,
                    "entry_time": time.time()
                }
                
            self.notification_manager.notify_trade_open(
                symbol=symbol, direction=signal.direction, entry=current_price, 
                lot=lot, sl=signal.stop_loss, tp=signal.take_profit
            )
            # Record circuit breaker
            self.risk_manager.circuit_breaker.record_trade()
            return True
            
        return False
