import logging
from typing import Dict, Any, Optional
from core.strategy_engine import StrategyEngine, TradeSignal
from core.ai_advisor import AIAdvisor
from core.risk_manager import RiskManager
from core.connection import MT5Connection, PositionManager
from core.notifications import NotificationManager
from core.lot_calculator import LotCalculator

logger = logging.getLogger("trading_bot.execution")

class ExecutionPipeline:
    """Orchestrates signal generation, AI vetting, risk checks, and order execution."""
    
    def __init__(self, 
                 config: "BotConfig", 
                 connection: MT5Connection, 
                 position_manager: PositionManager,
                 strategy: StrategyEngine, 
                 ai_advisor: AIAdvisor, 
                 risk_manager: RiskManager,
                 notification_manager: NotificationManager):
        self.config = config
        self.connection = connection
        self.position_manager = position_manager
        self.strategy = strategy
        self.ai_advisor = ai_advisor
        self.risk_manager = risk_manager
        self.notification_manager = notification_manager
        
    def execute_cycle(self, symbol: str, m30: "CandleArray", h1: "CandleArray", h4: "CandleArray", m5: "CandleArray", d1: "CandleArray", current_price: float, session: str) -> bool:
        """Runs one full execution cycle for a symbol."""
        # 1. Generate Signal
        signal, trend, regime = self.strategy.analyze(
            m30_candles=m30,
            h1_candles=h1,
            h4_candles=h4,
            m5_candles=m5,
            d1_candles=d1,
            current_price=current_price,
            session=session
        )
        
        if not signal:
            return False
            
        # 2. Open Position Check
        if self.position_manager.has_open_position(symbol):
            logger.info("Signal ignored — position already open for %s", symbol)
            return False
            
        # 3. Circuit Breaker Check
        allowed, reason = self.risk_manager.circuit_breaker.check_all({
            "daily_losses": self.risk_manager.trade_history,
            "margin_level": self.connection.account_info.get("margin_level", 9999) if self.connection.account_info else 9999
        })
        if not allowed:
            self.notification_manager.notify_critical("CIRCUIT BREAKER", reason)
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

        # 5. Risk Scaling
        account = self.connection.get_account_snapshot()
        current_balance = account.get("balance", 0.0)
        risk_pct = self.risk_manager.calculate_scaled_risk(current_balance, session=session)
        
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
        
        ticket = self.position_manager.place_order(
            symbol=symbol,
            order_type=signal.direction,
            volume=lot,
            price=current_price,
            sl=signal.stop_loss,
            tp=signal.take_profit,
            magic_number=self.config.magic_number if hasattr(self.config, 'magic_number') else self.config.get("magic_number", 234000),
            comment="B3 Signal"
        )
        
        if ticket:
            self.notification_manager.notify_trade_open(
                symbol=symbol, direction=signal.direction, entry=current_price, 
                lot=lot, sl=signal.stop_loss, tp=signal.take_profit
            )
            # Record circuit breaker
            self.risk_manager.circuit_breaker.record_trade()
            return True
            
        return False
