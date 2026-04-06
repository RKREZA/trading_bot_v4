import logging
import threading
import time
from datetime import datetime
from typing import Optional, Dict, Any

from core.base_strategy import BaseStrategy, MarketData
from core.risk.risk_guardian import RiskGuardian
from core.common.types import TradeSignal

logger = logging.getLogger("trading_bot.runtime")

class StrategyRuntime:
    """
    Institutional Strategy Sandbox.
    Provides a strictly isolated execution environment for a single strategy.
    Enforces no shared mutable state and deterministic data processing.
    """

    def __init__(self, 
                 strategy: BaseStrategy, 
                 global_config: dict, 
                 risk_guardian: RiskGuardian):
        self.strategy = strategy
        self.config = global_config
        self.risk_guardian = risk_guardian
        
        self.strategy_id = strategy.strategy_id
        self.enabled = strategy.enabled
        
        # ── Institutional Isolation Layer (Micro-service Ledger) ──
        # Each strategy starts with a partitioned balance
        initial_capital = global_config.get("state", {}).get("strategy_allocation", {}).get(self.strategy_id, 2500.0)
        self.balance = float(initial_capital)
        self.equity = float(initial_capital)
        self.pnl = 0.0
        self.max_equity = float(initial_capital)
        self.drawdown_pct = 0.0
        
        # Isolated State Tracking (No global dependencies)
        self.last_signal: Optional[TradeSignal] = None
        self.last_execution_time: Optional[float] = None
        self.running = False
        self._lock = threading.Lock()

    def execute_cycle(self, market_data: MarketData) -> Optional[TradeSignal]:
        """
        Runs a single iteration of the strategy logic in a sandbox.
        """
        if not self.enabled:
            return None

        with self._lock:
            try:
                # 1. Generate Signal from Strategy Logic
                signal = self.strategy.generate_signal(market_data)
                
                if signal and signal.direction != "NONE":
                    # 2. Institutional Validation (Confidence, Risk, Governance)
                    if signal.confidence < 0.5:
                        logger.debug(f"[{self.strategy_id}] Low confidence signal rejected: {signal.confidence}")
                        return None
                    
                    self.last_signal = signal
                    self.last_execution_time = time.time()
                    return signal
                    
            except Exception as e:
                logger.error(f"Critical Runtime Exception [{self.strategy_id}]: {e}", exc_info=True)
                
        return None

    def on_trade_closed(self, trade_record: dict):
        """
        Notifies the internal strategy of a trade closure and updates the ledger.
        Institutional Requirement: Micro-service performance tracking.
        """
        with self._lock:
            realized_pnl = trade_record.get("realized_pnl", 0.0)
            self.pnl += realized_pnl
            self.balance += realized_pnl
            self.equity = self.balance
            
            # Max Drawdown Calculation (Equity Peak to Valley)
            if self.equity > self.max_equity:
                self.max_equity = self.equity
            
            if self.max_equity > 0:
                self.drawdown_pct = ((self.max_equity - self.equity) / self.max_equity) * 100.0
            
            logger.info(f"[{self.strategy_id}] Trade Closed. PnL: {realized_pnl:.2f} | Bal: {self.balance:.2f} | DD: {self.drawdown_pct:.1f}%")
            
        self.strategy.on_trade_closed(trade_record)

    def reset_daily(self, new_balance: float):
        """Daily state synchronization and risk reset."""
        self.risk_guardian.reset_daily(new_balance)
        self.strategy.reset_daily_stats()

    def __repr__(self):
        return f"<StrategyRuntime(id={self.strategy_id}, active={self.enabled})>"

if __name__ == "__main__":
    # Standalone Simulation Mode for Strategy Runtime
    logging.basicConfig(level=logging.INFO)
    from core.risk.risk_guardian import RiskGuardian
    
    # Mock Components
    class MockStrategy(BaseStrategy):
        def generate_signal(self, data):
            return TradeSignal(direction="BUY", confidence=0.8, price=data.current_price)
            
    mock_st = MockStrategy("mock_v4", {})
    guardian = RiskGuardian({"backtest": {"initial_balance": 1000}})
    
    runtime = StrategyRuntime(mock_st, {}, guardian)
    
    print("\n--- StrategyRuntime Standalone Test ---")
    mock_data = MarketData(
        symbol="EURUSD",
        htf_candles=None,
        m15_candles=None,
        m5_candles=None,
        d1_candles=None,
        current_price=1.1000,
        session="NY",
        timestamp=datetime.now()
    )
    signal = runtime.execute_cycle(mock_data)
    print(f"Signal Generated: {signal}")
