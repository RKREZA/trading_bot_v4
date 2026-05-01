import logging
from typing import Dict, List, Any
from datetime import datetime

from core.strategy.engine import TradeSignal
from core.time.time_service import time_service

logger = logging.getLogger("trading_bot.risk_engine")

class RiskEngine:
    def __init__(self, config: Dict[str, Any]):
        """
        config should contain:
        - max_risk_per_trade_pct: float
        - max_daily_loss_pct: float
        - max_drawdown_pct: float
        - max_open_trades: int
        - max_symbol_exposure_pct: float
        """
        self.config = config
        self.daily_pnl = 0.0
        self.peak_equity = 0.0
        self.current_equity = 0.0
        self.last_reset_day = None

    def _check_daily_reset(self):
        """Resets daily PnL based on server time day."""
        current_time = time_service.get_current_time()
        current_day = current_time.date()
        
        if self.last_reset_day != current_day:
            logger.info(f"RiskEngine: Server day rolled over to {current_day}. Resetting daily PnL.")
            self.daily_pnl = 0.0
            self.last_reset_day = current_day

    def update_equity(self, current_equity: float, daily_pnl: float):
        """Called periodically by ExecutionEngine to update state."""
        self.current_equity = current_equity
        self.daily_pnl = daily_pnl
        if current_equity > self.peak_equity:
            self.peak_equity = current_equity
            
        self._check_daily_reset()

    def validate_trade(self, signal: TradeSignal, open_positions: List[Dict[str, Any]]) -> bool:
        """
        Validates if a signal can be executed based on risk parameters.
        open_positions: list of dicts with keys ['symbol', 'volume', 'margin']
        """
        self._check_daily_reset()
        
        if self.current_equity <= 0:
            logger.warning("RiskEngine: Current equity is 0 or uninitialized. Trade rejected.")
            return False

        # 1. Max Open Trades
        max_trades = self.config.get("max_open_trades", 5)
        if len(open_positions) >= max_trades:
            logger.warning(f"RiskEngine: Max open trades ({max_trades}) reached.")
            return False

        # 2. Daily Loss Limit
        max_daily_loss_pct = self.config.get("max_daily_loss_pct", 5.0)
        max_daily_loss_amount = (max_daily_loss_pct / 100.0) * self.current_equity
        if self.daily_pnl <= -max_daily_loss_amount:
            logger.warning(f"RiskEngine: Daily loss limit reached (PnL: {self.daily_pnl}).")
            return False

        # 3. Drawdown Limit
        max_dd_pct = self.config.get("max_drawdown_pct", 10.0)
        if self.peak_equity > 0:
            current_dd_pct = ((self.peak_equity - self.current_equity) / self.peak_equity) * 100.0
            if current_dd_pct >= max_dd_pct:
                logger.warning(f"RiskEngine: Max drawdown ({max_dd_pct}%) reached. Current DD: {current_dd_pct}%.")
                return False

        # 4. Symbol Exposure
        max_symbol_exposure = self.config.get("max_symbol_exposure_pct", 20.0)
        # Note: accurate exposure calculation requires open position margin/value.
        # This is simplified.
        symbol_count = sum(1 for p in open_positions if p.get('symbol') == signal.symbol)
        if symbol_count >= 2: # Simple arbitrary limit for now
             logger.warning(f"RiskEngine: Max exposure for {signal.symbol} reached.")
             return False

        return True

    def calculate_position_size(self, signal: TradeSignal, risk_pct: float = None) -> float:
        """
        Calculates position size (in lots) based on risk percentage and stop loss distance.
        """
        risk_pct = risk_pct or self.config.get("max_risk_per_trade_pct", 1.0)
        risk_amount = self.current_equity * (risk_pct / 100.0)
        
        sl_distance = abs(signal.entry - signal.stop_loss)
        if sl_distance == 0:
            logger.warning("RiskEngine: Stop loss distance is 0. Returning 0 volume.")
            return 0.0
            
        # Simplified lot calculation. 
        # In reality, needs tick_value and contract_size from MT5Service.
        # volume = risk_amount / (sl_distance * tick_value_per_lot)
        # Assuming normalized pip value is handled by the caller or MT5Service.
        
        return risk_amount # Caller must translate this to MT5 volume using MT5Service
