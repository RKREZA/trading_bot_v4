"""
core/portfolio_manager.py
Manages portfolio-level risks, allocation, and trade conflicts.
"""
import logging
from typing import List, Dict, Optional, Any

logger = logging.getLogger("trading_bot.portfolio")

class PortfolioManager:
    def __init__(self, risk_manager: Any, config: dict, state_manager: Any):
        """
        Args:
            risk_manager: Instance of RiskManager.
            config: Global bot configuration (config.json).
            state_manager: Instance of SecureStateManager (or similar).
        """
        self.risk_manager = risk_manager
        self.config = config
        self.state_manager = state_manager
        
        # Portfolio Config
        self.allocation = config.get("strategy_allocation", {
            "smc_v1": 0.6,
            "sniper_v1": 0.4
        })
        self.max_exposure_per_symbol = config.get("portfolio_risk", {}).get("max_exposure_per_symbol", 0.1)

    def process_signals(self, signals: List[dict], account_info: dict, open_positions: List[Any]) -> List[dict]:
        """
        Filters signals based on portfolio rules.
        Returns a list of approved signals.
        """
        approved = []
        equity = account_info.get("equity", 0.0)
        
        for signal in signals:
            strategy_id = signal.get("strategy")
            symbol = signal.get("symbol")
            direction = signal.get("direction")
            
            # 1. Check for Conflicts
            if self._has_conflict(symbol, direction, open_positions):
                logger.warning(f"CONFLICT: {strategy_id} signal rejected for {symbol} due to opposite open position.")
                continue
                
            # 2. Risk Manager Validation
            if not self.risk_manager.can_take_trade(signal):
                logger.info(f"RISK HALT: {strategy_id} signal rejected by risk manager.")
                continue
                
            # 3. Capital Allocation 
            # We scale the trade risk by the allocation percentage
            strat_alloc = self.allocation.get(strategy_id, 0.0)
            if strat_alloc <= 0:
                logger.warning(f"ALLOCATION ZERO: {strategy_id} has no capital allocated.")
                continue
            
            # Update signal with final risk amount based on allocated equity?
            # Or just pass it through with the strategy's suggested risk and let the execution engine handle it.
            # Requirement 4 says: allocated_equity = total_equity * allocation[strategy]
            # This implies the risk_dollar should be based on this allocated_equity.
            
            # Let's add the scale factor for the execution engine
            signal["allocation_scale"] = strat_alloc
            
            approved.append(signal)

        return approved

    def _has_conflict(self, symbol: str, direction: str, open_positions: List[Any]) -> bool:
        """
        Returns True if an opposite position exists on the same symbol.
        """
        # MT5 positions have type: 0 for BUY, 1 for SELL
        target_type = 0 if direction == "BUY" else 1
        
        for pos in open_positions:
            if pos.symbol == symbol:
                # Conflict if existing position is opposite of the signal
                if pos.type != target_type:
                    return True
        return False
