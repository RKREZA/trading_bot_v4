"""
TRADING BOT V3 — Strategy Runtime Container
The isolation unit: wraps a strategy with its own dedicated
RiskManager, PerformanceTracker, and PositionTracker.

Zero shared mutable state between runtimes.
"""

import logging
import threading
from datetime import date
from typing import Any, Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .broker_clock import BrokerClock

from .base_strategy import BaseStrategy, MarketData, TaggedSignal
from .risk_manager import RiskManager
from .performance_tracker import PerformanceTracker
from .position_tracker import PositionTracker
from .strategy_engine import TradeSignal

logger = logging.getLogger("trading_bot.strategy_runtime")


class StrategyState:
    """
    Mutable internal state for a single strategy runtime.
    Tracks daily counters, cooldowns, and session-specific data.
    """
    def __init__(self):
        self.daily_trades: int = 0
        self.daily_losses: int = 0
        self.consecutive_losses: int = 0
        self.last_reset_day: Optional[date] = None
        self.session_consecutive_losses: Dict[str, int] = {}

    def reset_daily(self):
        self.daily_trades = 0
        self.daily_losses = 0
        self.consecutive_losses = 0
        self.session_consecutive_losses = {}

    def record_trade_result(self, pnl: float, session: str = ""):
        self.daily_trades += 1
        if pnl < 0:
            self.daily_losses += 1
            self.consecutive_losses += 1
            if session:
                self.session_consecutive_losses[session] = \
                    self.session_consecutive_losses.get(session, 0) + 1
        else:
            self.consecutive_losses = 0
            if session:
                self.session_consecutive_losses[session] = 0


class StrategyRuntime:
    """
    The core isolation container. Each strategy gets its own runtime
    with completely independent components.
    
    Components (per-strategy, no sharing):
        - strategy:    BaseStrategy implementation
        - risk_manager: RiskManager with strategy-specific config
        - performance:  PerformanceTracker for PnL/DD/Sharpe
        - positions:    PositionTracker for open trades
        - state:        StrategyState for daily counters
    
    Usage:
        runtime = StrategyRuntime(
            strategy=SniperStrategy("sniper_v1", config),
            global_config=full_config,
            initial_balance=1000.0
        )
        signal = runtime.generate_signal(market_data)
    """

    def __init__(self, strategy: BaseStrategy, global_config: dict, 
                 initial_balance: float = 1000.0, broker_clock: 'BrokerClock' = None):
        """
        Args:
            strategy: A concrete BaseStrategy implementation
            global_config: Full bot config (for risk defaults, session config, etc.)
            initial_balance: Starting balance for performance tracking
            broker_clock: BrokerClock instance for authoritative time
        """
        self.strategy = strategy
        self.strategy_id = strategy.strategy_id

        # Build strategy-specific risk config by merging:
        # global risk defaults <- strategy-specific overrides
        risk_cfg = dict(global_config.get("risk", {}))
        strategy_risk = strategy.config.get("risk", {})
        risk_cfg.update(strategy_risk)

        # Create a merged config for the risk manager
        risk_config = dict(global_config)
        risk_config["risk"] = risk_cfg
        
        # Dedicated components — zero shared state
        self.risk_manager = RiskManager(risk_config, broker_clock=broker_clock)
        self.risk_manager.silent = True  # Controlled logging via runtime
        self.performance = PerformanceTracker(self.strategy_id, initial_balance)
        self.positions = PositionTracker(self.strategy_id)
        self.state = StrategyState()
        
        self._lock = threading.Lock()

    @property
    def enabled(self) -> bool:
        return self.strategy.enabled

    def generate_signal(self, market_data: MarketData) -> Optional[TaggedSignal]:
        """
        Generate a signal from the strategy and wrap it with attribution.
        
        Args:
            market_data: Frozen, read-only market data
            
        Returns:
            TaggedSignal if conditions met, None otherwise
        """
        if not self.enabled:
            return None

        signal = self.strategy.generate_signal(market_data)
        if signal is None:
            return None

        return TaggedSignal(
            signal=signal,
            strategy_id=self.strategy_id
        )

    def check_risk(self, balance: float, equity: float, session: str = "") -> tuple:
        """
        Run this runtime's circuit breakers and risk checks.
        
        Returns:
            Tuple of (allowed: bool, reason: str)
        """
        return self.risk_manager.check_circuit_breakers(
            current_balance=balance,
            current_equity=equity,
            daily_trades=self.state.daily_trades,
            daily_losses=self.state.daily_losses,
            consecutive_losses=self.state.consecutive_losses
        )

    def calculate_risk_pct(self, balance: float, equity: float = None, 
                           session: str = None) -> float:
        """Calculate scaled risk percentage for this strategy."""
        return self.risk_manager.calculate_scaled_risk(
            balance, current_equity=equity, session=session
        )

    def on_trade_opened(self, ticket: int, metadata: dict) -> None:
        """Called when a trade is successfully opened for this strategy."""
        self.positions.add_position(ticket, metadata)
        self.risk_manager.circuit_breaker.record_trade()
        logger.info("[%s] Trade opened: ticket=%d", self.strategy_id, ticket)

    def on_trade_closed(self, ticket: int, trade_record: dict) -> None:
        """
        Called when a trade belonging to this strategy closes.
        Updates all per-strategy tracking components.
        """
        pnl = trade_record.get("pnl", 0.0)
        session = trade_record.get("session", "")

        # Update position tracker
        self.positions.remove_position(ticket)
        self.positions.record_trade(trade_record)

        # Update performance tracker
        self.performance.record_trade(trade_record)

        # Update risk manager history (for Kelly)
        self.risk_manager.update_history(trade_record)

        # Update state counters
        self.state.record_trade_result(pnl, session)

        # Notify strategy (for internal state like cooldowns)
        self.strategy.on_trade_closed(trade_record)

        logger.info(
            "[%s] Trade closed: ticket=%d pnl=$%.2f result=%s",
            self.strategy_id, ticket, pnl, trade_record.get("result", "?")
        )

    def reset_daily(self, balance: float) -> None:
        """Reset all daily-scoped state."""
        self.state.reset_daily()
        self.performance.reset_daily_stats()
        self.risk_manager.reset_daily_stats(balance)
        self.strategy.reset_daily_stats()

    def get_state(self) -> dict:
        """Serialize runtime state for persistence."""
        return {
            "strategy_id": self.strategy_id,
            "enabled": self.enabled,
            "positions": self.positions.get_state(),
            "performance": self.performance.get_summary(),
            "daily_trades": self.state.daily_trades,
            "daily_losses": self.state.daily_losses,
            "consecutive_losses": self.state.consecutive_losses,
        }

    def load_state(self, state: dict) -> None:
        """Restore runtime state from persistence."""
        if not state:
            return
        self.positions.load_state(state.get("positions", {}))
        self.state.daily_trades = state.get("daily_trades", 0)
        self.state.daily_losses = state.get("daily_losses", 0)
        self.state.consecutive_losses = state.get("consecutive_losses", 0)

    def __repr__(self) -> str:
        return (
            f"<StrategyRuntime({self.strategy_id}) "
            f"enabled={self.enabled} "
            f"positions={self.positions.open_count} "
            f"trades={self.state.daily_trades}>"
        )
