"""
TRADING BOT V3 — Strategy Abstraction Layer
Defines the mandatory contract for all strategy implementations,
plus shared data types for the multi-strategy framework.
"""

import uuid
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from .types import CandleArray

from .strategy_engine import TradeSignal

logger = logging.getLogger("trading_bot.base_strategy")


@dataclass(frozen=True)
class MarketData:
    """
    Immutable (frozen) container for all market data fed to strategies.
    Shared read-only across all strategy runtimes in a single cycle.
    """
    symbol: str
    htf_candles: Any         # CandleArray (H1)
    m15_candles: Any         # CandleArray (M15)
    m5_candles: Any          # CandleArray (M5)
    d1_candles: Any          # CandleArray (D1)
    current_price: float
    session: str
    timestamp: datetime
    preprocessed: Optional[dict] = None   # Precomputed indicators per M5 bar


@dataclass
class TaggedSignal:
    """
    Wraps a TradeSignal with strategy attribution metadata.
    Every order in the system carries this tag for trade attribution.
    """
    signal: TradeSignal
    strategy_id: str
    trade_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])

    @property
    def direction(self) -> str:
        return self.signal.direction

    @property
    def entry_price(self) -> float:
        return self.signal.entry_price

    @property
    def stop_loss(self) -> float:
        return self.signal.stop_loss

    @property
    def take_profit(self) -> float:
        return self.signal.take_profit


class BaseStrategy(ABC):
    """
    Abstract base class that all strategies MUST implement.
    
    Contract:
        - generate_signal() → produces a TradeSignal or None
        - on_trade_closed()  → notified when one of its trades closes
        - preprocess()       → optional bulk preprocessing for backtests
        - get_config()       → returns strategy-specific config snapshot
    
    Each concrete strategy:
        - Owns its own internal state (cooldowns, counters, etc.)
        - Has ZERO shared mutable state with other strategies
        - Receives market data as a frozen, read-only MarketData object
    """

    def __init__(self, strategy_id: str, config: dict):
        """
        Args:
            strategy_id: Unique identifier (e.g. "sniper_v1", "smc_v2")
            config: Strategy-specific configuration dict
        """
        self.strategy_id = strategy_id
        self.config = config
        self.strategy_name = config.get("name", strategy_id)
        self._enabled = config.get("enabled", True)

    @property
    def enabled(self) -> bool:
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool):
        self._enabled = value

    @abstractmethod
    def generate_signal(self, market_data: MarketData) -> Optional[dict]:
        """
        Core signal generation. Receives immutable market data, returns
        a standardized signal dictionary if conditions are met, or None.
        
        Signal Format:
        {
            "strategy": str,
            "symbol": str,
            "direction": "BUY" | "SELL",
            "entry": float,
            "sl": float,
            "tp": float,
            "risk": float,          # fraction of capital (0.0–1.0)
            "confidence": float     # 0.0–1.0
        }
        
        MUST NOT modify market_data or any shared state.
        """
        ...

    def on_trade_closed(self, trade_record: dict) -> None:
        """
        Called when a trade belonging to this strategy closes.
        Strategies can override to update internal state (e.g. consecutive losses).
        
        Args:
            trade_record: Dict with keys: ticket, pnl, result, session, etc.
        """
        pass

    def on_tick(self, tick_data: dict) -> None:
        """
        Called on each tick for strategies that need tick-level processing.
        Default is a no-op; override if needed.
        
        Args:
            tick_data: Dict with keys: bid, ask, spread, time
        """
        pass

    def on_order_update(self, order_event: dict) -> None:
        """
        Called when an order status changes (filled, partial, rejected).
        Default is a no-op; override if needed.
        """
        pass

    def preprocess(self, htf: Any, m15: Any, m5: Any, d1: Any) -> Optional[dict]:
        """
        Optional bulk preprocessing for backtest efficiency.
        Returns a dict of precomputed indicators indexed by M5 bar.
        Default returns None (no preprocessing).
        """
        return None

    def reset_daily_stats(self) -> None:
        """Called at the start of each new trading day."""
        pass

    def get_config(self) -> dict:
        """Returns the strategy's configuration snapshot."""
        return dict(self.config)

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}(id={self.strategy_id}, enabled={self.enabled})>"
