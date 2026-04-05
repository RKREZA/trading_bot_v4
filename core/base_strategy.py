"""
TRADING BOT V4 — Strategy Abstraction Layer
===========================================
Defines the mandatory contract for all institutional strategy implementations.
"""

import uuid
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

# Institutional types
from .types import TradeSignal, CandleArray, MarketRegime

logger = logging.getLogger("trading_bot.base_strategy")

@dataclass(frozen=True)
class MarketData:
    """
    Immutable (frozen) container for all market data fed to strategies.
    Shared read-only across all strategy runtimes in a single cycle.
    """
    symbol: str
    htf_candles: CandleArray
    m15_candles: CandleArray
    m5_candles: CandleArray
    d1_candles: Optional[CandleArray]
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
        return self.signal.price

    @property
    def stop_loss(self) -> float:
        return self.signal.stop_loss

    @property
    def take_profit(self) -> float:
        return self.signal.take_profit


class BaseStrategy(ABC):
    """
    V4 Institutional Base Strategy Interface.
    Each strategy behaves as a micro-service: independent and stateless.
    """

    def __init__(self, strategy_id: str, config: dict):
        self.strategy_id = strategy_id
        self.config = config
        self.enabled = config.get("enabled", True)

    @abstractmethod
    def generate_signal(self, market_data: MarketData) -> Optional[TradeSignal]:
        """
        MUST return a TradeSignal object or None.
        Confidence is mandatory 0.0 to 1.0.
        """
        ...

    @abstractmethod
    def get_stop_loss(self, signal: TradeSignal, market_data: MarketData) -> float:
        """Calculate the absolute price for Stop Loss."""
        ...

    @abstractmethod
    def get_take_profit(self, signal: TradeSignal, market_data: MarketData) -> float:
        """Calculate the absolute price for Take Profit."""
        ...

    def on_trade_closed(self, trade_record: dict) -> None:
        """Optional hook for updating internal counters/cooldowns."""
        pass

    def reset_daily_stats(self) -> None:
        """Called at start of each trading day for reset logic."""
        pass

    def get_config(self) -> dict:
        return dict(self.config)

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}(id={self.strategy_id}, enabled={self.enabled})>"
