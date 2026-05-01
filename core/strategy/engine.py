from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field
import pandas as pd
from datetime import datetime
import uuid

class TradeSignal(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    execution_id: str = Field(..., description="Unique ID to prevent duplicate execution")
    symbol: str
    direction: str = Field(..., description="'BUY' or 'SELL'")
    entry: float
    stop_loss: float
    take_profit: float
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    timestamp: datetime = Field(..., description="TimeService aligned UTC timestamp")
    metadata: Dict[str, Any] = Field(default_factory=dict)

class BaseStrategy(ABC):
    def __init__(self, name: str, symbol: str, timeframe: int = 5):
        self.name = name
        self.symbol = symbol
        self.timeframe = timeframe  # Timeframe in minutes
        self.active = False
        
    @abstractmethod
    def on_tick(self, tick_data: dict, timestamp: datetime) -> Optional[TradeSignal]:
        """
        Called on every tick.
        timestamp: TimeService aligned timestamp in UTC.
        """
        pass

    @abstractmethod
    def on_candle(self, df: pd.DataFrame, timestamp: datetime) -> Optional[TradeSignal]:
        """
        Called on a new closed candle.
        timestamp: TimeService aligned timestamp in UTC of the exact candle close.
        """
        pass

    def get_execution_id(self, timestamp: datetime) -> str:
        """Generates a unique execution ID based on strategy, symbol, timeframe, and timestamp."""
        # This ensures that for a given candle/timestamp, only one signal is generated.
        return f"{self.name}_{self.symbol}_{self.timeframe}_{timestamp.isoformat()}"
