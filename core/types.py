from pydantic import BaseModel, Field, ConfigDict
import numpy as np
from dataclasses import dataclass, fields, field
from typing import Dict, List, Optional, Any
from enum import Enum
from datetime import datetime

class MarketRegime(Enum):
    TREND           = "TREND"
    RANGE           = "RANGE"
    HIGH_VOLATILITY = "HIGH_VOLATILITY"
    LOW_VOLATILITY  = "LOW_VOLATILITY"
    UNCERTAIN       = "UNCERTAIN"

class Candle:
    """Supports both attribute access (c.close) and dict access (c['close'])."""
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
    def __getitem__(self, key):
        return self.__dict__[key]
    def __repr__(self):
        return f"Candle({self.__dict__})"

@dataclass(slots=True)
class CandleArray:
    """Vectorized container for OHLCVT candle data."""
    time: np.ndarray      # int64 timestamps
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    tick_volume: np.ndarray

    @classmethod
    def from_dicts(cls, candles: list[dict]) -> "CandleArray":
        return cls(
            time=np.array([c['time'] for c in candles], dtype=np.int64),
            open=np.array([c['open'] for c in candles]),
            high=np.array([c['high'] for c in candles]),
            low=np.array([c['low'] for c in candles]),
            close=np.array([c['close'] for c in candles]),
            tick_volume=np.array([c.get('tick_volume', 0) for c in candles]),
        )
    
    def __len__(self):
        return len(self.time)
    
    def slice(self, start: int, end: int) -> "CandleArray":
        return CandleArray(*(getattr(self, f.name)[start:end] for f in fields(self)))

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            start = idx.start if idx.start is not None else 0
            stop = idx.stop if idx.stop is not None else len(self.time)
            return self.slice(start, stop)
        elif isinstance(idx, (np.ndarray, list)):
            return CandleArray(*(getattr(self, f.name)[idx] for f in fields(self)))
        elif isinstance(idx, int):
            return Candle(
                time=int(self.time[idx]),
                open=float(self.open[idx]),
                high=float(self.high[idx]),
                low=float(self.low[idx]),
                close=float(self.close[idx]),
                tick_volume=int(self.tick_volume[idx])
            )
        raise TypeError(f"Invalid argument type: {type(idx)}")

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

@dataclass
class TradeSignal:
    """
    V4 Institutional Trade Signal.
    Confidence is a mandatory float between 0.0 and 1.0.
    """
    direction: str  # BUY | SELL | NONE
    price: float = 0.0
    confidence: float = 0.0  # Mandatory 0.0 to 1.0
    timestamp: Optional[datetime] = None
    stop_loss: float = 0.0
    take_profit: float = 0.0
    session: str = "GLOBAL"
    tp1_price: float = 0.0
    tp2_price: float = 0.0
    confluence_score: int = 0
    reasons: List[str] = field(default_factory=list)
    rr_ratio: float = 2.0

class RiskConfig(BaseModel):
    model_config = ConfigDict(extra='allow')
    risk_per_trade: float = 1.0
    max_daily_loss_percent: float = 5.0
    max_drawdown_halt_pct: float = 20.0

class BotConfig(BaseModel):
    model_config = ConfigDict(extra='allow')
    symbol: str = "XAUUSDm"
    risk: RiskConfig = Field(default_factory=RiskConfig)
    backtest: Dict[str, Any] = Field(default_factory=dict)
