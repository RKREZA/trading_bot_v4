from pydantic import BaseModel, Field, ConfigDict
import numpy as np
from dataclasses import dataclass, fields, field
from typing import Dict, List, Optional, Any, Tuple
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
    @property
    def bid(self): return self.close
    @property
    def ask(self): return self.close + self.spread

@dataclass(slots=True)
class CandleArray:
    """Vectorized container for OHLCVT+S candle data."""
    time: np.ndarray      # int64 timestamps
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    tick_volume: np.ndarray
    spread: np.ndarray     # int/float spread in points/pips

    @classmethod
    def from_dicts(cls, candles: list[dict]) -> "CandleArray":
        return cls(
            time=np.array([c['time'] for c in candles], dtype=np.int64),
            open=np.array([c['open'] for c in candles]),
            high=np.array([c['high'] for c in candles]),
            low=np.array([c['low'] for c in candles]),
            close=np.array([c['close'] for c in candles]),
            tick_volume=np.array([c.get('tick_volume', 0) for c in candles]),
            spread=np.array([c.get('spread', 0) for c in candles]),
        )
    
    def __len__(self):
        return len(self.time)
    
    def slice(self, start: int, end: int) -> "CandleArray":
        args = [getattr(self, f.name)[start:end] for f in fields(self)]
        return CandleArray(*args)

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
                tick_volume=int(self.tick_volume[idx]),
                spread=int(self.spread[idx])
            )
        raise TypeError(f"Invalid argument type: {type(idx)}")

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def ema(self, period: int) -> np.ndarray:
        """High-performance vectorized Exponential Moving Average."""
        if len(self.close) < period:
            return np.full_like(self.close, np.nan)
        
        alpha = 2 / (period + 1)
        ema_values = np.zeros_like(self.close)
        ema_values[period-1] = np.mean(self.close[:period])
        
        for i in range(period, len(self.close)):
            ema_values[i] = (self.close[i] - ema_values[i-1]) * alpha + ema_values[i-1]
            
        return ema_values

    def rsi(self, period: int = 14) -> np.ndarray:
        """High-performance vectorized Relative Strength Index."""
        if len(self.close) < period + 1:
            return np.full_like(self.close, np.nan)
            
        delta = np.diff(self.close)
        gain = (delta > 0) * delta
        loss = (delta < 0) * -delta
        
        avg_gain = np.full_like(self.close, np.nan)
        avg_loss = np.full_like(self.close, np.nan)
        
        avg_gain[period] = np.mean(gain[:period])
        avg_loss[period] = np.mean(loss[:period])
        
        alpha = 1 / period
        for i in range(period + 1, len(self.close)):
            avg_gain[i] = (gain[i-1] - avg_gain[i-1]) * alpha + avg_gain[i-1]
            avg_loss[i] = (loss[i-1] - avg_loss[i-1]) * alpha + avg_loss[i-1]
            
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def atr(self, period: int = 14) -> np.ndarray:
        """High-performance vectorized Average True Range."""
        if len(self.close) < period + 1:
            return np.full_like(self.close, np.nan)
            
        h, l, c_prev = self.high[1:], self.low[1:], self.close[:-1]
        tr = np.maximum(h - l, np.maximum(np.abs(h - c_prev), np.abs(l - c_prev)))
        
        atr = np.full_like(self.close, np.nan)
        atr[period] = np.mean(tr[:period])
        
        alpha = 1.0 / period
        for i in range(period + 1, len(self.close)):
            atr[i] = (tr[i-1] - atr[i-1]) * alpha + atr[i-1]
            
        return atr

    def bollinger_bands(self, period: int = 20, std_dev: float = 2.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """High-performance vectorized Bollinger Bands."""
        if len(self.close) < period:
            nan_arr = np.full_like(self.close, np.nan)
            return nan_arr, nan_arr, nan_arr
            
        # Use Simple Moving Average
        sma = np.array([np.mean(self.close[i-period+1:i+1]) if i >= period-1 else np.nan for i in range(len(self.close))])
        
        # Vectorized Standard Deviation (Rolling)
        rolling_std = np.array([np.std(self.close[i-period+1:i+1]) if i >= period-1 else np.nan for i in range(len(self.close))])
        
        upper = sma + (rolling_std * std_dev)
        lower = sma - (rolling_std * std_dev)
        return upper, lower, sma

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
    risk_per_trade_pct: float = 0.5
    max_daily_loss_pct: float = 5.0
    max_drawdown_halt_pct: float = 20.0

class BotConfig(BaseModel):
    model_config = ConfigDict(extra='allow')
    symbol: str = "XAUUSDm"
    risk: RiskConfig = Field(default_factory=RiskConfig)
    backtest: Dict[str, Any] = Field(default_factory=dict)
