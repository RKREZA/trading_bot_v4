from pydantic import BaseModel, Field, ConfigDict
import numpy as np
from dataclasses import dataclass, fields, field
from typing import Dict, List, Optional, Any

@dataclass(slots=True)
class Trade:
    symbol: str
    direction: str  # BUY / SELL
    entry: float
    sl: float
    size: float
    remaining_size: float
    
    tp1: float
    tp2: float
    tp1_hit: bool = False
    tp2_hit: bool = False
    
    breakeven_moved: bool = False
    trailing_active: bool = False
    
    open_time: Any = None
    close_time: Any = None
    
    result: str = ""
    partial_pnls: list = field(default_factory=list)
    tick_size: float = 0.01
    tick_value: float = 1.0
    strategy_id: str = ""
    session: str = ""
    ticket: int = 0
    signal: dict = field(default_factory=dict)
    best_price: float = 0.0

@dataclass(slots=True)
class CandleArray:
    """
    High-performance vectorized container for OHLCVT candle data.
    Uses NumPy arrays internally to enable O(1) indicator calculations 
    and avoid the overhead of list-of-dicts processing.
    
    Attributes:
        time (np.ndarray): int64 Unix timestamps.
        open (np.ndarray): float64 open prices.
        high (np.ndarray): float64 high prices.
        low (np.ndarray): float64 low prices.
        close (np.ndarray): float64 close prices.
        tick_volume (np.ndarray): int64 tick volumes.
    """
    time: np.ndarray      # int64 timestamps
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    tick_volume: np.ndarray

    @classmethod
    def from_dicts(cls, candles: list[dict]) -> "CandleArray":
        """
        Factory method to convert a list of dictionaries into a vectorized CandleArray.
        
        Args:
            candles (list[dict]): List of MT5-style candle dictionaries.
            
        Returns:
            CandleArray: Initialized vectorized object.
        """
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
        """Slice the arrays similarly to Python lists."""
        return CandleArray(*(getattr(self, f.name)[start:end] for f in fields(self)))

    def __getitem__(self, idx):
        """
        Provides intuitive access to the data. 
        Returns a sliced CandleArray if idx is a slice, or a dictionary if idx is an int.
        
        Args:
            idx (slice | int): Index or range.
            
        Returns:
            CandleArray | dict: Sliced object or single candle dictionary.
        """
        if isinstance(idx, slice):
            start = idx.start if idx.start is not None else 0
            stop = idx.stop if idx.stop is not None else len(self.time)
            return self.slice(start, stop)
        elif isinstance(idx, (np.ndarray, list)):
            # Support boolean masking or advanced indexing
            return CandleArray(*(getattr(self, f.name)[idx] for f in fields(self)))
        elif isinstance(idx, int):
            return {
                'time': int(self.time[idx]),
                'open': float(self.open[idx]),
                'high': float(self.high[idx]),
                'low': float(self.low[idx]),
                'close': float(self.close[idx]),
                'tick_volume': int(self.tick_volume[idx])
            }
        raise TypeError(f"Invalid argument type: {type(idx)}")

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

# Configuration Models

class RiskConfig(BaseModel):
    """
    Schema for account-level risk management parameters.
    Encapsulates circuit breakers, position limits, and Kelly Criterion settings.
    """
    model_config = ConfigDict(extra='allow')
    risk_per_trade: float = 1.0
    max_daily_trades: int = 5
    daily_goal: float = 200.0
    max_daily_loss_percent: float = 5.0
    max_drawdown_halt_pct: float = 10.0
    max_lot_size: float = 1.0
    max_open_positions: int = 2
    kelly_min_trades: int = 15
    drawdown_scaling: bool = True

class StrategyConfig(BaseModel):
    """
    Schema for strategy-specific execution parameters.
    Defines thresholds for entry signals, SL buffers, and cooldowns.
    """
    model_config = ConfigDict(extra='allow')
    min_confluence_score: int = 4
    min_confidence: int = 60
    cooldown_candles: int = 20
    min_sl_points: int = 150
    # Additional keys mapping strategy specific config

# BotConfig combines all sub-configs

class BotConfig(BaseModel):
    """
    Root configuration model for the Trading Bot.
    Aggregates risk, strategy, session, and backtest configurations.
    """
    model_config = ConfigDict(extra='allow')
    
    symbol: str = Field(default="XAUUSDm", description="Primary trading symbol")
    magic_number: int = Field(default=234000, description="Unique ID for bot's trades")
    risk: RiskConfig = Field(default_factory=RiskConfig)
    strategy_defaults: StrategyConfig = Field(default_factory=StrategyConfig)
    session_config: Dict[str, Any] = Field(default_factory=dict)
    symbols_config: Dict[str, Any] = Field(default_factory=dict)
    backtest: Dict[str, Any] = Field(default_factory=dict)
