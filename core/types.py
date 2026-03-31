from pydantic import BaseModel, Field
import numpy as np
from dataclasses import dataclass, fields
from typing import Dict, List, Optional

@dataclass(slots=True)
class CandleArray:
    """Vectorized candle data for fast indicator computation."""
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
        """Slice the arrays similarly to Python lists."""
        return CandleArray(*(getattr(self, f.name)[start:end] for f in fields(self)))

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            start = idx.start if idx.start is not None else 0
            stop = idx.stop if idx.stop is not None else len(self.time)
            return self.slice(start, stop)
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
    min_confluence_score: int = 4
    min_confidence: int = 60
    cooldown_candles: int = 20
    min_sl_points: int = 150
    # Additional keys mapping strategy specific config

from pydantic import BaseModel, Field, ConfigDict

class BotConfig(BaseModel):
    model_config = ConfigDict(extra='allow')
    
    symbol: str = "XAUUSDm"
    magic_number: int = 234000
    risk: RiskConfig = Field(default_factory=RiskConfig)
    strategy_defaults: StrategyConfig = Field(default_factory=StrategyConfig)
    session_config: Dict[str, dict] = Field(default_factory=dict)
    symbols_config: Dict[str, dict] = Field(default_factory=dict)
    backtest: Dict[str, dict] = Field(default_factory=dict)
