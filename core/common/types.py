from pydantic import BaseModel, Field, ConfigDict
import numpy as np
from dataclasses import dataclass, fields, field
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
from datetime import datetime

class MarketRegime(Enum):
    TREND           = "TRENDING"
    RANGE           = "RANGING"
    UNCERTAIN       = "UNCERTAIN"

class VolatilityStatus(Enum):
    HIGH            = "HIGH"
    LOW             = "LOW"
    NORMAL          = "NORMAL"

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
    @property
    def spread_val(self): return self.spread

@dataclass(slots=True)
class CandleArray:
    """
    Vectorized container for OHLCVT+S candle data.
    Institutional V4-ULTRA Edition: Supports index-aware zero-copy views.
    """
    time: np.ndarray      # int64 timestamps
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    tick_volume: np.ndarray
    spread: np.ndarray     
    
    # Internal state for simulation fidelity
    _limit: Optional[int] = field(default=None, repr=False)
    indicators: Dict[str, np.ndarray] = field(default_factory=dict, repr=False)

    @property
    def limit(self) -> int:
        return self._limit if self._limit is not None else len(self.time)

    def set_limit(self, idx: int):
        """Restricts the view of the data to [0:idx]. Used for anti-lookahead simulation."""
        self._limit = idx

    @property
    def o(self) -> np.ndarray: return self.open[:self.limit]
    @property
    def h(self) -> np.ndarray: return self.high[:self.limit]
    @property
    def l(self) -> np.ndarray: return self.low[:self.limit]
    @property
    def c(self) -> np.ndarray: return self.close[:self.limit]
    @property
    def v(self) -> np.ndarray: return self.tick_volume[:self.limit]
    @property
    def s(self) -> np.ndarray: return self.spread[:self.limit]
    @property
    def t(self) -> np.ndarray: return self.time[:self.limit]

    def to_df(self) -> Any:
        import pandas as pd
        return pd.DataFrame({
            "time": self.time,
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "tick_volume": self.tick_volume,
            "spread": self.spread
        })

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
        return self.limit
    
    def slice(self, start: int, end: int) -> "CandleArray":
        """Returns a new CandleArray window. Preserves indicators (O(N) copy)."""
        new_indicators = {}
        for name, arr in self.indicators.items():
            if len(arr) >= end:
                new_indicators[name] = arr[start:end]
        
        return CandleArray(
            time=self.time[start:end],
            open=self.open[start:end],
            high=self.high[start:end],
            low=self.low[start:end],
            close=self.close[start:end],
            tick_volume=self.tick_volume[start:end],
            spread=self.spread[start:end],
            indicators=new_indicators
        )

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            start = idx.start if idx.start is not None else 0
            stop = idx.stop if idx.stop is not None else self.limit
            return self.slice(start, stop)
        elif isinstance(idx, int):
            if idx < 0: idx = self.limit + idx
            return Candle(
                time=self.time[idx],
                open=self.open[idx],
                high=self.high[idx],
                low=self.low[idx],
                close=self.close[idx],
                tick_volume=self.tick_volume[idx],
                spread=self.spread[idx]
            )
        elif isinstance(idx, (np.ndarray, list)):
            return CandleArray(
                time=self.time[idx],
                open=self.open[idx],
                high=self.high[idx],
                low=self.low[idx],
                close=self.close[idx],
                tick_volume=self.tick_volume[idx],
                spread=self.spread[idx]
            )
        raise TypeError(f"Invalid argument type: {type(idx)}")

    def get_indicator(self, name: str) -> np.ndarray:
        """Accesses pre-calculated indicators up to the current limit (O(1) view)."""
        if name not in self.indicators:
            return np.full(self.limit, np.nan)
        return self.indicators[name][:self.limit]

    # Legacy helper methods updated to use pre-calculation if available
    def ema(self, period: int) -> np.ndarray:
        key = f"ema_{period}"
        if key in self.indicators: return self.get_indicator(key)
        res = self._calc_ema(self.c, period)
        self.indicators[key] = res
        return res

    def rsi(self, period: int = 14) -> np.ndarray:
        key = f"rsi_{period}"
        if key in self.indicators: return self.get_indicator(key)
        res = self._calc_rsi(self.c, period)
        self.indicators[key] = res
        return res

    def atr(self, period: int = 14) -> np.ndarray:
        key = f"atr_{period}"
        if key in self.indicators: return self.get_indicator(key)
        res = self._calc_atr(period)
        self.indicators[key] = res
        return res

    def adx(self, period: int = 14) -> np.ndarray:
        key = f"adx_{period}"
        if key in self.indicators: return self.get_indicator(key)
        res = self._calc_adx(period)
        self.indicators[key] = res
        return res

    def _calc_ema(self, data: np.ndarray, period: int) -> np.ndarray:
        if len(data) < period: return np.full_like(data, np.nan)
        alpha = 2 / (period + 1)
        ema = np.zeros_like(data)
        ema[period-1] = np.mean(data[:period])
        for i in range(period, len(data)):
            ema[i] = (data[i] - ema[i-1]) * alpha + ema[i-1]
        return ema

    def _calc_rsi(self, data: np.ndarray, period: int) -> np.ndarray:
        if len(data) < period + 1: return np.full_like(data, np.nan)
        delta = np.diff(data)
        gain = (delta > 0) * delta
        loss = (delta < 0) * -delta
        avg_gain = np.full_like(data, np.nan)
        avg_loss = np.full_like(data, np.nan)
        avg_gain[period] = np.mean(gain[:period])
        avg_loss[period] = np.mean(loss[:period])
        alpha = 1 / period
        for i in range(period + 1, len(data)):
            avg_gain[i] = (gain[i-1] - avg_gain[i-1]) * alpha + avg_gain[i-1]
            avg_loss[i] = (loss[i-1] - avg_loss[i-1]) * alpha + avg_loss[i-1]
        # FIXED: Guard against zero loss to prevent ZeroDivisionError
        with np.errstate(divide='ignore', invalid='ignore'):
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
        return np.nan_to_num(rsi, nan=50.0) # Fallback to neutral 50

    def _calc_atr(self, period: int) -> np.ndarray:
        if self.limit < period + 1: return np.full(self.limit, np.nan)
        h, l, cp = self.high[1:self.limit], self.low[1:self.limit], self.close[:self.limit-1]
        tr = np.maximum(h - l, np.maximum(np.abs(h - cp), np.abs(l - cp)))
        atr = np.full(self.limit, np.nan)
        atr[period] = np.mean(tr[:period])
        alpha = 1.0 / period
        for i in range(period + 1, self.limit):
            atr[i] = (tr[i-1] - atr[i-1]) * alpha + atr[i-1]
        return atr

    def _calc_adx(self, period: int) -> np.ndarray:
        if self.limit < period * 2: return np.full(self.limit, np.nan)
        h, l, cp = self.high[1:self.limit], self.low[1:self.limit], self.close[:self.limit-1]
        
        tr = np.maximum(h - l, np.maximum(np.abs(h - cp), np.abs(l - cp)))
        up_move = h - self.high[:self.limit-1]
        down_move = self.low[:self.limit-1] - l
        
        pos_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        neg_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        
        alpha = 1.0 / period
        
        def wilders_smooth(data):
            smooth = np.full_like(data, np.nan)
            smooth[period-1] = np.mean(data[:period])
            for i in range(period, len(data)):
                smooth[i] = (data[i] - smooth[i-1]) * alpha + smooth[i-1]
            return smooth

        str_tr = wilders_smooth(tr)
        str_pdm = wilders_smooth(pos_dm)
        str_ndm = wilders_smooth(neg_dm)
        
        with np.errstate(divide='ignore', invalid='ignore'):
            plus_di = 100 * (str_pdm / str_tr)
            minus_di = 100 * (str_ndm / str_tr)
            # Guard against zero sum in DX calculation
            di_sum = plus_di + minus_di
            dx = 100 * np.abs(plus_di - minus_di) / di_sum
        
        plus_di = np.nan_to_num(plus_di)
        minus_di = np.nan_to_num(minus_di)
        dx = np.nan_to_num(dx)
        
        adx = np.full(self.limit, np.nan)
        # Shift DX to align with CandleArray indices (tr is size limit-1)
        adx_core = wilders_smooth(dx[period-1:])
        adx[period*2-1:] = adx_core[period-1:]
        return adx

    def bollinger_bands(self, period: int = 20, std_dev: float = 2.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """High-performance vectorized Bollinger Bands (anti-lookahead safe)."""
        data = self.c  # Use limited view to prevent lookahead bias
        if len(data) < period:
            nan_arr = np.full(len(data), np.nan)
            return nan_arr, nan_arr, nan_arr
            
        # Use Simple Moving Average
        sma = np.array([np.mean(data[i-period+1:i+1]) if i >= period-1 else np.nan for i in range(len(data))])
        
        # Vectorized Standard Deviation (Rolling)
        rolling_std = np.array([np.std(data[i-period+1:i+1]) if i >= period-1 else np.nan for i in range(len(data))])
        
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
    volume: float = 0.0  # Lot size (set by risk engine before execution)
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
