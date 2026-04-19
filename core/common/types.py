from pydantic import BaseModel, Field, ConfigDict
import numpy as np
from dataclasses import dataclass, fields, field
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
from datetime import datetime
import hashlib
import json
from types import MappingProxyType
import math

class CanonicalHasher:
    """
    V5-INSIGNIA Bit-Level Canonical Serialization Engine.
    Ensures cross-platform hash stability via string-canonicalization.
    """
    @staticmethod
    def normalize_float(x: Any) -> str:
        if not isinstance(x, (float, int, np.floating, np.integer)):
            return str(x)
        fx = float(x)
        if math.isnan(fx): return "NaN"
        if math.isinf(fx): return "Inf" if fx > 0 else "-Inf"
        # Institutional Rule 1.1: Normalize -0.0 and enforce .10f
        if fx == 0.0: return "0.0000000000"
        return format(fx, ".10f")

    @staticmethod
    def canonicalize(data: Any) -> Any:
        """Recursively normalizes data for bit-level JSON stability."""
        if isinstance(data, dict):
            return {str(k): CanonicalHasher.canonicalize(v) for k, v in sorted(data.items())}
        if isinstance(data, (list, tuple)):
            return [CanonicalHasher.canonicalize(v) for v in data]
        if isinstance(data, (float, int, np.floating, np.integer)):
            return CanonicalHasher.normalize_float(data)
        return str(data)

    @staticmethod
    def get_hash(domain: str, data: Dict[str, Any]) -> str:
        """Generates a domain-separated SHA256 hash from canonical JSON."""
        # Rule 1.2: Deterministic JSON (No whitespace, sorted keys)
        canonical_data = CanonicalHasher.canonicalize(data)
        serialized = json.dumps(canonical_data, sort_keys=True, separators=(",", ":"))
        # Rule 1.4: UTF-8 encoding lock
        payload = f"{domain}|{serialized}".encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

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
    Institutional V5-INSIGNIA Edition: Supports index-aware zero-copy views.
    """
    time: np.ndarray      # int64 timestamps
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    tick_volume: np.ndarray
    spread: np.ndarray     
    
    def __post_init__(self):
        """ المؤسسة (Institutional): Enforce immutable arrays. """
        for f in fields(self):
            attr = getattr(self, f.name)
            if isinstance(attr, np.ndarray):
                attr.setflags(write=False)
    
    # Internal state for simulation fidelity
    _limit: Optional[int] = field(default=None, repr=False)
    _indicators: Dict[str, np.ndarray] = field(default_factory=dict, repr=False)

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
        for name, arr in self._indicators.items():
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
            _indicators=new_indicators
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
        if name not in self._indicators:
            return np.full(self.limit, np.nan)
        return self._indicators[name][:self.limit]

    # Legacy helper methods updated to use pre-calculation if available
    def ema(self, period: int) -> np.ndarray:
        key = f"ema_{period}"
        if key in self._indicators: return self.get_indicator(key)
        res = self._calc_ema(self.c, period)
        self._indicators[key] = res
        return res

    def rsi(self, period: int = 14) -> np.ndarray:
        key = f"rsi_{period}"
        if key in self._indicators: return self.get_indicator(key)
        res = self._calc_rsi(self.c, period)
        self._indicators[key] = res
        return res

    def atr(self, period: int = 14) -> np.ndarray:
        key = f"atr_{period}"
        if key in self._indicators: return self.get_indicator(key)
        res = self._calc_atr(period)
        self._indicators[key] = res
        return res

    def adx(self, period: int = 14) -> np.ndarray:
        key = f"adx_{period}"
        if key in self._indicators: return self.get_indicator(key)
        res = self._calc_adx(period)
        self._indicators[key] = res
        return res

    def _calc_ema(self, data: np.ndarray, period: int) -> np.ndarray:
        if len(data) < period: return np.full_like(data, np.nan)
        import pandas as pd
        s = pd.Series(data)
        out = s.copy()
        out.iloc[:period-1] = np.nan
        out.iloc[period-1] = s.iloc[:period].mean()
        return out.ewm(span=period, adjust=False).mean().values

    def _calc_rsi(self, data: np.ndarray, period: int) -> np.ndarray:
        if len(data) < period + 1: return np.full_like(data, np.nan)
        import pandas as pd
        delta = np.diff(data)
        gain = (delta > 0) * delta
        loss = (delta < 0) * -delta
        
        alpha = 1 / period
        g_s = pd.Series(gain)
        l_s = pd.Series(loss)
        
        g_mod = g_s.copy()
        g_mod.iloc[:period-1] = np.nan
        g_mod.iloc[period-1] = g_s.iloc[:period].mean()
        
        l_mod = l_s.copy()
        l_mod.iloc[:period-1] = np.nan
        l_mod.iloc[period-1] = l_s.iloc[:period].mean()
        
        avg_gain = g_mod.ewm(alpha=alpha, adjust=False).mean().values
        avg_loss = l_mod.ewm(alpha=alpha, adjust=False).mean().values

        with np.errstate(divide='ignore', invalid='ignore'):
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            
        rsi = np.nan_to_num(rsi, nan=50.0)
        out_rsi = np.full_like(data, np.nan)
        out_rsi[1:] = rsi
        return out_rsi

    def _calc_atr(self, period: int) -> np.ndarray:
        if self.limit < period + 1: return np.full(self.limit, np.nan)
        import pandas as pd
        h, l, cp = self.h[1:], self.l[1:], self.c[:-1]
        tr = np.maximum(h - l, np.maximum(np.abs(h - cp), np.abs(l - cp)))
        
        tr_s = pd.Series(tr)
        tr_mod = tr_s.copy()
        tr_mod.iloc[:period-1] = np.nan
        tr_mod.iloc[period-1] = tr_s.iloc[:period].mean()
        
        atr = tr_mod.ewm(alpha=1.0/period, adjust=False).mean().values
        
        out_atr = np.full(self.limit, np.nan)
        out_atr[1:] = atr
        return out_atr

    def _calc_adx(self, period: int) -> np.ndarray:
        if self.limit < period * 2: return np.full(self.limit, np.nan)
        # Institutional Standard: True Range needs |high - prev_close|
        h, l, cp = self.h[1:], self.l[1:], self.c[:-1]
        
        tr = np.maximum(h - l, np.maximum(np.abs(h - cp), np.abs(l - cp)))
        up_move = h - self.h[:-1]
        down_move = self.l[:-1] - l
        
        pos_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        neg_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        
        alpha = 1.0 / period
        
        import pandas as pd
        def wilders_smooth(data_arr):
            s = pd.Series(data_arr)
            mod = s.copy()
            mod.iloc[:period-1] = np.nan
            mod.iloc[period-1] = s.iloc[:period].mean()
            return mod.ewm(alpha=alpha, adjust=False).mean().values

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
            
        # Institutional Vectorization (Step 10 Optimization)
        # Using a rolling window via stride_tricks or pandas if available (fallback to cumsum)
        try:
            import pandas as pd
            series = pd.Series(data)
            sma = series.rolling(window=period).mean().values
            rolling_std = series.rolling(window=period).std(ddof=0).values
        except ImportError:
            # Native NumPy vectorized approach
            sma = np.convolve(data, np.ones(period), 'valid') / period
            sma = np.concatenate([np.full(period-1, np.nan), sma])
            
            # Variance = E[X^2] - (E[X])^2
            data_sq = data**2
            sma_sq = np.convolve(data_sq, np.ones(period), 'valid') / period
            sma_sq = np.concatenate([np.full(period-1, np.nan), sma_sq])
            
            rolling_std = np.sqrt(np.maximum(0, sma_sq - sma**2))
        
        upper = sma + (rolling_std * std_dev)
        lower = sma - (rolling_std * std_dev)
        return upper, lower, sma

@dataclass(frozen=True, slots=True)
class ExecutionIntent:
    """
    Institutional V5-INSIGNIA Execution Intent.
    IMMUTABLE. Any mutation post-creation is an integrity violation.
    """
    symbol: str
    direction: str
    volume: float
    stop_loss: float
    take_profit: float
    strategy_id: str
    setup_timestamp: float
    
    @property
    def intent_hash(self) -> str:
        """Rule 1.1: Bit-Level Canonical Fingerprint."""
        data = {
            "symbol": self.symbol,
            "direction": self.direction,
            "volume": self.volume,
            "sl": self.stop_loss,
            "tp": self.take_profit,
            "sid": self.strategy_id,
            "t": self.setup_timestamp
        }
        return CanonicalHasher.get_hash("INTENT", data)

@dataclass(frozen=True, slots=True)
class MarketSnapshot:
    """
    Institutional V5-INSIGNIA Market Snapshot.
    FROZEN at the moment of intent processing.
    """
    timestamp: float
    bid: float
    ask: float
    spread: float
    point: float
    dfs: float  # Data Fidelity Score
    volatility: str # HIGH/LOW/NORMAL
    metadata: MappingProxyType = field(default_factory=lambda: MappingProxyType({}))

    @property
    def snapshot_id(self) -> str:
        """Rule 1.1: Domain-Separated Snapshot Hash."""
        data = {
            "t": self.timestamp,
            "bid": self.bid,
            "ask": self.ask,
            "spread": self.spread,
            "point": self.point,
            "dfs": self.dfs,
            "vol": self.volatility,
            "meta": dict(self.metadata)
        }
        return CanonicalHasher.get_hash("SNAPSHOT", data)

@dataclass(frozen=True, slots=True)
class ExecutionOutcome:
    """
    Institutional V5-INSIGNIA Execution Outcome.
    Includes friction decomposition and audit trail.
    """
    ticket: int
    fill_price: float
    actual_slippage_pips: float
    actual_latency_ms: float
    alpha_price: float       # Ideal entry price (no friction)
    microstructure_loss: float
    execution_drag: float
    timestamp: float
    intent_hash: str
    is_error: bool = False
    error_msg: str = ""

@dataclass(frozen=True)
class FilteredSignal:
    """
    Immutable safe DTO returned by AI predictor shielding original evaluation logic.
    """
    original: 'TradeSignal'
    approved: bool
    confidence: float
    comment: str = ""

@dataclass(frozen=True)
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
    vol_ratio: float = 1.0

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
