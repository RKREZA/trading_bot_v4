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
import pandas as pd

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
    LIQUIDITY_EVENT = "LIQUIDITY_EVENT"
    EXPANSION       = "EXPANSION"
    TRANSITION      = "TRANSITION"

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
        s = pd.Series(data)
        out = s.copy()
        out.iloc[:period-1] = np.nan
        out.iloc[period-1] = s.iloc[:period].mean()
        return out.ewm(span=period, adjust=False).mean().values

    def _calc_rsi(self, data: np.ndarray, period: int) -> np.ndarray:
        if len(data) < period + 1: return np.full_like(data, np.nan)
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
        # Institutional Optimize: Check IPC cache for standard settings
        if period == 20 and std_dev == 2.0:
            if "bb_upper" in self._indicators and "bb_lower" in self._indicators:
                return self.get_indicator("bb_upper"), self.get_indicator("bb_lower"), self.get_indicator("bb_mid")

        data = self.c  # Use limited view to prevent lookahead bias
        if len(data) < period:
            nan_arr = np.full(len(data), np.nan)
            return nan_arr, nan_arr, nan_arr
            
        # Institutional Vectorization (Step 10 Optimization)
        # Using a rolling window via stride_tricks or pandas if available (fallback to cumsum)
        try:
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

    def vwap(self, session_bars: int = 96) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Session VWAP with ±1σ and ±2σ Standard Deviation Bands.
        Returns: (vwap, upper_1sd, lower_1sd, upper_2sd, lower_2sd)
        
        Args:
            session_bars: Number of bars per session (96 for M15 = 24h, 288 for M5 = 24h).
        """
        key = f"vwap_{session_bars}"
        if key in self._indicators:
            vw = self.get_indicator(key)
            return vw, self.get_indicator(f"{key}_u1"), self.get_indicator(f"{key}_l1"), self.get_indicator(f"{key}_u2"), self.get_indicator(f"{key}_l2")

        n = self.limit
        if n < 2:
            nan_arr = np.full(n, np.nan)
            return nan_arr, nan_arr, nan_arr, nan_arr, nan_arr

        typical = (self.h + self.l + self.c) / 3.0
        vol = self.v.astype(np.float64)
        # Prevent zero-volume bars from breaking calculation
        vol = np.where(vol == 0, 1.0, vol)

        vwap_arr = np.full(n, np.nan)
        u1 = np.full(n, np.nan)
        l1 = np.full(n, np.nan)
        u2 = np.full(n, np.nan)
        l2 = np.full(n, np.nan)

        # Rolling session-anchored VWAP
        for i in range(n):
            # Session start index
            start = max(0, i - session_bars + 1)
            seg_tp = typical[start:i+1]
            seg_vol = vol[start:i+1]
            cum_vol = np.sum(seg_vol)
            if cum_vol == 0:
                continue
            vw = np.sum(seg_tp * seg_vol) / cum_vol
            vwap_arr[i] = vw
            # Variance = Σ(vol * (tp - vwap)^2) / Σ(vol)
            variance = np.sum(seg_vol * (seg_tp - vw) ** 2) / cum_vol
            sd = np.sqrt(variance)
            u1[i] = vw + sd
            l1[i] = vw - sd
            u2[i] = vw + 2 * sd
            l2[i] = vw - 2 * sd

        self._indicators[key] = vwap_arr
        self._indicators[f"{key}_u1"] = u1
        self._indicators[f"{key}_l1"] = l1
        self._indicators[f"{key}_u2"] = u2
        self._indicators[f"{key}_l2"] = l2
        return vwap_arr, u1, l1, u2, l2

    def volume_profile(self, lookback: int = 20, bins: int = 50) -> Dict:
        """
        Fixed Range Volume Profile over the last `lookback` bars.
        Returns dict with: poc (float), vah (float), val (float), profile (list of (price, volume)).
        
        POC = Point of Control (highest volume price level)
        VAH = Value Area High (upper boundary of 70% volume concentration)
        VAL = Value Area Low (lower boundary of 70% volume concentration)
        """
        key = f"vp_{lookback}_{bins}"
        if key in self._indicators:
            cached = self._indicators[key]
            return cached

        n = self.limit
        if n < lookback:
            result = {"poc": np.nan, "vah": np.nan, "val": np.nan, "profile": []}
            self._indicators[key] = result
            return result

        start = n - lookback
        highs = self.h[start:n]
        lows = self.l[start:n]
        closes = self.c[start:n]
        volumes = self.v[start:n].astype(np.float64)

        price_high = np.max(highs)
        price_low = np.min(lows)
        price_range = price_high - price_low

        if price_range <= 0 or np.sum(volumes) == 0:
            result = {"poc": np.nan, "vah": np.nan, "val": np.nan, "profile": []}
            self._indicators[key] = result
            return result

        # Build volume distribution across price bins
        bin_edges = np.linspace(price_low, price_high, bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
        vol_dist = np.zeros(bins, dtype=np.float64)

        for j in range(lookback):
            # Distribute each bar's volume across bins it touches
            bar_low = lows[j]
            bar_high = highs[j]
            bar_vol = volumes[j]
            if bar_vol == 0:
                bar_vol = 1.0
            for b in range(bins):
                # Check overlap between bar range and bin range
                overlap_low = max(bar_low, bin_edges[b])
                overlap_high = min(bar_high, bin_edges[b + 1])
                if overlap_high > overlap_low:
                    bar_range = bar_high - bar_low
                    if bar_range > 0:
                        proportion = (overlap_high - overlap_low) / bar_range
                    else:
                        proportion = 1.0 / bins
                    vol_dist[b] += bar_vol * proportion

        # POC: bin with highest volume
        poc_idx = np.argmax(vol_dist)
        poc = float(bin_centers[poc_idx])

        # Value Area: 70% of total volume centered around POC
        total_vol = np.sum(vol_dist)
        target_vol = total_vol * 0.70
        va_vol = vol_dist[poc_idx]
        lo_idx = poc_idx
        hi_idx = poc_idx

        while va_vol < target_vol and (lo_idx > 0 or hi_idx < bins - 1):
            expand_up = vol_dist[hi_idx + 1] if hi_idx < bins - 1 else 0
            expand_down = vol_dist[lo_idx - 1] if lo_idx > 0 else 0
            if expand_up >= expand_down and hi_idx < bins - 1:
                hi_idx += 1
                va_vol += vol_dist[hi_idx]
            elif lo_idx > 0:
                lo_idx -= 1
                va_vol += vol_dist[lo_idx]
            else:
                hi_idx += 1
                va_vol += vol_dist[hi_idx]

        vah = float(bin_centers[hi_idx])
        val = float(bin_centers[lo_idx])

        profile = [(float(bin_centers[b]), float(vol_dist[b])) for b in range(bins)]

        result = {"poc": poc, "vah": vah, "val": val, "profile": profile}
        self._indicators[key] = result
        return result

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

class OrderState(Enum):
    PENDING = "PENDING"
    SENT = "SENT"
    FILLED = "FILLED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"
    CANCELLED = "CANCELLED"


class TradeSignal(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    direction: str  # BUY | SELL | NONE
    symbol: str = ""
    price: float = 0.0
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    timestamp: Optional[datetime] = None
    stop_loss: float = 0.0
    take_profit: float = 0.0
    volume: float = 0.0
    session: str = "GLOBAL"
    tp1_price: float = 0.0
    tp2_price: float = 0.0
    confluence_score: int = 0
    reasons: List[str] = Field(default_factory=list)
    rr_ratio: float = 2.0
    vol_ratio: float = 1.0
    execution_id: str = ""
    strategy_id: str = ""
    metadata: Dict[str, Any] = Field(default_factory=dict)


@dataclass(frozen=True)
class FilteredSignal:
    original: TradeSignal
    approved: bool
    confidence: float
    comment: str = ""


class OrderEvent(BaseModel):
    model_config = ConfigDict(frozen=True)

    execution_id: str
    symbol: str
    direction: str
    state: OrderState
    volume: float = 0.0
    price: float = 0.0
    sl: float = 0.0
    tp: float = 0.0
    strategy_id: str = ""
    timestamp: datetime = Field(default_factory=lambda: datetime.now())
    reason: str = ""


class ExecutionReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    execution_id: str
    symbol: str
    direction: str
    ticket: int = 0
    fill_price: float = 0.0
    slippage_pips: float = 0.0
    latency_ms: float = 0.0
    spread_at_entry: float = 0.0
    strategy_id: str = ""
    timestamp: datetime = Field(default_factory=lambda: datetime.now())
    intent_hash: str = ""
    is_error: bool = False
    error_msg: str = ""


class RiskAlert(BaseModel):
    model_config = ConfigDict(frozen=True)

    alert_type: str  # KILL_SWITCH | MAX_DRAWDOWN | DAILY_LOSS | EXPOSURE
    severity: str = "WARNING"  # WARNING | CRITICAL
    message: str = ""
    current_value: float = 0.0
    threshold: float = 0.0
    timestamp: datetime = Field(default_factory=lambda: datetime.now())


class BacktestResult(BaseModel):
    model_config = ConfigDict(extra="allow")

    status: str = "PENDING"  # PENDING | RUNNING | COMPLETED | FAILED
    initial_balance: float = 0.0
    final_balance: float = 0.0
    net_profit: float = 0.0
    total_trades: int = 0
    win_rate: float = 0.0
    max_drawdown: float = 0.0
    profit_factor: float = 0.0
    sharpe_ratio: float = 0.0
    equity_curve: List[float] = Field(default_factory=list)
    monte_carlo: Dict[str, Any] = Field(default_factory=dict)
    trades: List[Dict[str, Any]] = Field(default_factory=list)
    config_snapshot: Dict[str, Any] = Field(default_factory=dict)
