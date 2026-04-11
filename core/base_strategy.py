"""
TRADING BOT V4 — Strategy Abstraction Layer
===========================================
Defines the mandatory contract for all institutional strategy implementations.
"""

import uuid
import logging
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

# Institutional types
from .common.types import TradeSignal, CandleArray, MarketRegime

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
    current_price: float # Deprecated: Use bid/ask for precision
    bid: float
    ask: float
    spread: float
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
        self.enabled = True # Default state; subclasses should override from their config block
        self.last_rejection_reason = ""
        
        # Institutional Gating Attributes
        self.min_confidence = float(config.get("min_confidence", 0.5))
        self.min_rr = float(config.get("min_rr", 2.0))

    @abstractmethod
    def generate_signal(self, market_data: MarketData) -> Optional[TradeSignal]:
        """
        MUST return a TradeSignal object with direction (BUY|SELL|NONE), 
        confidence (0.0 to 1.0), and timestamp.
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

    @abstractmethod
    def get_metrics(self, market_data: MarketData) -> Dict[str, Any]:
        """Returns live metrics used for strategy decision making."""
        ...

    def get_thresholds(self) -> Dict[str, Any]:
        """Returns target thresholds from strategy configuration."""
        return self.config

    def on_trade_closed(self, trade_record: dict) -> None:
        pass

    def reset_daily_stats(self) -> None:
        pass

    # [ State Management Interface ] - Step 3.1
    def get_state(self) -> Dict[str, Any]:
        """Returns a JSON-serializable snapshot of the strategy state."""
        return {
            "strategy_id": self.strategy_id,
            "enabled": self.enabled,
            "config": self.config
        }

    def set_state(self, state: Dict[str, Any]) -> None:
        """Restores the strategy state from a snapshot."""
        self.enabled = state.get("enabled", True)
        self.config.update(state.get("config", {}))

    def get_regime_scaler(self, market_data: MarketData) -> float:
        """
        Institutional Volatility Adaptive Scaler.
        Returns a multiplier (1.0 to 2.0) based on the current ATR relative to history.
        Used to expand stops during 'Extreme' regimes to avoid being wicked out.
        """
        m5 = market_data.m5_candles
        atr_14 = m5.get_indicator("atr_14")
        if len(atr_14) < 100:
            return 1.0
            
        current_atr = atr_14[-1]
        avg_atr_100 = np.mean(atr_14[-100:])
        
        if avg_atr_100 == 0 or np.isnan(current_atr) or np.isnan(avg_atr_100):
            return 1.0
            
        vol_ratio = current_atr / avg_atr_100
        
        # Scaling logic:
        # LOW/NORMAL (<1.2x): 1.0x (No change)
        # HIGH (1.2x - 1.5x): 1.25x (Moderate expansion)
        # EXTREME (>1.5x): 1.5x to 2.0x (Max safety expansion)
        if vol_ratio <= 1.2:
            return 1.0
        elif vol_ratio <= 1.5:
            return 1.25
        else:
            return min(2.0, 1.0 + (vol_ratio * 0.4))

    def is_spread_safe(self, market_data: MarketData) -> bool:
        """
        Institutional Liquidity Guard.
        Rejects entries if the spread exceeds the safety ceiling to prevent 'Death by Slippage'.
        """
        # Load from strategy-specific or global config
        max_spread_points = self.config.get("max_spread_points", 50)
        
        # Point Normalization: market_data.spread is absolute price
        # We assume 1 pip = 10 points (Institutional Standard)
        # But here we just want points: spread / point
        # Using a tiny epsilon to avoid div by zero
        current_spread_points = market_data.spread / 0.0001 if market_data.spread < 1.0 else market_data.spread / 0.01
        
        # Actually safer way: just use a standard point mapper if unknown
        # Better: let's use the point passed in backtest if available
        # FOR NOW: We assume XAUUSD point is 0.01 (2 dec) or 0.001 (3 dec)
        # We will use 0.01 as the base for 'Points' in XAUUSD
        point_val = 0.01 
        current_points = market_data.spread / point_val if point_val > 0 else 0
        
        if current_points > max_spread_points:
            self.last_rejection_reason = f"Spread Gated: {current_points:.1f} pts > {max_spread_points}"
            return False
        return True

    def get_session_confidence_floor(self, market_data: MarketData) -> float:
        """
        Institutional Time-Aware Gating.
        Increases the 'min_confidence' barrier during high-risk windows (London Stop Hunt).
        """
        session = market_data.session
        base_floor = float(self.config.get("min_confidence", 0.70))
        
        # LONDON Stop Hunt Window: Require A Setups (08:00 - 13:00 UTC)
        if session == "LONDON":
            return max(0.80, base_floor + 0.05)
            
        return base_floor

    def get_ema_trend(self, candles: CandleArray, fast: int = 50, slow: int = 200) -> int:
        """
        Returns institutional Trend State:
        1: Bullish (Fast > Slow)
        -1: Bearish (Fast < Slow)
        0: Neutral/Crossing or NaN
        """
        ema_fast = candles.ema(fast)
        ema_slow = candles.ema(slow)
        
        # Institutional Gating: Use pre-calculation validity instead of length check
        if len(ema_fast) == 0 or len(ema_slow) == 0:
            self.last_rejection_reason = "EMA: No data"
            return 0
            
        f_val, s_val = ema_fast[-1], ema_slow[-1]
        
        if np.isnan(f_val) or np.isnan(s_val):
            self.last_rejection_reason = "EMA: NaN"
            return 0
            
        if f_val > s_val:
            return 1
        elif f_val < s_val:
            return -1
        
        self.last_rejection_reason = "EMA: Neutral"
        return 0

    def check_mtf_consensus(self, market_data: MarketData) -> bool:
        """
        Verifies if both H1 (HTF) and M15 trends are in agreement.
        Essential for institutional trend following.
        """
        h1_trend = self.get_ema_trend(market_data.htf_candles)
        m15_trend = self.get_ema_trend(market_data.m15_candles)
        
        return h1_trend == m15_trend and h1_trend != 0

    def is_symbol_allowed(self, symbol: str) -> bool:
        include = self.config.get("symbols") or self.config.get("include_symbols") or []
        exclude = self.config.get("exclude_symbols") or []

        try:
            include_set = {str(s).upper() for s in include}
            exclude_set = {str(s).upper() for s in exclude}
        except Exception:
            include_set = set()
            exclude_set = set()

        sym = str(symbol).upper()
        if include_set and sym not in include_set:
            return False
        if exclude_set and sym in exclude_set:
            return False
        return True

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}(id={self.strategy_id}, enabled={self.enabled})>"
