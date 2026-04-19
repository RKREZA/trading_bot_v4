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
    current_price: float  # Deprecated: Use bid/ask for precision
    bid: float
    ask: float
    spread: float
    point: float
    session: str
    timestamp: datetime
    m1_candles: Optional[CandleArray] = None  # M1 data for scalping strategies
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
        
        # [ Institutional Config Resolution ]: Resolve Strategy-Specific Block
        strat_cfg = self.get_strat_config()
        self.enabled = strat_cfg.get("enabled", True)
        self.last_rejection_reason = ""
        
        # Institutional Gating Attributes (Resolved with fallbacks)
        self.min_confidence = float(strat_cfg.get("min_confidence", config.get("risk_governance", {}).get("min_confidence", 0.5)))
        self.min_rr = float(strat_cfg.get("min_rr", config.get("risk_governance", {}).get("min_rr", 2.0)))

    def get_strat_config(self) -> Dict[str, Any]:
        """Ensures consistent access to the strategy-specific configuration block regardless of nesting."""
        # Strip version suffix for config lookup
        base_name = self.strategy_id.rsplit('_v', 1)[0] if '_v' in self.strategy_id else self.strategy_id
        
        # Check 'strategies' key first (Standard V5 structure)
        strategies_block = self.config.get("strategies", {})
        if self.strategy_id in strategies_block:
            return strategies_block[self.strategy_id]
        if base_name in strategies_block:
            return strategies_block[base_name]
            
        # Fallback 1: Direct key at root (Legacy/Diagnostic support)
        if self.strategy_id in self.config:
            return self.config[self.strategy_id]
        if base_name in self.config:
            return self.config[base_name]
            
        # Fallback 2: Case-insensitive search
        for key in self.config:
            if key.lower() == self.strategy_id.lower() or key.lower() == base_name.lower():
                return self.config[key]
                
        return {}

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

    def get_parameter_grid(self) -> Dict[str, List[Any]]:
        """
        Returns the hyperparameter optimization boundaries for Walk-Forward Optimization.
        Format: {'metric_name': [val1, val2, val3]}
        Override this in children to enable Strategy Architecture WFO tuning.
        """
        return {}

    def get_thresholds(self) -> Dict[str, Any]:
        """Returns curated target thresholds from strategy configuration for the dashboard."""
        strat_cfg = self.get_strat_config()
        targets = {}
        
        # 1. Base Institutional Targets
        targets["min_conf"] = self.min_confidence
        targets["min_rr"] = self.min_rr
        
        # 2. Dynamic Discovery: Find parameters ending in _oversold, _threshold, _mult, etc.
        dashboard_keywords = [
            "oversold", "overbought", "threshold", "std", "period", "atr", 
            "max_vol", "max_slope", "ratio", "sessions"
        ]
        
        for k, v in strat_cfg.items():
            if any(key in k.lower() for key in dashboard_keywords):
                targets[k] = v
                
        return targets

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
        Rejects entries if the spread exceeds a dynamic volatility-based threshold.
        Rule: Spread cost should be < 15% of the current M15 ATR.
        """
        # 1. Point Normalization (Audit PASS #6 Fix)
        # Use the actual symbol point value from MarketData
        point_val = market_data.point
        current_points = market_data.spread / point_val if point_val > 0 else 0
        
        # 2. Static Safety Ceiling (Hard Max)
        max_points = self.config.get("max_spread_points", 100)
        if current_points > max_points:
            self.last_rejection_reason = f"Spread Gated (Hard Cap): {current_points:.1f} pts > {max_points}"
            return False
            
        # 3. Dynamic Cost Guard (A+ Requirement)
        # We ensure that the 'toll' paid to enter the trade doesn't eat too much of the target.
        m15_atr = market_data.m15_candles.get_indicator("atr_14")
        if len(m15_atr) > 0:
            current_atr = m15_atr[-1]
            if current_atr > 0:
                cost_ratio = market_data.spread / current_atr
                max_ratio = self.config.get("max_spread_atr_ratio", 0.15)
                
                if cost_ratio > max_ratio:
                    self.last_rejection_reason = f"Spread Gated (ATR): Cost Ratio {cost_ratio:.1%} > {max_ratio:.1%}"
                    return False
        
        # 4. Institutional Tick Density Guard (A+ Scale-Hardening)
        # Prevent trading in 'thin' markets where bid-ask spreads are volatile.
        # Threshold: Min 45 ticks per minute (675 per M15 bar) for institutional liquidity.
        min_density = self.config.get("risk_governance", {}).get("min_tick_density", 45)
        m15 = market_data.m15_candles
        if len(m15) > 0:
            last_vol = m15.v[-1]
            ticks_per_min = last_vol / 15.0
            if ticks_per_min < min_density:
                self.last_rejection_reason = f"Liquidity Gated: Density {ticks_per_min:.1f} < {min_density} ticks/min"
                return False
                    
        return True

    def is_volatility_safe(self, market_data: MarketData) -> bool:
        """
        Institutional Volatility Guard.
        Rejects entries if the ATR is too small relative to the spread (low profit potential per unit of cost).
        """
        min_ratio = self.config.get("min_atr_spread_ratio", 3.5)
        
        # Use M15 ATR for entry gating
        m15_atr = market_data.m15_candles.get_indicator("atr_14")
        if len(m15_atr) == 0:
            return True # Default to pass if no data
            
        current_atr = m15_atr[-1]
        
        if market_data.spread <= 0:
            return True
            
        ratio = current_atr / market_data.spread
        
        if ratio < min_ratio:
            self.last_rejection_reason = f"Volatility Gated: ATR/Spread ratio {ratio:.2f} < {min_ratio}"
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
