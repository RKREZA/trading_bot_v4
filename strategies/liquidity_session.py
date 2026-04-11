import numpy as np
import logging
from typing import Optional, Dict, Any
from core.base_strategy import BaseStrategy, MarketData
from core.common.types import TradeSignal
from core.session_detector import SessionDetector

logger = logging.getLogger("trading_bot.strategy.liquidity")


class LiquiditySessionStrategy(BaseStrategy):
    """
    V5 OPTIMIZED Liquidity Session Strategy.
    Trades breaks of the Asian/Tokyo range during London and NY sessions.
    Uses SessionDetector for proper session handling.
    """

    def __init__(self, strategy_id: str, config: dict):
        super().__init__(strategy_id, config)
        
        strat_config = self.config.get(strategy_id, self.config.get("LiquiditySession", {}))
        self.enabled = strat_config.get("enabled", True)
        
        self.range_maturity_limit = strat_config.get("range_maturity_limit", 3.0)
        self.vol_trigger_mult = strat_config.get("vol_trigger_mult", 0.4)
        self.sl_atr = strat_config.get("sl_atr", 1.5)
        self.tp_atr = strat_config.get("tp_atr", 4.0)
        self.lookback_bars = strat_config.get("lookback_bars", 60)
        self.min_range_bars = strat_config.get("min_range_bars", 20)
        self.min_confidence = strat_config.get("min_confidence", 0.65)
        
        self._asian_high = 0.0
        self._asian_low = 0.0
        self._range_set = False
        self._trade_today = False
        self._last_signal_bar = 0

    def generate_signal(self, market_data: MarketData) -> Optional[TradeSignal]:
        strat_config = self.config.get(self.strategy_id, self.config.get("LiquiditySession", {}))
        allowed_sessions = strat_config.get("allowed_sessions", [])
        if not SessionDetector.is_session_active(market_data.timestamp, allowed_sessions=allowed_sessions):
            self.last_rejection_reason = f"Out of Session ({market_data.session})"
            return None
        
        m5 = market_data.m5_candles
        if len(m5) < self.lookback_bars + 10:
            self.last_rejection_reason = "LiquiditySession: Insufficient data"
            return None
        
        bars_since_last = len(m5) - self._last_signal_bar
        if bars_since_last < 10:
            self.last_rejection_reason = "Signal cooldown active"
            return None
        
        price = market_data.current_price
        session = market_data.session
        
        atr_vals = m5.atr(14)
        if len(atr_vals) == 0:
            return None
        atr = atr_vals[-1]
        
        if np.isnan(atr) or atr <= 0:
            return None
        
        lookback = min(self.lookback_bars, len(m5) - 1)
        highs = m5.high[-lookback:]
        lows = m5.low[-lookback:]
        
        if session == "TOKYO" or (session == "GLOBAL" and market_data.timestamp.hour < 8):
            if not self._range_set:
                self._asian_high = price
                self._asian_low = price
                self._range_set = True
            else:
                self._asian_high = max(self._asian_high, price)
                self._asian_low = min(self._asian_low, price)
            self.last_rejection_reason = "LiquiditySession: Building Asian Range"
            return None
        
        if session not in ["LONDON", "LONDON/NY", "NEW_YORK"]:
            self.last_rejection_reason = f"LiquiditySession: Not active session ({session})"
            return None
        
        if not self._range_set:
            self.last_rejection_reason = "LiquiditySession: Range not set"
            return None
        
        if self._asian_high == self._asian_low:
            self.last_rejection_reason = "LiquiditySession: Invalid range"
            return None
        
        range_height = self._asian_high - self._asian_low
        if range_height > (atr * self.range_maturity_limit):
            self.last_rejection_reason = f"LiquiditySession: Range too wide ({range_height/atr:.1f}x ATR)"
            self._range_set = False
            return None
        
        breakout_buffer = atr * 0.05
        last_candle = m5[-1]
        candle_range = last_candle.high - last_candle.low
        
        is_volatile = candle_range > (atr * self.vol_trigger_mult)
        
        vol_sma = np.mean(m5.v[-20:]) if len(m5.v) >= 20 else 1
        is_high_volume = last_candle.tick_volume > (vol_sma * 1.1) if vol_sma > 0 else True
        
        if not (is_volatile or is_high_volume):
            self.last_rejection_reason = "LiquiditySession: Low momentum"
            return None
        
        self._last_signal_bar = len(m5)
        
        if price > (self._asian_high + breakout_buffer):
            confidence = 0.75 + (0.1 if is_volatile else 0) + (0.05 if is_high_volume else 0)
            return TradeSignal(direction="BUY", price=price, confidence=min(0.95, confidence))
            
        if price < (self._asian_low - breakout_buffer):
            confidence = 0.75 + (0.1 if is_volatile else 0) + (0.05 if is_high_volume else 0)
            return TradeSignal(direction="SELL", price=price, confidence=min(0.95, confidence))
        
        self.last_rejection_reason = "LiquiditySession: No breakout"
        return None

    def reset_daily_stats(self) -> None:
        self._asian_high = 0.0
        self._asian_low = 0.0
        self._range_set = False
        self._trade_today = False

    def get_stop_loss(self, signal: TradeSignal, market_data: MarketData) -> float:
        m5 = market_data.m5_candles
        atr_vals = m5.atr(14)
        atr = atr_vals[-1] if len(atr_vals) > 0 and not np.isnan(atr_vals[-1]) else 1.0
        
        if signal.direction == "BUY":
            return market_data.current_price - (atr * self.sl_atr)
        else:
            return market_data.current_price + (atr * self.sl_atr)

    def get_take_profit(self, signal: TradeSignal, market_data: MarketData) -> float:
        m5 = market_data.m5_candles
        atr_vals = m5.atr(14)
        atr = atr_vals[-1] if len(atr_vals) > 0 and not np.isnan(atr_vals[-1]) else 1.0
        
        if signal.direction == "BUY":
            return market_data.current_price + (atr * self.tp_atr)
        else:
            return market_data.current_price - (atr * self.tp_atr)

    def get_metrics(self, market_data: MarketData) -> Dict[str, Any]:
        return {
            "asian_high": self._asian_high,
            "asian_low": self._asian_low,
            "range_set": self._range_set,
            "session": market_data.session
        }

    def get_thresholds(self) -> Dict[str, str]:
        return {
            "Range Limit": f"< {self.range_maturity_limit} ATR",
            "Vol Trigger": f"> {self.vol_trigger_mult} ATR"
        }