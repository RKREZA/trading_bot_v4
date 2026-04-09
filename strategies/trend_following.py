import numpy as np
import logging
from typing import Optional, Dict, Any
from core.base_strategy import BaseStrategy, MarketData
from core.common.types import TradeSignal

logger = logging.getLogger("trading_bot.strategy.trend")

class TrendFollowingStrategy(BaseStrategy):
    """
    V4 Institutional Trend Following (Dynamic Parameter Edition).
    Exposes all core variables to the Walk-Forward Optimizer and Config.
    """

    def __init__(self, strategy_id: str, config: dict):
        super().__init__(strategy_id, config)
        
        # Pull parameters dynamically from config (or use robust Gold defaults)
        # Priority: strategy_id -> "TrendFollowing"
        strat_config = self.config.get(strategy_id, self.config.get("TrendFollowing", {}))
        self.enabled = strat_config.get("enabled", True)
        
        self.ema_fast = strat_config.get("ema_fast", 50)
        self.ema_slow = strat_config.get("ema_slow", 200)
        self.adx_period = 14
        self.adx_threshold = strat_config.get("adx_threshold", 20) 
        self.vol_exclusion_mult = strat_config.get("vol_exclusion_mult", 3.0)
        
        # Risk Parameters
        self.sl_atr = strat_config.get("sl_atr", 2.5) 
        self.rr_target = strat_config.get("rr_target", 1.5) 

    def generate_signal(self, market_data: MarketData) -> Optional[TradeSignal]:
        strat_config = self.config.get(self.strategy_id, self.config.get("TrendFollowing", {}))
        allowed_sessions = strat_config.get("allowed_sessions", [])
        
        if allowed_sessions and market_data.session not in allowed_sessions:
            self.last_rejection_reason = f"Out of Session ({market_data.session})"
            return None

        m15_trend = self.get_ema_trend(market_data.m15_candles)
        if m15_trend == 0:
            return None

        m5 = market_data.m5_candles
        if m5 is None or len(m5) < self.ema_slow:
            return None

        ema_fast_vals = m5.ema(self.ema_fast)
        ema_slow_vals = m5.ema(self.ema_slow)
        if len(ema_fast_vals) == 0 or len(ema_slow_vals) == 0:
             return None
             
        current_fast = ema_fast_vals[-1]
        current_slow = ema_slow_vals[-1]

        m5_trend = 1 if current_fast > current_slow else -1
        if m5_trend != m15_trend:
            self.last_rejection_reason = "M5 against M15 Trend"
            return None

        adx_vals = m5.adx(self.adx_period)
        if len(adx_vals) == 0 or adx_vals[-1] < self.adx_threshold:
            self.last_rejection_reason = f"ADX too low (<{self.adx_threshold})"
            return None

        atr_vals = m5.atr(self.adx_period)
        if len(atr_vals) == 0: return None
        
        current_atr = atr_vals[-1]
        avg_atr = np.mean(atr_vals[-14:]) 
        
        if current_atr > (avg_atr * self.vol_exclusion_mult):
            self.last_rejection_reason = "Volatility Spike Detected"
            return None

        current_price = market_data.current_price
        direction = "BUY" if m5_trend == 1 else "SELL"
        
        signal = TradeSignal(direction=direction, confidence=min(1.0, adx_vals[-1] / 100.0), price=current_price)
        
        signal.stop_loss = self.get_stop_loss(signal, market_data)
        signal.take_profit = self.get_take_profit(signal, market_data)

        return signal

    def get_stop_loss(self, signal: TradeSignal, market_data: MarketData) -> float:
        atr_vals = market_data.m5_candles.atr(14)
        atr = atr_vals[-1] if len(atr_vals) > 0 and not np.isnan(atr_vals[-1]) else 1.0
        
        if signal.direction == "BUY":
            return market_data.current_price - (atr * self.sl_atr)
        else:
            return market_data.current_price + (atr * self.sl_atr)

    def get_take_profit(self, signal: TradeSignal, market_data: MarketData) -> float:
        atr_vals = market_data.m5_candles.atr(14)
        atr = atr_vals[-1] if len(atr_vals) > 0 and not np.isnan(atr_vals[-1]) else 1.0
        
        risk_dist = atr * self.sl_atr
        target_dist = risk_dist * self.rr_target 
        
        if signal.direction == "BUY":
            return market_data.current_price + target_dist
        else:
            return market_data.current_price - target_dist

    # --- REQUIRED ABSTRACT METHODS FOR DASHBOARD UI ---
    def get_metrics(self, market_data: MarketData) -> Dict[str, Any]:
        """Provides live dashboard metrics."""
        m5 = market_data.m5_candles
        if m5 is None or len(m5) < self.ema_fast: return {}
        
        adx_vals = m5.adx(self.adx_period)
        atr_vals = m5.atr(14)
        
        return {
            "ADX": adx_vals[-1] if len(adx_vals) > 0 else 0,
            "Vol Spike": atr_vals[-1]/np.mean(atr_vals[-14:]) if len(atr_vals) >= 14 else 0
        }

    def get_thresholds(self) -> Dict[str, Any]:
        """Provides threshold references for the UI."""
        return {
            "ADX": f"> {self.adx_threshold}",
            "Vol Spike": f"< {self.vol_exclusion_mult}x"
        }