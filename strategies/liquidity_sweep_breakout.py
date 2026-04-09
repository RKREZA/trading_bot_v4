import numpy as np
import logging
from typing import Optional, Dict, Any
from core.base_strategy import BaseStrategy, MarketData
from core.common.types import TradeSignal
from core.session_detector import SessionDetector

logger = logging.getLogger("trading_bot.strategy.liquidity_sweep_breakout")

class LiquiditySweepBreakoutStrategy(BaseStrategy):
    """
    V4 Institutional Liquidity Sweep Breakout.
    Targeting High-Momentum Range Breaks with Candle Strength filters.
    """

    def __init__(self, strategy_id: str, config: dict):
        super().__init__(strategy_id, config)
        
        strat_config = self.config.get(strategy_id, self.config.get("LiquiditySweepBreakout", {}))
        self.enabled = strat_config.get("enabled", True)
        
        self.lookback = strat_config.get("lookback", 20)
        self.body_thresh = strat_config.get("body_thresh", 0.70)
        self.h1_strength_thresh = strat_config.get("h1_strength_thresh", 0.55)
        self.min_confidence = strat_config.get("min_confidence", 0.75)
        
        self.sl_atr = strat_config.get("sl_atr", 2.0)
        self.tp_atr = strat_config.get("tp_atr", 6.0)
        
        self.min_bars_between_signals = strat_config.get("min_bars_between_signals", 25)
        self._last_signal_bar = 0
        
        self.session_multipliers = {
            "TOKYO": {"body_boost": 0.05, "h1_boost": 0.05, "conf_boost": 0.05},
            "LONDON": {"body_boost": 0.05, "h1_boost": 0.05, "conf_boost": 0.05},
            "NEW_YORK": {"body_boost": 0.0, "h1_boost": 0.0, "conf_boost": 0.0},
            "LONDON/NY": {"body_boost": 0.10, "h1_boost": 0.10, "conf_boost": 0.10},
            "GLOBAL": {"body_boost": 0.0, "h1_boost": 0.0, "conf_boost": 0.0}
        }

    def generate_signal(self, market_data: MarketData) -> Optional[TradeSignal]:
        strat_config = self.config.get(self.strategy_id, self.config.get("LiquiditySweepBreakout", {}))
        allowed_sessions = strat_config.get("allowed_sessions", [])
        if not SessionDetector.is_session_active(market_data.timestamp, allowed_sessions=allowed_sessions):
            self.last_rejection_reason = f"Out of Session ({market_data.session})"
            return None
        
        m5 = market_data.m5_candles
        if len(m5) < self.lookback + 1:
            self.last_rejection_reason = "Breakout: M5 Insufficient data"
            return None
        
        bars_since_last = len(m5) - self._last_signal_bar
        if bars_since_last < self.min_bars_between_signals:
            self.last_rejection_reason = "Signal cooldown active"
            return None

        session_mult = self.session_multipliers.get(market_data.session, {"body_boost": 0, "h1_boost": 0, "conf_boost": 0})
        effective_body_thresh = self.body_thresh + session_mult["body_boost"]
        effective_h1_thresh = self.h1_strength_thresh + session_mult["h1_boost"]

        h1_candles = market_data.htf_candles
        if len(h1_candles) < 22:
            self.last_rejection_reason = "Breakout: H1 Insufficient data"
            return None
            
        h1 = h1_candles[-1]
        m15_trend = self.get_ema_trend(market_data.m15_candles)
        
        h1_candle_range = h1.high - h1.low
        h1_body = abs(h1.close - h1.open)
        h1_strength = (h1_body / h1_candle_range) if h1_candle_range > 0 else 0
        h1_dir = 1 if h1.close > h1.open else -1
        
        h1_v = h1_candles.v
        h1_vol_sma = np.mean(h1_v[-21:-1]) 
        minutes_into_hour = market_data.timestamp.minute
        completion_pct = max(0.05, (minutes_into_hour + 1) / 60.0)
        dynamic_threshold = h1_vol_sma * completion_pct
        vol_confirmed = h1.tick_volume > dynamic_threshold
        
        prev_range = m5[-self.lookback-1:-1]
        r_high = np.max(prev_range.high)
        r_low = np.min(prev_range.low)

        last = m5[-1]
        price = market_data.current_price
        m5_range = last.high - last.low
        m5_strength = (abs(last.close - last.open) / m5_range) if m5_range > 0 else 0

        if price > r_high and m5_strength >= effective_body_thresh:
            if h1_dir == 1 and h1_strength >= effective_h1_thresh and m15_trend != -1 and vol_confirmed:
                self._last_signal_bar = len(m5)
                confidence = 0.80 + session_mult["conf_boost"] + min(0.15, h1_strength * 0.2)
                return TradeSignal(direction="BUY", price=price, confidence=min(0.98, confidence))
        
        if price < r_low and m5_strength >= effective_body_thresh:
            if h1_dir == -1 and h1_strength >= effective_h1_thresh and m15_trend != 1 and vol_confirmed:
                self._last_signal_bar = len(m5)
                confidence = 0.80 + session_mult["conf_boost"] + min(0.15, h1_strength * 0.2)
                return TradeSignal(direction="SELL", price=price, confidence=min(0.98, confidence))
        
        if price > r_high or price < r_low:
             if m5_strength < effective_body_thresh: self.last_rejection_reason = f"Breakout: M5 Strength too low ({m5_strength:.2f})"
             elif h1_strength < effective_h1_thresh: self.last_rejection_reason = f"Breakout: H1 Strength too low ({h1_strength:.2f})"
             elif not vol_confirmed: self.last_rejection_reason = "Breakout: Volume not confirmed"
             else: self.last_rejection_reason = "Breakout: MTF/Dir Mismatch"
        else:
             self.last_rejection_reason = "Breakout: Price inside range"

        return None

    def get_metrics(self, market_data: MarketData) -> Dict[str, Any]:
        if not market_data.htf_candles or len(market_data.htf_candles) < 22:
            return {}
            
        h1 = market_data.htf_candles[-1]
        m5 = market_data.m5_candles
        price = market_data.current_price
        
        h1_candle_range = h1.high - h1.low
        h1_body = abs(h1.close - h1.open)
        h1_strength = (h1_body / h1_candle_range) if h1_candle_range > 0 else 0
        
        h1_v = market_data.htf_candles.v
        h1_vol_sma = np.mean(h1_v[-21:-1])
        minutes_into_hour = market_data.timestamp.minute
        completion_pct = max(0.05, (minutes_into_hour + 1) / 60.0)
        dynamic_threshold = h1_vol_sma * completion_pct
        
        if len(m5) < self.lookback + 1:
            return {"H1 Strength": h1_strength, "H1 Volume": h1.tick_volume}

        prev_range = m5[-self.lookback-1:-1]
        r_high = np.max(prev_range.high)
        r_low = np.min(prev_range.low)
        last = m5[-1]
        m5_range = last.high - last.low
        m5_strength = (abs(last.close - last.open) / m5_range) if m5_range > 0 else 0
        
        price_state = "Inside Range"
        if price > r_high: price_state = "Break High"
        elif price < r_low: price_state = "Break Low"
        
        return {
            "H1 Body": h1_strength,
            "M5 Body": m5_strength,
            "Volume": h1.tick_volume / dynamic_threshold if dynamic_threshold > 0 else 0,
            "Range": price_state
        }

    def get_thresholds(self) -> Dict[str, Any]:
        return {
            "H1 Body": f"> {self.h1_strength_thresh:.2f}",
            "M5 Body": f"> {self.body_thresh:.2f}",
            "Volume": "> 1.0x",
            "Range": "Breakout"
        }

    def get_stop_loss(self, signal: TradeSignal, market_data: MarketData) -> float:
        last = market_data.m5_candles[-1]
        atr_vals = market_data.m5_candles.atr(14)
        atr = atr_vals[-1] if len(atr_vals) > 0 and not np.isnan(atr_vals[-1]) else 1.0
        
        if signal.direction == "BUY":
            return min(last.low, market_data.current_price - (atr * self.sl_atr))
        return max(last.high, market_data.current_price + (atr * self.sl_atr))

    def get_take_profit(self, signal: TradeSignal, market_data: MarketData) -> float:
        atr_vals = market_data.m5_candles.atr(14)
        atr = atr_vals[-1] if len(atr_vals) > 0 and not np.isnan(atr_vals[-1]) else 1.0
        target_dist = atr * self.tp_atr
        
        if signal.direction == "BUY":
            return market_data.current_price + target_dist
        return market_data.current_price - target_dist
