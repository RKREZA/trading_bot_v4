import numpy as np
import logging
from typing import Optional, Dict, Any
from core.base_strategy import BaseStrategy, MarketData
from core.common.types import TradeSignal
from core.session_detector import SessionDetector

logger = logging.getLogger("trading_bot.strategy.trend")

class TrendFollowingStrategy(BaseStrategy):
    """
    V4 Institutional Trend Following - Clean Momentum Strategy.
    Uses ADX for trend strength confirmation + multi-TF alignment.
    """

    def __init__(self, strategy_id: str, config: dict):
        super().__init__(strategy_id, config)
        
        strat_config = self.config.get(strategy_id, self.config.get("TrendFollowing", {}))
        self.enabled = strat_config.get("enabled", True)
        
        self.adx_period = 14
        self.adx_threshold = strat_config.get("adx_threshold", 30)
        self.adx_strong = strat_config.get("adx_strong", 35)
        
        self.rsi_period = strat_config.get("rsi_period", 14)
        
        self.min_bars_between_signals = strat_config.get("min_bars_between_signals", 30)
        self._last_signal_bar = 0
        
        self.sl_atr = strat_config.get("sl_atr", 2.0)
        self.tp_atr = strat_config.get("tp_atr", 4.0)
        self.min_confidence = strat_config.get("min_confidence", 0.70)
        
        self.session_multipliers = {
            "TOKYO": {"adx_boost": 0, "conf_boost": 0.10},
            "LONDON": {"adx_boost": 5, "conf_boost": 0.05},
            "NEW_YORK": {"adx_boost": 0, "conf_boost": 0.05},
            "LONDON/NY": {"adx_boost": 8, "conf_boost": 0.00},
            "GLOBAL": {"adx_boost": 0, "conf_boost": 0.05}
        }

    def generate_signal(self, market_data: MarketData) -> Optional[TradeSignal]:
        strat_config = self.config.get(self.strategy_id, self.config.get("TrendFollowing", {}))
        allowed_sessions = strat_config.get("allowed_sessions", [])
        
        if not SessionDetector.is_session_active(market_data.timestamp, allowed_sessions=allowed_sessions):
            self.last_rejection_reason = f"Out of Session ({market_data.session})"
            return None

        m5 = market_data.m5_candles
        if m5 is None or len(m5) < 50:
            return None
        
        bars_since_last = len(m5) - self._last_signal_bar
        if bars_since_last < self.min_bars_between_signals:
            self.last_rejection_reason = "Signal cooldown active"
            return None
        
        session_mult = self.session_multipliers.get(market_data.session, {"adx_boost": 0, "conf_boost": 0})
        effective_adx_threshold = self.adx_threshold + session_mult["adx_boost"]
        
        adx_vals = m5.adx(self.adx_period)
        if len(adx_vals) < 2:
            return None
            
        adx = adx_vals[-1]
        adx_prev = adx_vals[-2]
        
        if adx < effective_adx_threshold:
            self.last_rejection_reason = f"ADX too low ({adx:.1f} < {effective_adx_threshold})"
            return None
        
        if adx <= adx_prev:
            self.last_rejection_reason = f"ADX not rising ({adx:.1f} <= {adx_prev:.1f})"
            return None
        
        m15_trend = self.get_ema_trend(market_data.m15_candles)
        if m15_trend == 0:
            self.last_rejection_reason = "M15 no trend"
            return None
        
        h1_trend = self.get_ema_trend(market_data.htf_candles)
        if h1_trend == 0:
            self.last_rejection_reason = "H1 no trend"
            return None
        
        if m15_trend != h1_trend:
            self.last_rejection_reason = "M15/H1 trend mismatch"
            return None
        
        m5_fast = m5.ema(8)
        m5_slow = m5.ema(21)
        if len(m5_fast) < 2 or len(m5_slow) < 2:
            return None
            
        m5_dir = 1 if m5_fast[-1] > m5_slow[-1] else -1
        m5_dir_prev = 1 if m5_fast[-2] > m5_slow[-2] else -1
        
        if m5_dir != m15_trend:
            self.last_rejection_reason = "M5/M15 trend mismatch"
            return None
        
        if m5_dir == m5_dir_prev:
            self.last_rejection_reason = "No EMA cross"
            return None
        
        rsi_vals = m5.rsi(self.rsi_period)
        if len(rsi_vals) > 0:
            rsi = rsi_vals[-1]
            if m5_dir == 1 and rsi > 65:
                self.last_rejection_reason = f"RSI stretched ({rsi:.1f})"
                return None
            if m5_dir == -1 and rsi < 35:
                self.last_rejection_reason = f"RSI stretched ({rsi:.1f})"
                return None
        
        last_candle = m5[-1]
        candle_range = last_candle.high - last_candle.low
        if candle_range > 0:
            candle_body = abs(last_candle.close - last_candle.open)
            candle_strength = candle_body / candle_range
            if candle_strength < 0.4:
                self.last_rejection_reason = f"Weak candle ({candle_strength:.2f})"
                return None
        
        atr_vals = m5.atr(self.adx_period)
        if len(atr_vals) == 0:
            return None
        avg_atr = np.mean(atr_vals[-14:])
        current_atr = atr_vals[-1]
        if current_atr > avg_atr * 2.0:
            self.last_rejection_reason = "High volatility"
            return None
        
        current_price = market_data.current_price
        direction = "BUY" if m5_dir == 1 else "SELL"
        
        base_conf = 0.70 + session_mult["conf_boost"]
        adx_conf = min(0.20, (adx - self.adx_threshold) / 25.0) if adx >= self.adx_threshold else 0.0
        momentum_conf = 0.1 if adx >= self.adx_strong else 0.0
        confidence = base_conf + adx_conf + momentum_conf
        
        self._last_signal_bar = len(m5)
        
        signal = TradeSignal(direction=direction, confidence=min(0.98, confidence), price=current_price)
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
        
        target_dist = atr * self.tp_atr
        
        if signal.direction == "BUY":
            return market_data.current_price + target_dist
        else:
            return market_data.current_price - target_dist

    def get_metrics(self, market_data: MarketData) -> Dict[str, Any]:
        m5 = market_data.m5_candles
        if m5 is None or len(m5) < 14: return {}
        
        adx_vals = m5.adx(self.adx_period)
        atr_vals = m5.atr(14)
        
        return {
            "ADX": adx_vals[-1] if len(adx_vals) > 0 else 0,
            "Vol": atr_vals[-1]/np.mean(atr_vals[-14:]) if len(atr_vals) >= 14 else 0
        }

    def get_thresholds(self) -> Dict[str, Any]:
        return {
            "ADX": f"> {self.adx_threshold}",
            "Cross": "EMA 8/21 required"
        }