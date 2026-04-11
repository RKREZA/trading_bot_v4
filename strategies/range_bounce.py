import numpy as np
import logging
from typing import Optional, Dict, Any
from core.base_strategy import BaseStrategy, MarketData
from core.common.types import TradeSignal
from core.session_detector import SessionDetector

logger = logging.getLogger("trading_bot.strategy.range_bounce")


class RangeBounceStrategy(BaseStrategy):
    """
    V5 OPTIMIZED Range Bounce Strategy for Mean Reversion.
    Uses RSI extremes + MACD confirmation for bounces.
    Includes trend filter to avoid counter-trend trades.
    """

    def __init__(self, strategy_id: str, config: dict):
        super().__init__(strategy_id, config)
        
        # [ Institutional Config Resolution ]: Access the resolved strategy block
        strat_config = self.get_strat_config()
        
        self.bb_period = strat_config.get("bb_period", 20)
        self.bb_std = strat_config.get("bb_std", 2.0)
        self.rsi_period = strat_config.get("rsi_period", 14)
        self.rsi_oversold = strat_config.get("rsi_oversold", 35)
        self.rsi_overbought = strat_config.get("rsi_overbought", 65)
        self.rsi_extreme_oversold = strat_config.get("rsi_extreme_oversold", 25)
        self.rsi_extreme_overbought = strat_config.get("rsi_extreme_overbought", 75)
        
        self.sl_atr = strat_config.get("sl_atr", 2.0)
        self.tp_atr = strat_config.get("tp_atr", 4.0)
        self.min_confidence = float(strat_config.get("min_confidence", self.min_confidence))
        self.min_bars_between_signals = strat_config.get("min_bars_between_signals", 10)
        self._last_signal_bar = 0
        
        self.adx_threshold = strat_config.get("adx_threshold", 25)
        self.trend_ema_fast = strat_config.get("trend_ema_fast", 50)
        self.trend_ema_slow = strat_config.get("trend_ema_slow", 200)

    def _calculate_rsi(self, prices: np.ndarray, period: int) -> np.ndarray:
        """Calculate RSI values."""
        if len(prices) < period + 2:
            return np.full(len(prices), 50.0)
        
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0.0)
        losses = np.where(deltas < 0, -deltas, 0.0)
        
        n = len(prices)
        avg_gains = np.zeros(n)
        avg_losses = np.zeros(n)
        
        avg_gains[period] = np.mean(gains[:period])
        avg_losses[period] = np.mean(losses[:period])
        
        for i in range(period + 1, n):
            avg_gains[i] = (avg_gains[i-1] * (period - 1) + gains[i-1]) / period
            avg_losses[i] = (avg_losses[i-1] * (period - 1) + losses[i-1]) / period
        
        rs = np.divide(avg_gains, avg_losses, out=np.zeros_like(avg_gains), where=avg_losses != 0)
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def _calculate_bollinger_bands(self, prices: np.ndarray, period: int, std_mult: float) -> tuple:
        """Calculate Bollinger Bands (upper, middle, lower)."""
        if len(prices) < period:
            return 0.0, 0.0, 0.0
        
        sma = np.mean(prices[-period:])
        std = np.std(prices[-period:])
        upper = sma + (std_mult * std)
        lower = sma - (std_mult * std)
        
        return upper, sma, lower

    def _get_trend_direction(self, m5) -> int:
        """Get trend direction from EMA crossover. Returns 1=up, -1=down, 0=neutral."""
        if len(m5.close) < self.trend_ema_slow + 2:
            return 0
        
        ema_fast = m5.ema(self.trend_ema_fast)
        ema_slow = m5.ema(self.trend_ema_slow)
        
        if len(ema_fast) < 2 or len(ema_slow) < 2:
            return 0
        
        if ema_fast[-1] > ema_slow[-1] and ema_fast[-2] <= ema_slow[-2]:
            return 1
        elif ema_fast[-1] < ema_slow[-1] and ema_fast[-2] >= ema_slow[-2]:
            return -1
        elif ema_fast[-1] > ema_slow[-1]:
            return 1
        elif ema_fast[-1] < ema_slow[-1]:
            return -1
        return 0

    def generate_signal(self, market_data: MarketData) -> Optional[TradeSignal]:
        strat_config = self.get_strat_config()
        allowed_sessions = strat_config.get("allowed_sessions", [])
        if not SessionDetector.is_session_active(market_data.timestamp, allowed_sessions=allowed_sessions):
            self.last_rejection_reason = f"Out of Session ({market_data.session})"
            return None
        
        m5 = market_data.m5_candles
        if len(m5) < max(self.bb_period, self.rsi_period, self.trend_ema_slow, 100) + 2:
            self.last_rejection_reason = "RangeBounce: Insufficient data"
            return None
        
        # 1. HARDENING: Relative Volatility Gate (Audit Pass #10)
        # Prevents entries during runaway volatility or news breakouts
        atr_14 = m5.get_indicator("atr_14")
        if len(atr_14) > 100:
            current_atr = atr_14[-1]
            avg_atr_100 = np.mean(atr_14[-100:])
            vol_ratio = current_atr / avg_atr_100 if avg_atr_100 > 0 else 1.0
            max_vol = strat_config.get("max_vol_ratio", 1.5)
            if vol_ratio > max_vol:
                self.last_rejection_reason = f"Volatility Gated: Ratio {vol_ratio:.2f} > {max_vol}"
                return None

        # 2. HARDENING: ADX Momentum & Absolute Filter
        # Reject if ADX is rising rapidly or if the trend is already too strong (Absolute Gate)
        adx_14 = m5.get_indicator("adx_14")
        if len(adx_14) > 3:
            current_adx = adx_14[-1]
            # Absolute Gate: Institutional mean-reversion is suicide if ADX > 35
            if current_adx > 35:
                self.last_rejection_reason = f"Trend Too Strong (ADX {current_adx:.1f} > 35)"
                return None
                
            adx_slope = adx_14[-1] - adx_14[-3]
            max_adx_slope = strat_config.get("max_adx_slope", 7.0)
            if adx_slope > max_adx_slope:
                self.last_rejection_reason = f"ADX Slope Gated: Slope {adx_slope:.1f} > {max_adx_slope}"
                return None

        closes = m5.close
        price = market_data.current_price
        
        bb_upper, bb_mid, bb_lower = self._calculate_bollinger_bands(closes, self.bb_period, self.bb_std)
        if bb_upper == 0:
            self.last_rejection_reason = "RangeBounce: BB not ready"
            return None

        # 2.5 HARDENING: Bollinger "Walk" Detection
        # If the last 3 closes were all within 2% of the outer band, it's a trend, not a bounce.
        if len(closes) > 3:
            # We check the relative position of previous closes
            recent_closes = closes[-3:]
            
            # Detect Long Walk (Price hugging upper band)
            is_hugging_upper = all(c > (bb_upper * 0.98) for c in recent_closes)
            # Detect Short Walk (Price hugging lower band)
            is_hugging_lower = all(c < (bb_lower * 1.02) for c in recent_closes)
            
            if is_hugging_upper or is_hugging_lower:
                self.last_rejection_reason = "Bollinger Walk Detected (Momentum prevents bounce)"
                return None

        bars_since_last = len(m5) - self._last_signal_bar
        if bars_since_last < self.min_bars_between_signals:
            self.last_rejection_reason = "Signal cooldown active"
            return None

        rsi_vals = m5.get_indicator("rsi_14")
        rsi = rsi_vals[-1] if len(rsi_vals) > 0 else 50
        prev_rsi = rsi_vals[-2] if len(rsi_vals) > 1 else rsi
        
        trend = self._get_trend_direction(m5)
        
        bb_range = bb_upper - bb_lower
        # Protect against division by zero
        bb_position = (price - bb_lower) / bb_range if bb_range > 0.00001 else 0.5
        
        oversold = rsi <= self.rsi_oversold
        overbought = rsi >= self.rsi_overbought
        extreme_oversold = rsi <= self.rsi_extreme_oversold
        extreme_overbought = rsi >= self.rsi_extreme_overbought
        
        # 3. HARDENING: RSI "Return to Range" (Precision Entry)
        # Only required for standard signals, extreme signals can enter instantly
        is_rsi_rev_up = rsi >= (prev_rsi - 0.5)
        is_rsi_rev_down = rsi <= (prev_rsi + 0.5)

        buy_signal = False
        sell_signal = False
        confidence = 0.0
        
        if extreme_oversold and bb_position < 0.20:
            buy_signal = True
            confidence = 0.85
        elif oversold and bb_position < 0.35 and is_rsi_rev_up:
            if trend == 1:
                buy_signal = True
                confidence = 0.75
            elif trend == 0 and rsi <= 35:
                buy_signal = True
                confidence = 0.65
        
        if extreme_overbought and bb_position > 0.80:
            sell_signal = True
            confidence = 0.85
        elif overbought and bb_position > 0.65 and is_rsi_rev_down:
            if trend == -1:
                sell_signal = True
                confidence = 0.75
            elif trend == 0 and rsi >= 65:
                sell_signal = True
                confidence = 0.65
        
        if buy_signal:
            self._last_signal_bar = len(m5)
            return TradeSignal(direction="BUY", price=price, confidence=min(0.95, confidence))
        
        if sell_signal:
            self._last_signal_bar = len(m5)
            return TradeSignal(direction="SELL", price=price, confidence=min(0.95, confidence))
        
        if bb_position < 0.10:
            self.last_rejection_reason = f"RangeBounce: Price at BB lower ({bb_position:.2f})"
        elif bb_position > 0.90:
            self.last_rejection_reason = f"RangeBounce: Price at BB upper ({bb_position:.2f})"
        elif rsi > self.rsi_overbought:
            self.last_rejection_reason = f"RangeBounce: RSI overbought ({rsi:.1f})"
        elif rsi < self.rsi_oversold:
            self.last_rejection_reason = f"RangeBounce: RSI oversold ({rsi:.1f})"
        else:
            self.last_rejection_reason = f"RangeBounce: No setup (RSI={rsi:.1f}, BB%={bb_position:.2f})"
        
        return None

    def get_stop_loss(self, signal: TradeSignal, market_data: MarketData) -> float:
        m5 = market_data.m5_candles
        atr_vals = m5.atr(14)
        atr = atr_vals[-1] if len(atr_vals) > 0 and not np.isnan(atr_vals[-1]) else 1.0
        
        scaler = self.get_regime_scaler(market_data)
        
        if signal.direction == "BUY":
            return market_data.current_price - (atr * self.sl_atr * scaler)
        else:
            return market_data.current_price + (atr * self.sl_atr * scaler)

    def get_take_profit(self, signal: TradeSignal, market_data: MarketData) -> float:
        m5 = market_data.m5_candles
        atr_vals = m5.atr(14)
        atr = atr_vals[-1] if len(atr_vals) > 0 and not np.isnan(atr_vals[-1]) else 1.0
        
        scaler = self.get_regime_scaler(market_data)
        
        if signal.direction == "BUY":
            return market_data.current_price + (atr * self.tp_atr * scaler)
        else:
            return market_data.current_price - (atr * self.tp_atr * scaler)

    def get_metrics(self, market_data: MarketData) -> Dict[str, Any]:
        if len(market_data.m5_candles) < self.bb_period + 2:
            return {}
        
        m5 = market_data.m5_candles
        rsi_vals = self._calculate_rsi(m5.close, self.rsi_period)
        rsi = rsi_vals[-1] if len(rsi_vals) > 0 else 50
        
        bb_upper, bb_mid, bb_lower = self._calculate_bollinger_bands(m5.close, self.bb_period, self.bb_std)
        bb_range = bb_upper - bb_lower
        bb_position = (market_data.current_price - bb_lower) / bb_range if bb_range > 0 else 0.5
        
        return {
            "rsi": rsi,
            "bb_upper": bb_upper,
            "bb_lower": bb_lower,
            "bb_position": bb_position,
            "price": market_data.current_price
        }
