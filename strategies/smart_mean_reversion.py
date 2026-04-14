import numpy as np
import logging
from typing import Optional, Dict, Any
from core.base_strategy import BaseStrategy, MarketData
from core.common.types import TradeSignal
from core.session_detector import SessionDetector

logger = logging.getLogger("trading_bot.strategy.smart_mean_reversion")

class SmartMeanReversionStrategy(BaseStrategy):
    """
    V5 Institutional Smart Mean Reversion (Numpy-Hardened).
    Fades algorithmic chop using manual mathematical arrays for BB and RSI.
    """

    def __init__(self, strategy_id: str, config: dict):
        super().__init__(strategy_id, config)
        
        # [ Institutional Config Resolution ]: Access the resolved strategy block
        strat_config = self.get_strat_config()
        
        self.bb_period = strat_config.get("bb_period", 20)
        self.bb_std = strat_config.get("bb_std", 2.0)
        self.rsi_period = strat_config.get("rsi_period", 14)
        self.rsi_overbought = strat_config.get("rsi_overbought", 75)
        self.rsi_oversold = strat_config.get("rsi_oversold", 25)
        
        self.sl_atr = strat_config.get("sl_atr", 1.5)
        self.tp_atr = strat_config.get("tp_atr", 4.5)
        self.min_confidence = float(strat_config.get("min_confidence", self.min_confidence))
        
        # [ Institutional Upgrades ]
        self.adx_max_threshold = strat_config.get("adx_max_threshold", 25.0)
        self.vol_mult_threshold = strat_config.get("vol_mult_threshold", 1.25)
        self.min_wick_ratio = strat_config.get("min_wick_ratio", 0.10)
        
        self.min_bars_between_signals = strat_config.get("min_bars_between_signals", 15)
        self._last_signal_bar = 0

    def _calculate_rsi(self, closes: np.ndarray, period: int) -> float:
        if len(closes) < period + 1: return 50.0
        
        deltas = np.diff(closes)
        gains = np.where(deltas > 0, deltas, 0.0)
        losses = np.where(deltas < 0, -deltas, 0.0)
        
        relevant_gains = gains[-period:]
        relevant_losses = losses[-period:]
        
        avg_gain = np.mean(relevant_gains)
        avg_loss = np.mean(relevant_losses)
        
        if avg_loss == 0: return 100.0
        rs = avg_gain / avg_loss
        return 100.0 - (100.0 / (1.0 + rs))

    def generate_signal(self, market_data: MarketData) -> Optional[TradeSignal]:
        strat_config = self.get_strat_config()
        allowed_sessions = strat_config.get("allowed_sessions", [])
        if not SessionDetector.is_session_active(market_data.timestamp, allowed_sessions=allowed_sessions):
            self.last_rejection_reason = f"Out of Session ({market_data.session})"
            return None

        m5 = market_data.m5_candles
        required_len = max(self.bb_period, self.rsi_period + 1, 20)
        if m5 is None or len(m5) < required_len:
            self.last_rejection_reason = "MR: Warming up data buffer"
            return None
        
        bars_since_last = len(m5) - self._last_signal_bar
        if bars_since_last < self.min_bars_between_signals:
            self.last_rejection_reason = "Signal cooldown active"
            return None

        # [ Institutional Gating ]: Trend Filter (ADX)
        # Prevents fading a strong directional trend
        adx_vals = market_data.m15_candles.get_indicator("adx_14")
        current_adx = adx_vals[-1] if len(adx_vals) > 0 else 0
        if current_adx > self.adx_max_threshold:
            self.last_rejection_reason = f"MR: Trend too strong (ADX {current_adx:.1f})"
            return None

        # [ Institutional Gating ]: Volume Filter (Exhaustion)
        # MR works best on volume climaxes
        vol_sma = m5.get_indicator("vol_sma_20")[-1]
        last_vol = m5[-1].tick_volume
        if vol_sma > 0 and last_vol < vol_sma * self.vol_mult_threshold:
            self.last_rejection_reason = f"MR: No Volume Climax ({last_vol}/{int(vol_sma)})"
            return None

        upper_bands, lower_bands, sma_bands = m5.bollinger_bands(self.bb_period, self.bb_std)
        upper_band = upper_bands[-1]
        lower_band = lower_bands[-1]
        
        rsi_series = m5.rsi(self.rsi_period)
        current_rsi = rsi_series[-1]

        if np.isnan(upper_band) or np.isnan(current_rsi):
            return None

        last_candle = m5[-1]
        candle_range = last_candle.high - last_candle.low
        if candle_range <= 0: return None
        
        current_price = market_data.current_price
        
        bb_width = upper_band - lower_band
        price_deviation = abs(current_price - sma_bands[-1]) / bb_width if bb_width > 0 else 0
        
        # [ Calibration Relaxation ]: Lowered to ensure density in low-volatility pockets
        if price_deviation < 0.0005: 
            self.last_rejection_reason = "Price not extended from mean"
            return None
        
        top_wick = last_candle.high - max(last_candle.open, last_candle.close)
        bottom_wick = min(last_candle.open, last_candle.close) - last_candle.low

        # Wick rejection logic for better quality
        min_wick_ratio = self.min_wick_ratio 

        if current_price >= upper_band and current_rsi >= self.rsi_overbought:
            wick_ratio = top_wick / candle_range if candle_range > 0 else 0
            if wick_ratio > min_wick_ratio:
                self._last_signal_bar = len(m5)
                confidence = 0.70 + min(0.15, (current_rsi - self.rsi_overbought) / 50)
                return TradeSignal(direction="SELL", price=current_price, confidence=min(0.95, confidence))
            else:
                self.last_rejection_reason = f"MR: Weak Top Wick ({wick_ratio:.3f})"
                return None

        if current_price <= lower_band and current_rsi <= self.rsi_oversold:
            wick_ratio = bottom_wick / candle_range if candle_range > 0 else 0
            if wick_ratio > min_wick_ratio:
                self._last_signal_bar = len(m5)
                confidence = 0.70 + min(0.15, (self.rsi_oversold - current_rsi) / 50)
                return TradeSignal(direction="BUY", price=current_price, confidence=min(0.95, confidence))
            else:
                self.last_rejection_reason = f"MR: Weak Bottom Wick ({wick_ratio:.3f})"
                return None

        return None

    def get_metrics(self, market_data: MarketData) -> Dict[str, Any]:
        m5 = market_data.m5_candles
        if m5 is None or len(m5) < self.rsi_period + 1: return {}
        rsi = self._calculate_rsi(m5.c[-(self.rsi_period+1):], self.rsi_period)
        return {"RSI": rsi}

    def get_thresholds(self) -> Dict[str, Any]:
        return {"RSI Limit": f">{self.rsi_overbought} or <{self.rsi_oversold}"}

    def get_stop_loss(self, signal: TradeSignal, market_data: MarketData) -> float:
        atr_vals = market_data.m5_candles.atr(14)
        atr = atr_vals[-1] if len(atr_vals) > 0 else 1.0
        dist = atr * self.sl_atr
        return market_data.current_price - dist if signal.direction == "BUY" else market_data.current_price + dist

    def get_take_profit(self, signal: TradeSignal, market_data: MarketData) -> float:
        atr_vals = market_data.m5_candles.atr(14)
        atr = atr_vals[-1] if len(atr_vals) > 0 else 1.0
        dist = atr * self.tp_atr
        return market_data.current_price + dist if signal.direction == "BUY" else market_data.current_price - dist
