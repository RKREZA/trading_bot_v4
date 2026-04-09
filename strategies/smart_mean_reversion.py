import numpy as np
import logging
from typing import Optional, Dict, Any
from core.base_strategy import BaseStrategy, MarketData
from core.common.types import TradeSignal

logger = logging.getLogger("trading_bot.strategy.smart_mean_reversion")

class SmartMeanReversionStrategy(BaseStrategy):
    """
    V4 Institutional Smart Mean Reversion (Numpy-Hardened).
    Fades algorithmic chop using manual mathematical arrays for BB and RSI.
    """

    def __init__(self, strategy_id: str, config: dict):
        super().__init__(strategy_id, config)
        
        # Priority: strategy_id -> "SmartMeanReversion"
        strat_config = self.config.get(strategy_id, self.config.get("SmartMeanReversion", {}))
        self.enabled = strat_config.get("enabled", True)
        
        self.bb_period = strat_config.get("bb_period", 20)
        self.bb_std = strat_config.get("bb_std", 2.5) 
        self.rsi_period = strat_config.get("rsi_period", 14)
        self.rsi_overbought = strat_config.get("rsi_overbought", 70)
        self.rsi_oversold = strat_config.get("rsi_oversold", 30)
        
        self.sl_atr = strat_config.get("sl_atr", 1.5)
        self.rr_target = strat_config.get("rr_target", 2.0) 

    def _calculate_rsi(self, closes: np.ndarray, period: int) -> float:
        """Manual RSI calculation with safe-guard for empty/small arrays."""
        if len(closes) < period + 1: return 50.0
        
        deltas = np.diff(closes)
        gains = np.where(deltas > 0, deltas, 0.0)
        losses = np.where(deltas < 0, -deltas, 0.0)
        
        # Look only at the required period
        relevant_gains = gains[-period:]
        relevant_losses = losses[-period:]
        
        avg_gain = np.mean(relevant_gains)
        avg_loss = np.mean(relevant_losses)
        
        if avg_loss == 0: return 100.0
        rs = avg_gain / avg_loss
        return 100.0 - (100.0 / (1.0 + rs))

    def generate_signal(self, market_data: MarketData) -> Optional[TradeSignal]:
        # 1. Session Gating
        strat_config = self.config.get(self.strategy_id, self.config.get("SmartMeanReversion", {}))
        allowed_sessions = strat_config.get("allowed_sessions", [])
        if allowed_sessions and market_data.session not in allowed_sessions:
            self.last_rejection_reason = f"Out of Session ({market_data.session})"
            return None

        # 2. Hard Data Buffer Guard
        m5 = market_data.m5_candles
        required_len = max(self.bb_period, self.rsi_period + 1)
        if m5 is None or len(m5) < required_len:
            self.last_rejection_reason = "MR: Warming up data buffer"
            return None

        # 3. Standardized Institutional Indicators (Audit Bug #2 Fix)
        # Using unified CandleArray methods ensures parity with backtester.
        upper_bands, lower_bands, sma_bands = m5.bollinger_bands(self.bb_period, self.bb_std)
        upper_band = upper_bands[-1]
        lower_band = lower_bands[-1]
        
        rsi_series = m5.rsi(self.rsi_period)
        current_rsi = rsi_series[-1]

        if np.isnan(upper_band) or np.isnan(current_rsi):
            return None

        # 4. Price Action Context
        last_candle = m5[-1]
        candle_range = last_candle.high - last_candle.low
        if candle_range <= 0: return None
        
        top_wick = last_candle.high - max(last_candle.open, last_candle.close)
        bottom_wick = min(last_candle.open, last_candle.close) - last_candle.low
        current_price = market_data.current_price

        # --- SIGNAL GENERATION ---
        
        # SELL: Over-extended High
        if current_price >= upper_band and current_rsi >= self.rsi_overbought:
            if (top_wick / candle_range) > 0.25:
                return TradeSignal(direction="SELL", price=current_price, confidence=0.80)
            else:
                self.last_rejection_reason = "MR: Weak Top Wick"
                return None

        # BUY: Over-extended Low
        if current_price <= lower_band and current_rsi <= self.rsi_oversold:
            if (bottom_wick / candle_range) > 0.25:
                return TradeSignal(direction="BUY", price=current_price, confidence=0.80)
            else:
                self.last_rejection_reason = "MR: Weak Bottom Wick"
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
        risk = atr * self.sl_atr
        dist = risk * self.rr_target
        return market_data.current_price + dist if signal.direction == "BUY" else market_data.current_price - dist
