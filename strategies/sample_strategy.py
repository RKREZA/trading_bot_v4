"""
Sample Strategy: Simple Moving Average Crossover
================================================
This strategy demonstrates how to create a new strategy for the V5-INSIGNIA system.
It uses a simple moving average crossover on the M15 timeframe.
"""

import numpy as np
from typing import Optional
from core.base_strategy import BaseStrategy, MarketData
from core.common.types import TradeSignal


class SMASampleStrategy(BaseStrategy):
    """
    A simple moving average crossover strategy.
    - Fast SMA: 20 periods
    - Slow SMA: 50 periods
    - BUY when fast SMA crosses above slow SMA
    - SELL when fast SMA crosses below slow SMA
    """

    def __init__(self, strategy_id: str, config: dict):
        super().__init__(strategy_id, config)
        # Strategy-specific parameters from config
        self.fast_period = self.config.get("fast_sma_period", 20)
        self.slow_period = self.config.get("slow_sma_period", 50)
        self.min_confidence = self.config.get("min_confidence", 0.7)

    def generate_signal(self, market_data: MarketData) -> Optional[TradeSignal]:
        """
        Generate a trading signal based on SMA crossover.
        Returns a TradeSignal with direction, confidence, and price.
        """
        # Use M15 candles for this example
        candles = market_data.m15_candles
        if candles is None or len(candles) < self.slow_period:
            return None

        close_prices = candles.close

        # Calculate SMAs
        fast_sma = np.mean(close_prices[-self.fast_period:])
        slow_sma = np.mean(close_prices[-self.slow_period:])

        # Calculate previous SMAs to detect crossover
        if len(close_prices) < self.slow_period + 1:
            return None

        prev_close = candles.close[:-1]
        prev_fast_sma = np.mean(prev_close[-self.fast_period:])
        prev_slow_sma = np.mean(prev_close[-self.slow_period:])

        # Check for crossover
        signal = None
        if prev_fast_sma <= prev_slow_sma and fast_sma > slow_sma:
            # Bullish crossover
            signal = TradeSignal(
                direction="BUY",
                price=float(candles.close[-1]),
                confidence=self.min_confidence,
                timestamp=market_data.timestamp
            )
        elif prev_fast_sma >= prev_slow_sma and fast_sma < slow_sma:
            # Bearish crossover
            signal = TradeSignal(
                direction="SELL",
                price=float(candles.close[-1]),
                confidence=self.min_confidence,
                timestamp=market_data.timestamp
            )

        return signal

    def get_stop_loss(self, signal: TradeSignal, market_data: MarketData) -> float:
        """
        Calculate stop loss based on ATR.
        """
        # Use ATR(14) on M15
        atr = market_data.m15_candles.get_indicator("atr_14")
        if len(atr) == 0:
            # Fallback to a fixed percentage
            return signal.price * 0.98 if signal.direction == "BUY" else signal.price * 1.02

        atr_value = atr[-1]
        if signal.direction == "BUY":
            return signal.price - (atr_value * 2.0)  # 2 ATR below
        else:
            return signal.price + (atr_value * 2.0)  # 2 ATR above

    def get_take_profit(self, signal: TradeSignal, market_data: MarketData) -> float:
        """
        Calculate take profit based on a fixed risk-reward ratio.
        """
        sl = self.get_stop_loss(signal, market_data)
        risk = abs(signal.price - sl)

        # Use 1:2 risk-reward ratio
        if signal.direction == "BUY":
            return signal.price + (risk * 2.0)
        else:
            return signal.price - (risk * 2.0)

    def get_metrics(self, market_data: MarketData) -> dict:
        """
        Return current strategy metrics for the dashboard.
        """
        candles = market_data.m15_candles
        if candles is None or len(candles) < self.slow_period:
            return {}

        close_prices = candles.close
        fast_sma = np.mean(close_prices[-self.fast_period:])
        slow_sma = np.mean(close_prices[-self.slow_period:])

        return {
            "fast_sma": fast_sma,
            "slow_sma": slow_sma,
            "sma_diff": fast_sma - slow_sma,
            "current_price": float(candles.close[-1]) if len(close_prices) > 0 else 0.0
        }

    def get_thresholds(self) -> dict:
        """
        Return thresholds for the dashboard.
        """
        return {
            "fast_sma_period": self.fast_period,
            "slow_sma_period": self.slow_period,
            "min_confidence": self.min_confidence
        }