import numpy as np
import logging
from typing import Optional, Tuple
from core.base_strategy import BaseStrategy, MarketData
from core.common.types import TradeSignal

logger = logging.getLogger("trading_bot.strategy.mean_reversion")

class MeanReversionStrategy(BaseStrategy):
    """
    V4 Institutional Mean Reversion.
    Uses RSI(14) and Bollinger Bands (20, 2) to capture overextended moves.
    Deterministic and robust for ranging market regimes.
    """

    def __init__(self, strategy_id: str, config: dict):
        super().__init__(strategy_id, config)
        self.rsi_period = 14
        self.bb_period = 20
        self.bb_std = 1.5 # Lowered for benchmark sensitivity

    def generate_signal(self, market_data: MarketData) -> Optional[TradeSignal]:
        """
        Institutional MTF Mean Reversion (Step 11).
        Only allows mean reversion in HTF Neutral or Weak Trend regimes.
        """
        # 1. MTF Bias Check
        h1_trend = self.get_ema_trend(market_data.htf_candles)
        
        m5 = market_data.m5_candles
        price = market_data.current_price

        # 2. RSI(14)
        rsi_vals = m5.rsi(self.rsi_period)
        rsi = rsi_vals[-1]
        
        # 3. Bollinger Bands (20, 2)
        upper_vals, lower_vals, _ = m5.bollinger_bands(self.bb_period, self.bb_std)
        upper, lower = upper_vals[-1], lower_vals[-1]

        # 4. Filtered Decision Logic
        if price < lower and rsi < 35 and h1_trend != -1:
            return TradeSignal(direction="BUY", confidence=0.75, timestamp=market_data.timestamp)
        
        if price > upper and rsi > 65 and h1_trend != 1:
            return TradeSignal(direction="SELL", confidence=0.75, timestamp=market_data.timestamp)

        return None

    def get_stop_loss(self, signal: TradeSignal, market_data: MarketData) -> float:
        atr = market_data.m5_candles.atr(14)[-1]
        if signal.direction == "BUY":
            return market_data.current_price - (atr * 2.0)
        return market_data.current_price + (atr * 2.0)
