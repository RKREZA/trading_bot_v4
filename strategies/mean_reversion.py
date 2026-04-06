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
        self.bb_std = 2.0  # Increased for higher conviction
        self.atr_threshold_mult = 2.5 # Max atr allowed relative to mean
        self.loss_cooldown = 0
        self.last_loss_time = 0

    def generate_signal(self, market_data: MarketData) -> Optional[TradeSignal]:
        """
        Institutional MTF Mean Reversion (Step 11).
        Only allows mean reversion in HTF Neutral or Weak Trend regimes.
        """
        # 1. MTF Bias Check
        h1_trend = self.get_ema_trend(market_data.htf_candles)
        
        m5 = market_data.m5_candles
        price = market_data.current_price

        # 2. RSI(14) - Tightened for Institutional Conviction
        rsi_vals = m5.rsi(self.rsi_period)
        rsi = rsi_vals[-1] if len(rsi_vals) > 0 else np.nan
        
        # 3. ATR Volatility Guard (Phase 2 Hardening)
        atr_vals = m5.atr(14)
        if len(atr_vals) < 14 or np.isnan(rsi):
            return None
        
        current_atr = atr_vals[-1]
        avg_atr = np.mean(atr_vals[-14:])
        if current_atr > (avg_atr * self.atr_threshold_mult):
            # Market is moved too fast (Parabolic), Mean Reversion is high risk
            return None

        # 4. Loss Cooldown Check
        if self.loss_cooldown > 0:
            self.loss_cooldown -= 1
            return None
        
        # 5. Bollinger Bands (20, 2)
        upper_vals, lower_vals, _ = m5.bollinger_bands(self.bb_period, self.bb_std)
        upper, lower = upper_vals[-1], lower_vals[-1]
        if np.isnan(upper) or np.isnan(lower):
            return None

        # 6. Filtered Decision Logic
        # BUY: Oversold + Price < Lower BB + HTF is NOT Bearish
        if price < lower and rsi < 30 and h1_trend != -1:
            return TradeSignal(direction="BUY", price=price, confidence=0.75, timestamp=market_data.timestamp)
        
        # SELL: Overbought + Price > Upper BB + HTF is NOT Bullish
        if price > upper and rsi > 70 and h1_trend != 1:
            return TradeSignal(direction="SELL", price=price, confidence=0.75, timestamp=market_data.timestamp)

        return None

    def on_trade_closed(self, trade_record: dict) -> None:
        if trade_record.get("pnl", 0) < 0:
            # Implement 3-period cooldown on loss (Institutional Hardening)
            self.loss_cooldown = 3
            logger.info(f"[{self.strategy_id}] Loss detected. Cooldown activated.")

    def get_stop_loss(self, signal: TradeSignal, market_data: MarketData) -> float:
        atr_vals = market_data.m5_candles.atr(14)
        atr = atr_vals[-1] if len(atr_vals) > 0 and not np.isnan(atr_vals[-1]) else 1.0
        if signal.direction == "BUY":
            return market_data.current_price - (atr * 2.0)
        return market_data.current_price + (atr * 2.0)

    def get_take_profit(self, signal: TradeSignal, market_data: MarketData) -> float:
        sl_price = self.get_stop_loss(signal, market_data)
        risk = abs(market_data.current_price - sl_price)
        if signal.direction == "BUY":
            return market_data.current_price + (risk * 2.5)
        return market_data.current_price - (risk * 2.5)
