import numpy as np
import logging
from typing import Optional, Dict, Any
from core.base_strategy import BaseStrategy, MarketData
from core.common.types import TradeSignal

logger = logging.getLogger("trading_bot.strategy.pure_breakout_one_minute")

class PureBreakoutOneMinuteStrategy(BaseStrategy):
    """
    PureBreakoutOneMinute - SIMPLE MOMENTUM VERSION
    
    Trades on strong momentum candles with fixed SL/TP.
    """

    def __init__(self, strategy_id: str, config: dict):
        super().__init__(strategy_id, config)
        
        strat_config = self.config.get("PureBreakoutOneMinute", self.get_strat_config())
        
        self.min_body_ratio = strat_config.get("min_body_ratio", 0.60)
        self.sl_points = strat_config.get("sl_points", 5000)  # Fixed 5000 points SL ($50)
        self.tp_mult = strat_config.get("tp_mult", 2.0)  # 2x risk for TP
        self.lookback = strat_config.get("lookback", 5)
        self.fixed_lot = 0.05  # Fixed 0.05 lot
        
        self._last_setup = {}
        self._cooldown = 0
        self._warmup_done = False
        self._cycle_count = 0
        self.last_rejection_reason = "No signal generated"

    def generate_signal(self, market_data: MarketData) -> Optional[TradeSignal]:
        candles = market_data.m5_candles
        
        if candles is None:
            self.last_rejection_reason = "No m5 data"
            return None
        
        if len(candles) < 5:
            self.last_rejection_reason = f"Insufficient candles: {len(candles)} < 5"
            return None
        
        point = market_data.point
        
        current_price = market_data.bid
        spread = market_data.spread
        
        sl_distance = self.sl_points * point
        tp_distance = sl_distance * self.tp_mult
        
        last_candle = candles[-1]
        
        body = abs(last_candle.close - last_candle.open)
        candle_range = last_candle.high - last_candle.low
        if candle_range > 0:
            body_ratio = body / candle_range
        else:
            return None
        
        if body_ratio < self.min_body_ratio:
            self.last_rejection_reason = f"Body ratio {body_ratio:.2f} < {self.min_body_ratio}"
            return None
        
        is_bullish = last_candle.close > last_candle.open
        is_bearish = last_candle.close < last_candle.open
        
        if is_bullish:
            entry = current_price
            sl = entry - sl_distance
            tp = entry + tp_distance
            
            sig = TradeSignal(
                direction="BUY",
                price=entry,
                confidence=0.75,
                volume=0.05,
                reasons=[f"Momentum body={body_ratio:.2f}"]
            )
            
            self._last_setup = {"sl": sl, "tp": tp, "risk": sl_distance}
            self._cooldown = 3
            return sig
            
        elif is_bearish:
            entry = current_price + spread
            sl = entry + sl_distance
            tp = entry - tp_distance
            
            sig = TradeSignal(
                direction="SELL",
                price=entry,
                confidence=0.75,
                volume=0.05,
                reasons=[f"Momentum body={body_ratio:.2f}"]
            )
            
            self._last_setup = {"sl": sl, "tp": tp, "risk": sl_distance}
            self._cooldown = 3
            return sig
        
        self.last_rejection_reason = f"Body ratio too small (body={body_ratio:.2f} < {self.min_body_ratio})"
        return None

    def get_stop_loss(self, signal: TradeSignal, market_data: MarketData) -> float:
        return self._last_setup.get("sl", signal.price - self.sl_points * market_data.point)

    def get_take_profit(self, signal: TradeSignal, market_data: MarketData) -> float:
        return self._last_setup.get("tp", signal.price + self.sl_points * market_data.point * self.tp_mult)

    def on_trade_closed(self, trade_record: dict) -> None:
        self._last_setup = {}

    def get_metrics(self, market_data: MarketData) -> Dict[str, Any]:
        return {"Cooldown": self._cooldown, "Lot Size": self.fixed_lot}

    def get_thresholds(self) -> Dict[str, Any]:
        return {"Lot Size": self.fixed_lot, "Timeframe": "M5", "SL Points": self.sl_points}

    def get_parameter_grid(self) -> Dict[str, Any]:
        """Provides Walk-Forward optimization hyperparameter boundaries."""
        return {
            "min_body_ratio": [0.55, 0.60, 0.65],
            "sl_points": [3000, 5000, 7000]
        }

    def is_spread_safe(self, market_data: MarketData) -> bool: return True
    def is_volatility_safe(self, market_data: MarketData) -> bool: return True
    def check_mtf_consensus(self, market_data: MarketData) -> bool: return True
