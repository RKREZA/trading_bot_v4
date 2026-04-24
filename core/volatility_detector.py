import numpy as np
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger("trading_bot.volatility_detector")


class VolatilityLevel(Enum):
    EXTREME_LOW = "EXTREME_LOW"
    VERY_LOW = "VERY_LOW"
    LOW = "LOW"
    NORMAL = "NORMAL"
    HIGH = "HIGH"
    VERY_HIGH = "VERY_HIGH"
    EXTREME = "EXTREME"


@dataclass
class VolatilityAnalysis:
    level: VolatilityLevel
    ratio: float
    atr_value: float
    atr_percentile: float
    avg_true_range: float
    daily_range_pct: float
    recommended_strategy: str
    risk_multiplier: float
    signal_frequency_boost: float


class VolatilityDetector:
    """
    V5-INSIGNIA Advanced Volatility Detection System.
    
    Features:
    - Multi-timeframe volatility analysis (M5, M15, H1, D1)
    - ATR-based regime classification
    - Historical percentile ranking
    - Dynamic parameter recommendations
    """
    
    THRESHOLDS = {
        VolatilityLevel.EXTREME_LOW: 0.25,
        VolatilityLevel.VERY_LOW: 0.40,
        VolatilityLevel.LOW: 0.60,
        VolatilityLevel.NORMAL: 1.00,
        VolatilityLevel.HIGH: 1.50,
        VolatilityLevel.VERY_HIGH: 2.00,
        VolatilityLevel.EXTREME: 2.50,
    }
    
    def __init__(self, atr_period: int = 14, lookback: int = 100):
        self.atr_period = atr_period
        self.lookback = lookback
        self._atr_history = []
        
    def analyze(self, m5_candles, h1_candles=None, d1_candles=None) -> VolatilityAnalysis:
        """
        Comprehensive volatility analysis across timeframes.
        
        Args:
            m5_candles: M5 candle data
            h1_candles: Optional H1 candle data
            d1_candles: Optional D1 candle data
            
        Returns:
            VolatilityAnalysis with level, ratios, and recommendations
        """
        if m5_candles is None or len(m5_candles) < self.lookback:
            return self._default_analysis()
        
        atr_values = m5_candles.atr(self.atr_period)
        if len(atr_values) == 0:
            return self._default_analysis()
        
        current_atr = atr_values[-1]
        atr_history = atr_values[-self.lookback:]
        
        avg_atr = np.mean(atr_history)
        atr_std = np.std(atr_history)
        
        vol_ratio = current_atr / avg_atr if avg_atr > 0 else 1.0
        
        percentile = self._calculate_percentile(atr_history, current_atr)
        
        daily_range_pct = self._calculate_daily_range_pct(m5_candles)
        
        level = self._classify_level(vol_ratio)
        
        recommended_strategy = self._get_recommended_strategy(level)
        risk_multiplier = self._get_risk_multiplier(level)
        signal_boost = self._get_signal_frequency_boost(level)
        
        return VolatilityAnalysis(
            level=level,
            ratio=vol_ratio,
            atr_value=current_atr,
            atr_percentile=percentile,
            avg_true_range=avg_atr,
            daily_range_pct=daily_range_pct,
            recommended_strategy=recommended_strategy,
            risk_multiplier=risk_multiplier,
            signal_frequency_boost=signal_boost
        )
    
    def _classify_level(self, ratio: float) -> VolatilityLevel:
        """Classify volatility level based on ATR ratio."""
        if ratio <= self.THRESHOLDS[VolatilityLevel.EXTREME_LOW]:
            return VolatilityLevel.EXTREME_LOW
        elif ratio <= self.THRESHOLDS[VolatilityLevel.VERY_LOW]:
            return VolatilityLevel.VERY_LOW
        elif ratio <= self.THRESHOLDS[VolatilityLevel.LOW]:
            return VolatilityLevel.LOW
        elif ratio <= self.THRESHOLDS[VolatilityLevel.NORMAL]:
            return VolatilityLevel.NORMAL
        elif ratio <= self.THRESHOLDS[VolatilityLevel.HIGH]:
            return VolatilityLevel.HIGH
        elif ratio <= self.THRESHOLDS[VolatilityLevel.VERY_HIGH]:
            return VolatilityLevel.VERY_HIGH
        else:
            return VolatilityLevel.EXTREME
    
    def _calculate_percentile(self, values: np.ndarray, current: float) -> float:
        """Calculate what percentile the current value is in."""
        if len(values) == 0:
            return 50.0
        sorted_vals = np.sort(values)
        rank = np.searchsorted(sorted_vals, current)
        return (rank / len(sorted_vals)) * 100
    
    def _calculate_daily_range_pct(self, m5_candles) -> float:
        """Calculate average daily range as percentage of price."""
        limit = len(m5_candles)
        if limit < 288:
            return 0.0
        
        # Use institutional view properties (.h, .l, .c) to obey anti-lookahead limit
        highs = m5_candles.h
        lows = m5_candles.l
        closes = m5_candles.c
        
        daily_ranges = []
        # Calculate daily range for each 24h window available in the history
        # i is the end-boundary of the 24h window
        for i in range(288, limit + 1, 288):
            slice_h = highs[i-288:i]
            slice_l = lows[i-288:i]
            
            # Robustness guard: Ensure slices are non-empty
            if len(slice_h) == 0 or len(slice_l) == 0:
                continue
            
            try:
                day_high = np.max(slice_h)
                day_low = np.min(slice_l)
            except ValueError:
                continue
                
            day_start = closes[i-288]
            
            if day_start > 0:
                daily_range = (day_high - day_low) / day_start * 100
                daily_ranges.append(daily_range)
        
        return np.mean(daily_ranges) if daily_ranges else 0.0
    
    def _get_recommended_strategy(self, level: VolatilityLevel) -> str:
        """Get recommended strategy type for volatility level."""
        recommendations = {
            VolatilityLevel.EXTREME_LOW: "NO_TRADE",
            VolatilityLevel.VERY_LOW: "MEAN_REVERSION_CONSERVATIVE",
            VolatilityLevel.LOW: "MEAN_REVERSION",
            VolatilityLevel.NORMAL: "MOMENTUM_BREAKOUT",
            VolatilityLevel.HIGH: "MOMENTUM_BREAKOUT_AGGRESSIVE",
            VolatilityLevel.VERY_HIGH: "TREND_CATCHING",
            VolatilityLevel.EXTREME: "TREND_CATCHING_REVERSED",
        }
        return recommendations.get(level, "MOMENTUM_BREAKOUT")
    
    def _get_risk_multiplier(self, level: VolatilityLevel) -> float:
        """Get recommended risk multiplier for volatility level."""
        multipliers = {
            VolatilityLevel.EXTREME_LOW: 0.0,
            VolatilityLevel.VERY_LOW: 0.3,
            VolatilityLevel.LOW: 0.5,
            VolatilityLevel.NORMAL: 1.0,
            VolatilityLevel.HIGH: 1.2,
            VolatilityLevel.VERY_HIGH: 1.5,
            VolatilityLevel.EXTREME: 0.8,
        }
        return multipliers.get(level, 1.0)
    
    def _get_signal_frequency_boost(self, level: VolatilityLevel) -> float:
        """Get signal frequency adjustment factor."""
        boosts = {
            VolatilityLevel.EXTREME_LOW: 0.0,
            VolatilityLevel.VERY_LOW: 0.3,
            VolatilityLevel.LOW: 0.5,
            VolatilityLevel.NORMAL: 1.0,
            VolatilityLevel.HIGH: 1.2,
            VolatilityLevel.VERY_HIGH: 1.5,
            VolatilityLevel.EXTREME: 1.3,
        }
        return boosts.get(level, 1.0)
    
    def _default_analysis(self) -> VolatilityAnalysis:
        """Return default analysis when data is insufficient."""
        return VolatilityAnalysis(
            level=VolatilityLevel.NORMAL,
            ratio=1.0,
            atr_value=0.0,
            atr_percentile=50.0,
            avg_true_range=0.0,
            daily_range_pct=0.0,
            recommended_strategy="MOMENTUM_BREAKOUT",
            risk_multiplier=1.0,
            signal_frequency_boost=1.0
        )
    
    def get_atr_ratio(self, candles) -> float:
        """Quick ATR ratio calculation for simple use cases."""
        if candles is None or len(candles) < self.lookback:
            return 1.0
        
        atr_values = candles.atr(self.atr_period)
        if len(atr_values) == 0:
            return 1.0
        
        current_atr = atr_values[-1]
        avg_atr = np.mean(atr_values[-self.lookback:])
        
        return current_atr / avg_atr if avg_atr > 0 else 1.0


class VolatilityAdaptiveParameters:
    """
    Dynamic parameter adjustment based on volatility conditions.
    Returns optimized parameters for different volatility regimes.
    """
    
    @staticmethod
    def get_parameters_for_volatility(level: VolatilityLevel, strategy_type: str) -> Dict[str, Any]:
        """
        Get optimized parameters for a strategy based on volatility level.
        
        Args:
            level: Current volatility level
            strategy_type: Type of strategy (breakout, mean_reversion, trend)
            
        Returns:
            Dictionary of adjusted parameters
        """
        params = {
            "body_thresh": 0.55,
            "h1_strength_thresh": 0.45,
            "sl_atr": 1.5,
            "tp_atr": 6.0,
            "min_confidence": 0.60,
            "min_bars_between_signals": 20,
            "rsi_oversold": 30,
            "rsi_overbought": 70,
            "bb_std": 2.0,
        }
        
        if strategy_type == "breakout":
            params = VolatilityAdaptiveParameters._breakout_params(level)
        elif strategy_type == "mean_reversion":
            params = VolatilityAdaptiveParameters._mean_reversion_params(level)
        elif strategy_type == "trend":
            params = VolatilityAdaptiveParameters._trend_params(level)
        
        return params
    
    @staticmethod
    def _breakout_params(level: VolatilityLevel) -> Dict[str, Any]:
        """Get breakout strategy parameters for volatility level."""
        
        param_sets = {
            VolatilityLevel.EXTREME_LOW: {
                "body_thresh": 0.80,
                "h1_strength_thresh": 0.70,
                "sl_atr": 1.0,
                "tp_atr": 4.0,
                "min_confidence": 0.85,
                "min_bars_between_signals": 40,
            },
            VolatilityLevel.VERY_LOW: {
                "body_thresh": 0.70,
                "h1_strength_thresh": 0.60,
                "sl_atr": 1.2,
                "tp_atr": 5.0,
                "min_confidence": 0.80,
                "min_bars_between_signals": 30,
            },
            VolatilityLevel.LOW: {
                "body_thresh": 0.65,
                "h1_strength_thresh": 0.55,
                "sl_atr": 1.5,
                "tp_atr": 5.5,
                "min_confidence": 0.70,
                "min_bars_between_signals": 25,
            },
            VolatilityLevel.NORMAL: {
                "body_thresh": 0.55,
                "h1_strength_thresh": 0.45,
                "sl_atr": 1.5,
                "tp_atr": 6.0,
                "min_confidence": 0.60,
                "min_bars_between_signals": 20,
            },
            VolatilityLevel.HIGH: {
                "body_thresh": 0.50,
                "h1_strength_thresh": 0.40,
                "sl_atr": 2.0,
                "tp_atr": 7.0,
                "min_confidence": 0.55,
                "min_bars_between_signals": 15,
            },
            VolatilityLevel.VERY_HIGH: {
                "body_thresh": 0.45,
                "h1_strength_thresh": 0.35,
                "sl_atr": 2.5,
                "tp_atr": 8.0,
                "min_confidence": 0.50,
                "min_bars_between_signals": 10,
            },
            VolatilityLevel.EXTREME: {
                "body_thresh": 0.40,
                "h1_strength_thresh": 0.30,
                "sl_atr": 3.0,
                "tp_atr": 10.0,
                "min_confidence": 0.45,
                "min_bars_between_signals": 8,
            },
        }
        
        return param_sets.get(level, param_sets[VolatilityLevel.NORMAL])
    
    @staticmethod
    def _mean_reversion_params(level: VolatilityLevel) -> Dict[str, Any]:
        """Get mean reversion strategy parameters for volatility level."""
        
        param_sets = {
            VolatilityLevel.EXTREME_LOW: {
                "rsi_oversold": 35,
                "rsi_overbought": 65,
                "bb_std": 1.5,
                "sl_atr": 1.0,
                "tp_atr": 2.0,
                "min_confidence": 0.80,
                "min_bars_between_signals": 30,
            },
            VolatilityLevel.VERY_LOW: {
                "rsi_oversold": 32,
                "rsi_overbought": 68,
                "bb_std": 1.8,
                "sl_atr": 1.2,
                "tp_atr": 2.5,
                "min_confidence": 0.75,
                "min_bars_between_signals": 20,
            },
            VolatilityLevel.LOW: {
                "rsi_oversold": 30,
                "rsi_overbought": 70,
                "bb_std": 2.0,
                "sl_atr": 1.5,
                "tp_atr": 3.0,
                "min_confidence": 0.70,
                "min_bars_between_signals": 15,
            },
            VolatilityLevel.NORMAL: {
                "rsi_oversold": 30,
                "rsi_overbought": 70,
                "bb_std": 2.0,
                "sl_atr": 2.0,
                "tp_atr": 3.0,
                "min_confidence": 0.60,
                "min_bars_between_signals": 10,
            },
            VolatilityLevel.HIGH: {
                "rsi_oversold": 25,
                "rsi_overbought": 75,
                "bb_std": 2.5,
                "sl_atr": 2.5,
                "tp_atr": 4.0,
                "min_confidence": 0.55,
                "min_bars_between_signals": 8,
            },
            VolatilityLevel.VERY_HIGH: {
                "rsi_oversold": 20,
                "rsi_overbought": 80,
                "bb_std": 3.0,
                "sl_atr": 3.0,
                "tp_atr": 5.0,
                "min_confidence": 0.50,
                "min_bars_between_signals": 5,
            },
            VolatilityLevel.EXTREME: {
                "rsi_oversold": 15,
                "rsi_overbought": 85,
                "bb_std": 3.0,
                "sl_atr": 4.0,
                "tp_atr": 6.0,
                "min_confidence": 0.45,
                "min_bars_between_signals": 3,
            },
        }
        
        return param_sets.get(level, param_sets[VolatilityLevel.NORMAL])
    
    @staticmethod
    def _trend_params(level: VolatilityLevel) -> Dict[str, Any]:
        """Get trend following strategy parameters for volatility level."""
        
        param_sets = {
            VolatilityLevel.EXTREME_LOW: {
                "adx_threshold": 35,
                "rsi_period": 14,
                "sl_atr": 1.5,
                "tp_atr": 5.0,
                "min_confidence": 0.80,
                "min_bars_between_signals": 30,
            },
            VolatilityLevel.VERY_LOW: {
                "adx_threshold": 32,
                "rsi_period": 14,
                "sl_atr": 1.8,
                "tp_atr": 5.5,
                "min_confidence": 0.75,
                "min_bars_between_signals": 25,
            },
            VolatilityLevel.LOW: {
                "adx_threshold": 30,
                "rsi_period": 14,
                "sl_atr": 2.0,
                "tp_atr": 6.0,
                "min_confidence": 0.70,
                "min_bars_between_signals": 20,
            },
            VolatilityLevel.NORMAL: {
                "adx_threshold": 25,
                "rsi_period": 14,
                "sl_atr": 2.0,
                "tp_atr": 4.0,
                "min_confidence": 0.70,
                "min_bars_between_signals": 25,
            },
            VolatilityLevel.HIGH: {
                "adx_threshold": 25,
                "rsi_period": 14,
                "sl_atr": 2.5,
                "tp_atr": 5.0,
                "min_confidence": 0.65,
                "min_bars_between_signals": 20,
            },
            VolatilityLevel.VERY_HIGH: {
                "adx_threshold": 20,
                "rsi_period": 10,
                "sl_atr": 3.0,
                "tp_atr": 6.0,
                "min_confidence": 0.60,
                "min_bars_between_signals": 15,
            },
            VolatilityLevel.EXTREME: {
                "adx_threshold": 15,
                "rsi_period": 7,
                "sl_atr": 4.0,
                "tp_atr": 8.0,
                "min_confidence": 0.55,
                "min_bars_between_signals": 10,
            },
        }
        
        return param_sets.get(level, param_sets[VolatilityLevel.NORMAL])
