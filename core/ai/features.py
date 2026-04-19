import numpy as np
import pandas as pd
from datetime import datetime
from core.common.types import CandleArray

class FeatureEngineer:
    """
    Extracts features from raw CandleArray and Strategy Signal state for the ML Model.
    """

    @staticmethod
    def extract_features(candles: CandleArray, signal_direction: str, current_sl_pips: float) -> dict:
        """
        Calculates standard ML features.
        """
        # Ensure we have enough data
        if len(candles.close) < 20:
            return {}

        features = {}
        
        # 1. Price Action (Body vs Wick)
        open_p = candles.open[-1]
        close_p = candles.close[-1]
        high_p = candles.high[-1]
        low_p = candles.low[-1]
        
        body_size = abs(open_p - close_p)
        upper_wick = high_p - max(open_p, close_p)
        lower_wick = min(open_p, close_p) - low_p
        total_size = high_p - low_p
        
        features["body_ratio"] = body_size / total_size if total_size > 0 else 0
        features["upper_wick_ratio"] = upper_wick / total_size if total_size > 0 else 0
        features["lower_wick_ratio"] = lower_wick / total_size if total_size > 0 else 0

        # 2. Volatility (ATR Ratio)
        # Assumes IndicatorEngine has populated atr_14
        atr_14_series = candles.get_indicator("atr_14")
        if len(atr_14_series) > 100:
            current_atr = atr_14_series[-1]
            avg_atr = np.mean(atr_14_series[-100:])
            features["atr_ratio"] = current_atr / avg_atr if avg_atr > 0 else 1.0
        else:
            features["atr_ratio"] = 1.0

        # 3. Spread (Current vs Mean)
        current_spread = candles.spread[-1]
        avg_spread = np.mean(candles.spread[-20:]) if len(candles.spread) >= 20 else current_spread
        features["spread_ratio"] = current_spread / avg_spread if avg_spread > 0 else 1.0

        # 4. Momentum (ADX)
        adx_series = candles.get_indicator("adx_14")
        if len(adx_series) > 0:
            features["adx"] = adx_series[-1]
        else:
            features["adx"] = 0.0

        # 5. Temporal
        timestamp = candles.time[-1]
        dt = datetime.utcfromtimestamp(timestamp)
        features["hour_of_day"] = dt.hour
        features["day_of_week"] = dt.weekday()

        # 6. Contextual
        features["signal_dir"] = 1.0 if signal_direction == "BUY" else -1.0
        features["sl_pips"] = current_sl_pips

        # 7. Institutional AR-3 Sequential Lags (New 10/10 Logic)
        # We capture the state of the 3 previous candles to provide sequence context
        for lag in range(1, 4):
            idx = -1 - lag
            if len(candles.close) > abs(idx):
                o, c, h, l = candles.open[idx], candles.close[idx], candles.high[idx], candles.low[idx]
                sz = h - l
                features[f"body_ratio_lag_{lag}"] = abs(o - c) / sz if sz > 0 else 0
                features[f"trend_dir_lag_{lag}"] = 1.0 if c > o else -1.0
                features[f"size_ratio_lag_{lag}"] = sz / total_size if total_size > 0 else 1.0
            else:
                features[f"body_ratio_lag_{lag}"] = 0
                features[f"trend_dir_lag_{lag}"] = 0
                features[f"size_ratio_lag_{lag}"] = 0

        return features
