import numpy as np
from typing import List, Dict, Any

class MarketRegime:
    """
    Classifies the current market environment into one of four regimes:
    - TRENDING: High efficiency ratio (directional move).
    - RANGING: Low efficiency ratio (choppy/sideways).
    - HIGH_VOLATILITY: ATR exceeds historical 75th percentile.
    - LOW_LIQUIDITY: Volume is significantly below 50-period average.
    """
    TRENDING = "TRENDING"
    RANGING = "RANGING"
    HIGH_VOLATILITY = "HIGH_VOLATILITY"
    LOW_LIQUIDITY = "LOW_LIQUIDITY"

    @staticmethod
    def classify(candles: Any, lookback: int = 50) -> str:
        """
        Analyzes candle data to determine the current market regime.
        
        Args:
            candles (CandleArray): Input candle data as a CandleArray.
            lookback (int): Number of candles to consider for structural analysis.
            
        Returns:
            str: One of the MarketRegime constants.
        """
        if len(candles) < lookback:
            return MarketRegime.RANGING

        closes = candles.close[-lookback:]
        highs = candles.high[-lookback:]
        lows = candles.low[-lookback:]
        volumes = candles.tick_volume[-lookback:]

        # 1. Volatility: ATR as % of price (normalized)
        # Using vectorized TR calculation for speed
        tr = np.maximum(highs[1:] - lows[1:],
                        np.maximum(np.abs(highs[1:] - closes[:-1]),
                                   np.abs(lows[1:] - closes[:-1])))
        atr = np.mean(tr[-14:])
        atr_pct = (atr / closes[-1]) * 100

        # Relative Volatility: Use 75th percentile of recent ATR% as baseline
        atr_series = (tr / closes[1:]) * 100
        atr_baseline = np.percentile(atr_series, 75)

        if atr_pct > atr_baseline:
            return MarketRegime.HIGH_VOLATILITY

        # 2. Low Liquidity: Compare recent volume (last 5 bars) to average volume (last 50 bars)
        avg_vol = np.mean(volumes)
        recent_vol = np.mean(volumes[-5:])
        if avg_vol > 0 and recent_vol < avg_vol * 0.5:
            return MarketRegime.LOW_LIQUIDITY

        # 3. Trend: Kaufman's Efficiency Ratio (ER)
        # ER = |net price change| / sum of absolute individual changes
        net_change = abs(closes[-1] - closes[0])
        sum_abs_changes = np.sum(np.abs(np.diff(closes)))
        efficiency_ratio = net_change / sum_abs_changes if sum_abs_changes > 0 else 0

        # If ER > 0.3, it is trending
        if efficiency_ratio > 0.3:
            return MarketRegime.TRENDING

        return MarketRegime.RANGING
