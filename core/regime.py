import numpy as np
from typing import List, Dict

class MarketRegime:
    TRENDING = "TRENDING"
    RANGING = "RANGING"
    HIGH_VOLATILITY = "HIGH_VOLATILITY"
    LOW_LIQUIDITY = "LOW_LIQUIDITY"

    @staticmethod
    def classify(candles: List[Dict], lookback: int = 50) -> str:
        if len(candles) < lookback:
            return MarketRegime.RANGING

        closes = np.array([c['close'] for c in candles[-lookback:]])
        highs = np.array([c['high'] for c in candles[-lookback:]])
        lows = np.array([c['low'] for c in candles[-lookback:]])
        volumes = np.array([c.get('tick_volume', 1) for c in candles[-lookback:]])

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
