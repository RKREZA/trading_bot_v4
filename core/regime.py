import numpy as np
from typing import List, Dict

class MarketRegime:
    TRENDING = "TRENDING"
    RANGING = "RANGING"
    HIGH_VOLATILITY = "HIGH_VOLATILITY"
    LOW_LIQUIDITY = "LOW_LIQUIDITY"

    @staticmethod
    def classify(candles: List[Dict]) -> str:
        if len(candles) < 20:
            return MarketRegime.RANGING

        closes = np.array([c['close'] for c in candles[-20:]])
        highs = np.array([c['high'] for c in candles[-20:]])
        lows = np.array([c['low'] for c in candles[-20:]])
        
        # Volatility check (using ATR or standard deviation)
        returns = np.diff(np.log(closes))
        volatility = np.std(returns)
        
        # Simple trend detection (ADX or EMA slope would be better, but let's keep it robust)
        sma20 = np.mean(closes)
        current_price = closes[-1]
        
        # Liquidity / Volume check (if available)
        avg_volume = np.mean([c.get('tick_volume', 0) for i, c in enumerate(candles[-20:])])
        current_volume = candles[-1].get('tick_volume', 0)
        
        if volatility > np.mean(returns) * 5: 
            return MarketRegime.HIGH_VOLATILITY
            
        if current_volume < avg_volume * 0.1: # Relaxed from 0.2
            return MarketRegime.LOW_LIQUIDITY
            
        # Trending if price is far from SMA (Ultra-Aggressive: 0.01%)
        if abs(current_price - sma20) / sma20 > 0.0001:
            return MarketRegime.TRENDING
            
        return MarketRegime.RANGING
