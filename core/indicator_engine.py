import pandas as pd
import numpy as np
import logging

logger = logging.getLogger("trading_bot.indicators")

class IndicatorEngine:
    """
    V4-ULTRA High-Performance Indicator Engine.
    Pre-calculates all strategy features (RSI, ATR, BB, EMA) exactly once 
    before the backtest loop to eliminate the O(N^2) bottleneck.
    """
    
    @staticmethod
    def precalculate_all(symbol: str, timeframe: str, candles: 'CandleArray') -> dict:
        """
        Computes a comprehensive feature set for a given CandleArray.
        Returns a dictionary of numpy arrays (same length as candles).
        """
        logger.info(f"IPC: Pre-calculating features for {symbol} [{timeframe}]...")
        
        # Convert to pandas for vectorized speed
        df = candles.to_df()
        
        features = {}
        
        # 1. Trend & Averages
        features["ema_20"] = df['close'].ewm(span=20, adjust=False).mean().values
        features["ema_50"] = df['close'].ewm(span=50, adjust=False).mean().values
        features["ema_200"] = df['close'].ewm(span=200, adjust=False).mean().values
        
        # 2. Volatility (ATR)
        high_low = df['high'] - df['low']
        high_close = (df['high'] - df['close'].shift()).abs()
        low_close = (df['low'] - df['close'].shift()).abs()
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        features["atr_14"] = true_range.rolling(14).mean().values
        
        # 3. Momentum (RSI)
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        features["rsi_14"] = (100 - (100 / (1 + rs))).values
        
        # 4. Bollinger Bands
        ma_20 = df['close'].rolling(20).mean()
        std_20 = df['close'].rolling(20).std()
        features["bb_upper"] = (ma_20 + (std_20 * 2)).values
        features["bb_lower"] = (ma_20 - (std_20 * 2)).values
        features["bb_mid"] = ma_20.values
        
        # 5. Price Action
        features["body_size"] = (df['close'] - df['open']).abs().values
        
        # Institutional Mitigation: Keep NaNs for initial bars instead of Zero
        # This prevents strategies (e.g. RSI < 30) from triggering false positives 
        # on the incomplete first few bars of the dataset.
        return features
