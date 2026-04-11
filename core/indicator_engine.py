import pandas as pd
import numpy as np
import logging

logger = logging.getLogger("trading_bot.indicators")

class IndicatorEngine:
    """
    V5-INSIGNIA High-Performance Indicator Engine.
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
        features["ema_8"] = df['close'].ewm(span=8, adjust=False).mean().values
        features["ema_21"] = df['close'].ewm(span=21, adjust=False).mean().values
        features["ema_20"] = df['close'].ewm(span=20, adjust=False).mean().values
        features["ema_50"] = df['close'].ewm(span=50, adjust=False).mean().values
        features["ema_200"] = df['close'].ewm(span=200, adjust=False).mean().values
        
        # 2. Volatility (ATR)
        high_low = df['high'] - df['low']
        high_close = (df['high'] - df['close'].shift()).abs()
        low_close = (df['low'] - df['close'].shift()).abs()
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        # Calculate both 14 and 20 periods for different strategy/regime defaults
        features["atr_14"] = true_range.rolling(14).mean().values
        features["atr_20"] = true_range.rolling(20).mean().values
        
        # 3. Momentum (RSI) — Using Wilder's EMA to match types.py
        # ... (rest of momentum code)
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = (-delta.where(delta < 0, 0))
        avg_gain = gain.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
        with np.errstate(divide='ignore', invalid='ignore'):
            rs = avg_gain / avg_loss
            # Vectorized guard for RSI: where avg_loss is 0, RSI stays neutral or NaN handled
            features["rsi_14"] = (100 - (100 / (1 + rs))).fillna(50.0).values
        
        # 4. Bollinger Bands (calculates sma_20 as bb_mid)
        ma_20 = df['close'].rolling(20).mean()
        std_20 = df['close'].rolling(20).std()
        features["bb_upper"] = (ma_20 + (std_20 * 2)).values
        features["bb_lower"] = (ma_20 - (std_20 * 2)).values
        features["bb_mid"] = ma_20.values
        features["sma_20"] = ma_20.values # Alias for strategy volume checks
        
        # 5. Price Action & Volume
        features["body_size"] = (df['close'] - df['open']).abs().values
        features["vol_sma_20"] = df['tick_volume'].rolling(20).mean().values
        
        # 6. Trend Strength (ADX) — Institutional Gating
        # ... (rest of ADX code)
        plus_dm = (df['high'] - df['high'].shift(1)).clip(lower=0)
        minus_dm = (df['low'].shift(1) - df['low']).clip(lower=0)
        plus_dm = np.where((plus_dm > minus_dm) & (plus_dm > 0), plus_dm, 0)
        minus_dm = np.where((minus_dm > plus_dm) & (minus_dm > 0), minus_dm, 0)
        
        tr_smooth = true_range.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
        plus_dm_smooth = pd.Series(plus_dm).ewm(alpha=1/14, min_periods=14, adjust=False).mean()
        minus_dm_smooth = pd.Series(minus_dm).ewm(alpha=1/14, min_periods=14, adjust=False).mean()
        
        with np.errstate(divide='ignore', invalid='ignore'):
            plus_di = 100 * (plus_dm_smooth / tr_smooth)
            minus_di = 100 * (minus_dm_smooth / tr_smooth)
            # Guard for DX calculation to prevent division by zero sum
            di_sum = plus_di + minus_di
            dx = 100 * (plus_di - minus_di).abs() / di_sum.replace(0, np.nan)
        
        features["adx_14"] = dx.ffill().fillna(0).ewm(alpha=1/14, min_periods=14, adjust=False).mean().values
        
        return features
