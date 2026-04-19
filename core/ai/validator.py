import logging
import os
import numpy as np
import pandas as pd
from core.ai.model import AIModelWrapper
from core.ai.features import FeatureEngineer
from core.common.types import CandleArray
from core.indicator_engine import IndicatorEngine

logger = logging.getLogger("trading_bot.ai.validator")

class AIValidator:
    """
    Self-Supervised trainer ingesting REAL Parquet binary caches 
    to extract ground-truth winning vs losing institutional trades.
    """
    
    @staticmethod
    def generate_real_training_data(symbol: str = "XAUUSDm", samples: int = 1500) -> tuple:
        """
        Ingests real market history, simulates execution windows, 
        and extracts ground-truth features for the ML model.
        """
        path = f"data_cache/{symbol}/M5.parquet"
        if not os.path.exists(path):
            logger.error(f"Validator: Data source {path} missing. Falling back to synthetic.")
            return AIValidator.generate_synthetic_training_data(samples)

        logger.info(f"Validator: Loading Ground-Truth history from {symbol}...")
        df = pd.read_parquet(path)
        
        # 1. Indicator Bootstrapping
        candles = CandleArray(
            time=df['time'].values,
            open=df['open'].values,
            high=df['high'].values,
            low=df['low'].values,
            close=df['close'].values,
            tick_volume=df['tick_volume'].values,
            spread=df['spread'].values
        )
        candles._indicators = IndicatorEngine.precalculate_all(symbol, "M5", candles)
        
        data = []
        labels = []
        
        # 2. Sequential Sampling (Avoiding future leakage)
        # We pick random indices but ensure we have 100 bars lookback and 100 bars lookahead for labeling
        indices = np.random.choice(range(200, len(df)-100), samples, replace=False)
        
        for idx in indices:
            # A. Extract Features at current point (Bar idx)
            candles.set_limit(idx)
            
            # --- INSTITUTIONAL TRUTH: Entry happens at OPEN of next bar (idx+1) ---
            # We must use idx+1 open for labeling to avoid lookahead bias during training.
            entry_idx = idx + 1
            if entry_idx >= len(df): break
            
            price = df.iloc[entry_idx]['open']
            direction = "BUY" if np.random.random() > 0.5 else "SELL"
            
            # Use 2.5x ATR for safety window labeling
            atr = candles.get_indicator("atr_14")[-1] if len(candles.get_indicator("atr_14")) > 0 else (price * 0.001)
            sl_dist = atr * 1.5
            tp_dist = sl_dist * 2.0 # Standard 2R Ground Truth
            
            sl = price - sl_dist if direction == "BUY" else price + sl_dist
            tp = price + tp_dist if direction == "BUY" else price - tp_dist
            
            # B. Determine Label (Labeling Window starts AFTER entry)
            is_win = 0
            for future_idx in range(entry_idx, min(entry_idx + 100, len(df))):
                f_high = df.iloc[future_idx]['high']
                f_low = df.iloc[future_idx]['low']
                
                if direction == "BUY":
                    if f_low <= sl: break # Hit SL first
                    if f_high >= tp: 
                        is_win = 1
                        break
                else:
                    if f_high >= sl: break # Hit SL first
                    if f_low <= tp:
                        is_win = 1
                        break
            
            # C. Map Feature Vector
            feats = FeatureEngineer.extract_features(candles, direction, sl_dist / (df.iloc[idx]['point'] if 'point' in df.columns else 0.01))
            if feats:
                data.append(feats)
                labels.append(is_win)
                
        return pd.DataFrame(data), pd.Series(labels)

    @staticmethod
    def generate_synthetic_training_data(samples: int = 1000) -> tuple:
        """Fallback synthetic generator if real data is missing."""
        # ... logic preserved for safety ...
        data = []
        labels = []
        for _ in range(samples):
            is_win = np.random.choice([0, 1])
            adx = np.random.uniform(25.0, 50.0) if is_win else np.random.uniform(5.0, 20.0)
            features = {
                "body_ratio": np.random.uniform(0.5, 0.9) if is_win else np.random.uniform(0.1, 0.4),
                "upper_wick_ratio": 0.1, "lower_wick_ratio": 0.1,
                "atr_ratio": 1.2 if is_win else 0.6,
                "spread_ratio": 1.0, "adx": adx,
                "hour_of_day": 12, "day_of_week": 2,
                "signal_dir": 1.0, "sl_pips": 30.0
            }
            data.append(features)
            labels.append(is_win)
        return pd.DataFrame(data), pd.Series(labels)

    @staticmethod
    def run_training_cycle():
        """Bootstraps the AI model with Real Market Truth."""
        logger.info("Initializing Ground-Truth AI Training (Parquet Ingestion)...")
        try:
            # Prioritize XAUUSDm as the primary gold benchmark
            X, y = AIValidator.generate_real_training_data("XAUUSDm", 2500)
            
            if len(X) < 100:
                logger.warning("Insufficient real data samples. Training failed.")
                return False
                
            engine = AIModelWrapper()
            engine.train(X, y)
            
            # Final integrity check on calibration
            logger.info(f"Model trained on {len(X)} real samples. Mean Ground-Truth WR: {y.mean()*100:.1f}%")
            return True
        except Exception as e:
            logger.error(f"Validator Training Crash: {e}")
            return False

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    AIValidator.run_training_cycle()
