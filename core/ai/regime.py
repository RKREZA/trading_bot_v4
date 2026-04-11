import logging
import numpy as np
import pandas as pd
from core.common.types import MarketRegime, CandleArray
try:
    from sklearn.cluster import KMeans
except ImportError:
    KMeans = None

logger = logging.getLogger("trading_bot.ai.regime")

class AIRegimeCluster:
    """
    Supplements the mathematics-based ADX framework by grouping 
    multidimensional Volatility, Volume, and Drift indicators.
    """
    def __init__(self):
        self.model = None
        self.is_ready = False
        self.centers = None

    def train_clusters(self, historical_df: pd.DataFrame):
        """Fit a KMeans model on historical features (ADX, ATR Ratio, Wick Ratio)"""
        if not KMeans:
            return False
            
        features = historical_df[['adx', 'atr_ratio', 'body_ratio']].dropna()
        if len(features) < 100:
            return False
            
        self.model = KMeans(n_clusters=3, random_state=42, n_init='auto')
        self.model.fit(features)
        
        self.centers = self.model.cluster_centers_
        self.is_ready = True
        logger.info("KMeans Regime Clustering built.")
        
    def classify_regime(self, adx: float, atr_ratio: float, body_ratio: float) -> str:
        """Categorize incoming real-time ticks into one of the ML clusters."""
        if not self.is_ready or not KMeans:
            return "UNKNOWN"
            
        try:
            # Predict
            vec = np.array([[adx, atr_ratio, body_ratio]])
            cluster_id = self.model.predict(vec)[0]
            
            # Map logical interpretation (this is simplified logic for cluster mapping)
            center = self.centers[cluster_id]
            
            # center[0] is ADX. High ADX + High Body = Trend
            if center[0] > 25 and center[2] > 0.6:
                return "TRENDING"
            elif center[1] > 1.5:
                # High ATR ratio
                return "HIGH_VOLATILITY"
            else:
                return "RANGING"
                
        except Exception:
            return "UNKNOWN"
