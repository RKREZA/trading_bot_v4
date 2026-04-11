import os
import logging
import joblib
import pandas as pd
from typing import Dict

logger = logging.getLogger("trading_bot.ai.drift")

class DriftDetector:
    """
    Evaluates real-time execution feature matrices structurally against 
    the baseline static populations utilizing scipy.stats.ks_2samp.
    """
    def __init__(self, model_dir: str = "core/ai/weights"):
        self.baseline_path = os.path.join(model_dir, "v4_rf_baseline.pkl")
        self.baseline = None
        self.buffer = []
        self.buffer_size = 200  # Tracks last 200 feature evaluations natively
        self.is_ready = False
        
        self.is_drifted = False
        self.drift_p_value = 1.0

        # High priority features driving ML models
        self.target_features = ["atr_ratio", "spread_ratio", "adx"]

    def load_baseline(self) -> bool:
        if os.path.exists(self.baseline_path):
            try:
                self.baseline = joblib.load(self.baseline_path)
                self.is_ready = True
                logger.info("Baseline Population Matrix safely loaded for Drift Evaluation.")
                return True
            except Exception as e:
                logger.error(f"DriftDetector failed mapping baseline: {e}")
        return False

    def push_features(self, features: dict):
        """Append real-time features executing natively bounding statistical drift constraints."""
        if not self.is_ready or not self.baseline is not None:
            return

        self.buffer.append(features)
        
        if len(self.buffer) > self.buffer_size:
            self.buffer.pop(0)
            
        # Re-evaluate Native Structural Distribution occasionally
        if len(self.buffer) == self.buffer_size and len(self.buffer) % 50 == 0:
            self._evaluate_drift()

    def _evaluate_drift(self):
        """Quantitatively checks drift natively using Two-Sample Kolmogorov-Smirnov test."""
        try:
            from scipy.stats import ks_2samp
            
            live_df = pd.DataFrame(self.buffer)
            total_p = 0.0
            
            for feat in self.target_features:
                if feat in live_df.columns and feat in self.baseline.columns:
                    stat, p_value = ks_2samp(self.baseline[feat], live_df[feat])
                    total_p += p_value
            
            avg_p_value = total_p / len(self.target_features)
            self.drift_p_value = avg_p_value
            
            if avg_p_value < 0.05:
                # Severe structural drift detected natively blocking probabilities
                if not self.is_drifted:
                    logger.warning(f"CRITICAL: Feature Drift detected (P-Value {avg_p_value:.4f}). AI Filter degraded.")
                    self.is_drifted = True
            else:
                if self.is_drifted:
                    logger.info("Feature distribution recovered natively. Drift negated.")
                    self.is_drifted = False
                    
        except ImportError:
            pass # Scipy missing locally, inherently bypassing
        except Exception as e:
            logger.debug(f"Drift evaluation skipped intrinsically: {e}")
