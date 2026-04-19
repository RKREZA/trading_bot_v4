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
    
    INSTITUTIONAL HARDENING (v6):
    - Fixed double-negation logic bug in push_features guard
    - Counter-based re-evaluation every 50 pushes (replaces broken modular check)
    - Warns on first scipy ImportError instead of silent bypass
    """
    def __init__(self, model_dir: str = "core/ai/weights"):
        # Path matches what AIModeWrapper.train() saves as the baseline reference.
        self.baseline_path = os.path.join(model_dir, "v4_ai_baseline.pkl")
        self.baseline = None
        self.buffer = []
        self.buffer_size = 200  # Tracks last 200 feature evaluations natively
        self.is_ready = False
        
        self.is_drifted = False
        self.drift_p_value = 1.0
        self._push_count = 0  # Counter for periodic re-evaluation
        self._scipy_warned = False  # Prevent log spam for missing scipy

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
        """Append real-time features and periodically evaluate drift."""
        # FIXED: Clear guard logic — both conditions must be met to proceed
        if not self.is_ready or self.baseline is None:
            return

        self.buffer.append(features)
        self._push_count += 1
        
        if len(self.buffer) > self.buffer_size:
            self.buffer.pop(0)
            
        # FIXED: Counter-based re-evaluation every 50 pushes once buffer is warm.
        # The old check `len % 50 == 0` only fired once at exactly buffer_size.
        if len(self.buffer) >= self.buffer_size and self._push_count % 50 == 0:
            self._evaluate_drift()

    def _evaluate_drift(self):
        """Quantitatively checks drift natively using Two-Sample Kolmogorov-Smirnov test."""
        try:
            from scipy.stats import ks_2samp
            
            live_df = pd.DataFrame(self.buffer)
            total_p = 0.0
            matched_features = 0
            
            for feat in self.target_features:
                if feat in live_df.columns and feat in self.baseline.columns:
                    stat, p_value = ks_2samp(self.baseline[feat], live_df[feat])
                    total_p += p_value
                    matched_features += 1
            
            # Guard: avoid division by zero if no features matched
            if matched_features == 0:
                logger.warning("Drift evaluation skipped: No matching features between live and baseline.")
                return

            avg_p_value = total_p / matched_features
            self.drift_p_value = avg_p_value
            
            if avg_p_value < 0.05:
                # Severe structural drift detected
                if not self.is_drifted:
                    logger.warning(f"CRITICAL: Feature Drift detected (P-Value {avg_p_value:.4f}). AI Filter degraded.")
                    self.is_drifted = True
            else:
                if self.is_drifted:
                    logger.info("Feature distribution recovered natively. Drift negated.")
                    self.is_drifted = False
                    
        except ImportError:
            # FIXED: Warn on first occurrence instead of silent bypass
            if not self._scipy_warned:
                logger.warning("scipy is not installed. DriftDetector KS-test evaluation is disabled.")
                self._scipy_warned = True
        except Exception as e:
            logger.debug(f"Drift evaluation skipped: {e}")
