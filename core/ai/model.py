import os
import logging
import joblib
import pandas as pd

try:
    from sklearn.ensemble import RandomForestClassifier
except ImportError:
    RandomForestClassifier = None

logger = logging.getLogger("trading_bot.ai.model")

class AIModeWrapper:
    """Wrapper for Scikit-Learn Model persistence and prediction."""
    def __init__(self, model_dir: str = "core/ai/weights"):
        self.model_dir = model_dir
        self.model_path = os.path.join(model_dir, "v4_rf_filter.pkl")
        self.model = None
        self.is_ready = False
        
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir, exist_ok=True)

    def load(self) -> bool:
        """Loads serialized joblib model weights."""
        if not RandomForestClassifier:
            logger.error("Scikit-learn not installed. AI Layer disabled.")
            return False

        if os.path.exists(self.model_path):
            try:
                self.model = joblib.load(self.model_path)
                self.is_ready = True
                logger.info(f"AI Model loaded from {self.model_path}")
                return True
            except Exception as e:
                logger.error(f"Failed to load AI model: {e}")
        return False

    def train(self, X: pd.DataFrame, y: pd.Series):
        """Trains the Calibrated Random Forest on historical execution states."""
        if not RandomForestClassifier:
            return False
            
        logger.info("Training V4 Calibrated AI Probability Filter...")
        from sklearn.calibration import CalibratedClassifierCV
        
        base_model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42, class_weight='balanced')
        # Priority 1: Institutional Calibration (Isotonic Regression)
        self.model = CalibratedClassifierCV(estimator=base_model, method='isotonic', cv=5)
        self.model.fit(X, y)
        
        # Save Model and Baseline Features (Phase 3.7 Drift Tracking)
        joblib.dump(self.model, self.model_path)
        joblib.dump(X, os.path.join(self.model_dir, "v4_rf_baseline.pkl"))
        
        self.is_ready = True
        logger.info("Calibrated AI Model successfully trained and implicitly saved.")

    def predict_probability(self, features: dict) -> float:
        """Evaluates single feature dictionary and returns confidence for Class 1 (Winning Trade)."""
        if not self.is_ready or not self.model:
            return -1.0
            
        try:
            # Reconstruct dictionary into single row DataFrame matching training columns
            df = pd.DataFrame([features])
            
            # Predict probabilities [Prob(Loss), Prob(Win)]
            probs = self.model.predict_proba(df)[0]
            
            if len(probs) > 1:
                return probs[1] # Probability that trade is SUCCESSFUL (Class=1)
            else:
                return float(probs[0]) # Rare edge case in homogenous dataset
                
        except Exception as e:
            logger.error(f"AI Inference failure: {e}")
            return -1.0
