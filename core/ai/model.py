import os
import logging
import joblib
import pandas as pd

try:
    from sklearn.ensemble import HistGradientBoostingClassifier
    GRADIENT_BOOSTING_AVAILABLE = True
except ImportError:
    from sklearn.ensemble import RandomForestClassifier
    GRADIENT_BOOSTING_AVAILABLE = False

logger = logging.getLogger("trading_bot.ai.model")

class AIModeWrapper:
    """Wrapper for Scikit-Learn Model persistence and prediction."""
    def __init__(self, model_dir: str = "core/ai/weights"):
        self.model_dir = model_dir
        self.model_path = os.path.join(model_dir, "v4_hgb_filter.pkl") # Updated path for HGB
        self.model = None
        self.is_ready = False
        
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir, exist_ok=True)

    def load(self) -> bool:
        """Loads serialized joblib model weights."""
        if self.is_ready and self.model is not None:
            return True

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
        """Trains the Advanced Gradient Boosting model on historical execution states."""
        logger.info("Training V4 Institutional Gradient Boosting Filter...")
        
        if GRADIENT_BOOSTING_AVAILABLE:
            # High-performance histogram-based gradient boosting
            self.model = HistGradientBoostingClassifier(
                max_iter=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            )
        else:
            # Fallback to Random Forest if scikit-learn is old
            from sklearn.ensemble import RandomForestClassifier
            self.model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
            
        self.model.fit(X, y)
        
        # Save Model and Baseline Features
        joblib.dump(self.model, self.model_path)
        joblib.dump(X, os.path.join(self.model_dir, "v4_ai_baseline.pkl"))
        
        self.is_ready = True
        logger.info(f"AI Model ({'HistGradientBoosting' if GRADIENT_BOOSTING_AVAILABLE else 'RandomForest'}) trained and saved.")

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
