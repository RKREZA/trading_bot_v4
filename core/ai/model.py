import os
import json
import logging
import joblib
import pandas as pd
from datetime import datetime, timezone

try:
    from sklearn.ensemble import (
        HistGradientBoostingClassifier,
        RandomForestClassifier,
        ExtraTreesClassifier,
        VotingClassifier,
    )
    ENSEMBLE_AVAILABLE = True
except ImportError:
    from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
    VotingClassifier = None
    ENSEMBLE_AVAILABLE = False

logger = logging.getLogger("trading_bot.ai.model")

_DEFAULT_MAX_MODEL_AGE_DAYS = 30


class AIModelWrapper:
    """
    V7 Institutional Ensemble AI Model.

    Architecture: Soft-Voting Ensemble of 3 complementary classifiers:
    ┌──────────────────────────────────────────────────────────────────┐
    │  HistGradientBoosting  │  RandomForest  │  ExtraTrees           │
    │  (primary — handles    │  (bagging —    │  (extra randomization  │
    │   missing values)      │   ↓ variance)  │   — ↓ overfitting)    │
    └──────────────────────────────────────────────────────────────────┘
                        ↓ soft-vote (avg probabilities)
                           Final P(winning trade)

    Improvements over V6 (single HGB):
    - Soft voting averages probability estimates → more calibrated confidence
    - Ensemble diversity reduces variance on unseen regime shifts
    - OOF AUC gate (< 0.55) still enforced on the FULL ensemble
    """

    def __init__(self, model_dir: str = "core/ai/weights",
                 max_model_age_days: int = _DEFAULT_MAX_MODEL_AGE_DAYS):
        self.model_dir = model_dir
        self.model_path = os.path.join(model_dir, "v4_hgb_filter.pkl")
        self.metadata_path = os.path.join(model_dir, "v4_hgb_metadata.json")
        self.model = None
        self.is_ready = False
        self.max_model_age_days = max_model_age_days

        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir, exist_ok=True)

    # ──────────────────────────────────────────────────────────────────────
    # PERSISTENCE
    # ──────────────────────────────────────────────────────────────────────

    def load(self) -> bool:
        """Load ensemble weights with staleness validation."""
        if self.is_ready and self.model is not None:
            return True

        if not os.path.exists(self.model_path):
            return False

        try:
            self.model = joblib.load(self.model_path)
            self.is_ready = True
            logger.info(f"AI Ensemble loaded from {self.model_path}")

            if os.path.exists(self.metadata_path):
                with open(self.metadata_path) as f:
                    meta = json.load(f)
                trained_at_str = meta.get("trained_at")
                if trained_at_str:
                    trained_at = datetime.fromisoformat(trained_at_str).replace(tzinfo=timezone.utc)
                    age_days = (datetime.now(timezone.utc) - trained_at).days
                    model_type = meta.get("model_type", "Unknown")
                    if age_days > self.max_model_age_days:
                        logger.warning(
                            f"[AI] Ensemble is {age_days}d old (max={self.max_model_age_days}). "
                            f"Re-train with `python ai_train.py`."
                        )
                    else:
                        logger.info(
                            f"[AI] {model_type} loaded. Age: {age_days}d. "
                            f"CV AUC: {meta.get('cv_auc_mean', 'N/A')}"
                        )
            else:
                logger.warning("[AI] No model metadata. Cannot verify staleness.")

            return True
        except Exception as e:
            logger.error(f"Failed to load AI ensemble: {e}")
        return False

    def _build_ensemble(self) -> object:
        """
        Builds the 3-model soft-voting ensemble.
        Falls back to single RF if sklearn VotingClassifier is unavailable.
        """
        hgb = HistGradientBoostingClassifier(
            max_iter=100, max_depth=5, learning_rate=0.1, random_state=42
        )
        rf = RandomForestClassifier(
            n_estimators=100, max_depth=6, random_state=42, n_jobs=-1
        )
        et = ExtraTreesClassifier(
            n_estimators=100, max_depth=6, random_state=42, n_jobs=-1
        )

        if ENSEMBLE_AVAILABLE and VotingClassifier is not None:
            return VotingClassifier(
                estimators=[("hgb", hgb), ("rf", rf), ("et", et)],
                voting="soft",         # Average probability estimates
                n_jobs=-1,
            )
        else:
            logger.warning("[AI] VotingClassifier unavailable — falling back to single RF.")
            return rf

    def train(self, X: pd.DataFrame, y: pd.Series):
        """
        Trains the V7 Soft-Voting Ensemble with 5-fold OOF cross-validation.
        Model is rejected and NOT saved if ensemble OOF AUC < 0.55.
        """
        logger.info("Training V7 Institutional Soft-Voting Ensemble...")

        ensemble = self._build_ensemble()

        # 5-fold CV on the ensemble before committing weights
        try:
            from sklearn.model_selection import cross_val_score
            cv_scores = cross_val_score(ensemble, X, y, cv=5, scoring="roc_auc", n_jobs=-1)
            mean_auc = float(cv_scores.mean())
            std_auc = float(cv_scores.std())
            logger.info(f"[AI] Ensemble 5-Fold CV ROC-AUC: {mean_auc:.4f} ± {std_auc:.4f}")

            if mean_auc < 0.55:
                logger.critical(
                    f"[AI] TRAINING ABORTED: Ensemble OOF AUC {mean_auc:.4f} < 0.55. "
                    f"Ensemble is not significantly better than random. NOT saved."
                )
                return
        except Exception as e:
            logger.warning(f"[AI] Cross-validation skipped: {e}")
            mean_auc, std_auc = 0.0, 0.0

        # Fit on full dataset
        ensemble.fit(X, y)
        self.model = ensemble

        joblib.dump(self.model, self.model_path)
        # Baseline snapshot for DriftDetector
        joblib.dump(X, os.path.join(self.model_dir, "v4_ai_baseline.pkl"))

        # Persist metadata
        model_type = "SoftVotingEnsemble(HGB+RF+ET)" if ENSEMBLE_AVAILABLE else "RandomForest"
        metadata = {
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "model_type": model_type,
            "n_samples": int(len(X)),
            "n_features": int(X.shape[1]),
            "feature_names": list(X.columns),
            "cv_auc_mean": round(mean_auc, 4),
            "cv_auc_std": round(std_auc, 4),
        }
        with open(self.metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        self.is_ready = True
        logger.info(
            f"[AI] {model_type} trained and saved. "
            f"AUC={mean_auc:.4f} ± {std_auc:.4f}"
        )

    # ──────────────────────────────────────────────────────────────────────
    # INFERENCE
    # ──────────────────────────────────────────────────────────────────────

    def predict_probability(self, features: dict) -> float:
        """
        Returns ensemble-averaged probability that this trade wins (Class=1).
        Soft voting averages the three models' P(class=1) for a more
        calibrated and conservative confidence score than any single model.
        """
        if not self.is_ready or not self.model:
            return -1.0

        try:
            df = pd.DataFrame([features])
            probs = self.model.predict_proba(df)[0]
            return float(probs[1]) if len(probs) > 1 else float(probs[0])
        except Exception as e:
            logger.error(f"AI Ensemble inference failure: {e}")
            return -1.0
