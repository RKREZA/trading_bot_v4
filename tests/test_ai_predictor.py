"""
test_ai_predictor.py — AI & Sentiment Filter Test Suite
=======================================================
Proves fail-closed architecture, sentiment blending, drift bypass,
and AI-disabled passthrough behavior.

V5-INSIGNIA Institutional Certification.
"""

import pytest
from unittest.mock import MagicMock, patch, PropertyMock
from core.common.types import TradeSignal, FilteredSignal, CandleArray
from core.ai.predictor import AIPredictor
import numpy as np


# ============================================================================
# HELPERS
# ============================================================================

def _make_base_signal(direction="BUY"):
    """Creates a valid TradeSignal for testing."""
    return TradeSignal(
        direction=direction,
        price=2000.0,
        confidence=0.75,
        stop_loss=1995.0,
        take_profit=2010.0,
    )


def _make_candles(n=100):
    """Creates a minimal CandleArray for feature extraction."""
    t = (np.arange(n) * 300 + 1700000000).astype(np.int64)
    close = np.full(n, 2000.0)
    return CandleArray(
        time=t,
        open=close - 0.5,
        high=close + 1.5,
        low=close - 1.5,
        close=close,
        tick_volume=np.full(n, 300, dtype=np.int64),
        spread=np.full(n, 15, dtype=np.int64),
    )


# ============================================================================
# 1. FAIL-CLOSED MECHANISM
# ============================================================================

class TestFailClosed:
    """
    Proves that ANY crash in the AI pipeline blocks trading
    instead of silently approving trades.
    """

    def test_fail_closed_on_feature_crash(self):
        """
        Exception in FeatureEngineer.extract_features → approved=False, confidence=0.0.
        """
        config = {
            "ai_layer": {
                "enabled": True,
                "confidence_threshold": 0.65,
                "sentiment_vetting": {"enabled": False},
            }
        }

        predictor = AIPredictor(config)
        predictor.engine = MagicMock()
        predictor.engine.is_ready = True

        # Force feature extraction to crash
        with patch("core.ai.predictor.FeatureEngineer.extract_features", side_effect=RuntimeError("Feature Crash")):
            result = predictor.filter_signal(_make_base_signal(), _make_candles(), 5.0)

        assert result.approved is False, "FAIL-CLOSED: Must block trade on feature crash"
        assert result.confidence == 0.0
        assert "EVAL_ERR_SYSTEM_HALT" in result.comment

    def test_fail_closed_on_model_crash(self):
        """
        Exception in AIModeWrapper.predict_probability → approved=False.
        """
        config = {
            "ai_layer": {
                "enabled": True,
                "confidence_threshold": 0.65,
                "sentiment_vetting": {"enabled": False},
            }
        }

        predictor = AIPredictor(config)
        predictor.engine = MagicMock()
        predictor.engine.is_ready = True

        # Feature extraction succeeds
        with patch("core.ai.predictor.FeatureEngineer.extract_features", return_value={"atr_ratio": 1.0}):
            # Model crash
            predictor.engine.predict_probability.side_effect = RuntimeError("Model Inference Crash")
            result = predictor.filter_signal(_make_base_signal(), _make_candles(), 5.0)

        assert result.approved is False, "FAIL-CLOSED: Must block trade on model crash"
        assert result.confidence == 0.0
        assert "EVAL_ERR_SYSTEM_HALT" in result.comment

    def test_fail_closed_on_sentiment_crash(self):
        """
        Exception in SentimentVetter.vet_signal → approved=False.
        """
        config = {
            "ai_layer": {
                "enabled": True,
                "confidence_threshold": 0.65,
                "sentiment_vetting": {"enabled": True, "model": "test"},
            }
        }

        predictor = AIPredictor(config)
        predictor.engine = MagicMock()
        predictor.engine.is_ready = True

        with patch("core.ai.predictor.FeatureEngineer.extract_features", return_value={"atr_ratio": 1.0}):
            predictor.engine.predict_probability.return_value = 0.80  # Above threshold
            predictor.drift_layer = MagicMock()
            predictor.drift_layer.is_drifted = False
            predictor.drift_layer.push_features = MagicMock()

            # Sentiment crash
            predictor.sentiment_vetter = MagicMock()
            predictor.sentiment_vetter.enabled = True
            predictor.sentiment_vetter.vet_signal.side_effect = RuntimeError("Sentiment API Down")

            result = predictor.filter_signal(_make_base_signal(), _make_candles(), 5.0)

        assert result.approved is False, "FAIL-CLOSED: Must block trade on sentiment crash"
        assert "EVAL_ERR_SYSTEM_HALT" in result.comment


# ============================================================================
# 2. SENTIMENT BLENDING
# ============================================================================

class TestSentimentBlending:
    """Verifies the weighted blend: 60% Model + 40% Sentiment."""

    def test_sentiment_blend_pass(self):
        """
        Model=0.70 (above threshold 0.65), Sentiment=0.8
        → blended = 0.6*0.70 + 0.4*0.8 = 0.42 + 0.32 = 0.74.
        This should PASS the threshold and the blended confidence should be 0.74.
        
        NOTE: Sentiment blending only occurs AFTER the model passes the
        threshold gate. If model prob < threshold, AI_VETO fires first.
        """
        config = {
            "ai_layer": {
                "enabled": True,
                "confidence_threshold": 0.65,
                "sentiment_vetting": {"enabled": True},
            }
        }

        predictor = AIPredictor(config)
        predictor.engine = MagicMock()
        predictor.engine.is_ready = True

        with patch("core.ai.predictor.FeatureEngineer.extract_features", return_value={"atr_ratio": 1.0}):
            predictor.engine.predict_probability.return_value = 0.70  # Above threshold → blending occurs
            predictor.drift_layer = MagicMock()
            predictor.drift_layer.is_drifted = False
            predictor.drift_layer.push_features = MagicMock()

            # Sentiment approves with high confidence
            predictor.sentiment_vetter = MagicMock()
            predictor.sentiment_vetter.enabled = True
            predictor.sentiment_vetter.vet_signal.return_value = {
                "approved": True,
                "confidence": 0.8,
                "reason": "MACRO_ALIGNED",
            }

            signal = _make_base_signal()
            result = predictor.filter_signal(signal, _make_candles(), 5.0)

        # Blended: 0.6 * 0.70 + 0.4 * 0.8 = 0.42 + 0.32 = 0.74
        assert result.approved is True, "Blended 0.74 should pass threshold 0.65"
        assert result.confidence == pytest.approx(0.74, abs=0.01)

    def test_sentiment_blend_fail(self):
        """
        Model=0.5, Sentiment=0.4 → blended = 0.6*0.5 + 0.4*0.4 = 0.46.
        With threshold=0.65, this should FAIL (AI_VETO on the model prob first).
        """
        config = {
            "ai_layer": {
                "enabled": True,
                "confidence_threshold": 0.65,
                "sentiment_vetting": {"enabled": True},
            }
        }

        predictor = AIPredictor(config)
        predictor.engine = MagicMock()
        predictor.engine.is_ready = True

        with patch("core.ai.predictor.FeatureEngineer.extract_features", return_value={"atr_ratio": 1.0}):
            predictor.engine.predict_probability.return_value = 0.5  # Below threshold
            predictor.drift_layer = MagicMock()
            predictor.drift_layer.is_drifted = False
            predictor.drift_layer.push_features = MagicMock()

            result = predictor.filter_signal(_make_base_signal(), _make_candles(), 5.0)

        # Model prob 0.5 < threshold 0.65 → AI_VETO
        assert result.approved is False, "Model prob 0.5 < 0.65 should be vetoed"
        assert "AI_VETO" in result.comment


# ============================================================================
# 3. AI DISABLED / DRIFT BYPASS
# ============================================================================

class TestAIBypass:
    """Verifies passthrough and drift behaviors."""

    def test_ai_disabled_passthrough(self):
        """
        When AI layer is disabled, all signals pass with confidence=1.0.
        """
        config = {
            "ai_layer": {
                "enabled": False,
                "confidence_threshold": 0.65,
                "sentiment_vetting": {"enabled": False},
            }
        }

        predictor = AIPredictor(config)
        result = predictor.filter_signal(_make_base_signal(), _make_candles(), 5.0)

        assert result.approved is True
        assert result.confidence == 1.0
        assert result.comment == "AI_DISABLED"

    def test_drift_bypass(self):
        """
        When drift is detected, the signal should return DRIFT_BYPASS
        with confidence=0.0.
        """
        config = {
            "ai_layer": {
                "enabled": True,
                "confidence_threshold": 0.65,
                "sentiment_vetting": {"enabled": False},
            }
        }

        predictor = AIPredictor(config)
        predictor.engine = MagicMock()
        predictor.engine.is_ready = True

        with patch("core.ai.predictor.FeatureEngineer.extract_features", return_value={"atr_ratio": 1.0}):
            predictor.drift_layer = MagicMock()
            predictor.drift_layer.is_drifted = True
            predictor.drift_layer.push_features = MagicMock()

            result = predictor.filter_signal(_make_base_signal(), _make_candles(), 5.0)

        assert result.comment == "DRIFT_BYPASS"
        assert result.confidence == 0.0

    def test_no_signal_rejected(self):
        """
        A NONE-direction signal should be immediately rejected.
        """
        config = {
            "ai_layer": {
                "enabled": True,
                "confidence_threshold": 0.65,
                "sentiment_vetting": {"enabled": False},
            }
        }

        predictor = AIPredictor(config)
        predictor.engine = MagicMock()
        predictor.engine.is_ready = True

        none_signal = TradeSignal(direction="NONE")
        result = predictor.filter_signal(none_signal, _make_candles(), 5.0)

        assert result.approved is False
        assert result.comment == "NO_SIGNAL"
