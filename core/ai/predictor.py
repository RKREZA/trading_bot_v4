import logging
from core.ai.features import FeatureEngineer
from core.ai.model import AIModelWrapper
from core.common.types import TradeSignal, CandleArray, FilteredSignal
from core.ai.drift import DriftDetector
from core.ai.sentiment import SentimentVetter
import copy

logger = logging.getLogger("trading_bot.ai.predictor")

class AIPredictor:
    """
    Main Interface orchestrating AI Feature Extraction and Prediction bounds.
    """
    def __init__(self, config: dict):
        self.config = config.get("ai_layer", {})
        self.enabled = self.config.get("enabled", False)
        self.threshold = self.config.get("confidence_threshold", 0.65)
        self.engine = AIModelWrapper()
        self.drift_layer = DriftDetector()
        self.sentiment_vetter = SentimentVetter(config)

        # Configurable blend weights (default: 60% model + 40% sentiment)
        self.model_weight = float(self.config.get("model_weight", 0.6))
        self.sentiment_weight = float(self.config.get("sentiment_weight", 0.4))
        
        if self.enabled:
            # Boot load
            self.engine.load()
            self.drift_layer.load_baseline()

    def filter_signal(self, base_signal: TradeSignal, candles: CandleArray, current_sl_pips: float, news_events: list = None) -> FilteredSignal:
        """
        Takes an active BaseStrategy Signal, processes its AI Confidence, 
        and evaluates if it clears threshold (via immutable returned wrapper).
        """
        # Short circuits
        if not self.enabled or not self.engine.is_ready:
            return FilteredSignal(original=base_signal, approved=True, confidence=1.0, comment="AI_DISABLED")
            
        if not base_signal or base_signal.direction == "NONE":
            return FilteredSignal(original=base_signal, approved=False, confidence=0.0, comment="NO_SIGNAL")
            
        try:
            # 1. Feature Map Extraction
            features = FeatureEngineer.extract_features(candles, base_signal.direction, current_sl_pips)
            if not features:
                return FilteredSignal(original=base_signal, approved=True, confidence=0.5, comment="FEATURE_ERR")
                
            # 1.5 Drift Tracking & Auto-Killswitch 
            self.drift_layer.push_features(features)
            if self.drift_layer.is_drifted:
                return FilteredSignal(original=base_signal, approved=True, confidence=0.0, comment="DRIFT_BYPASS")
                
            # 2. Probability Classification
            prob = self.engine.predict_probability(features)
            
            # 3. Gating Enforcement (Immutable Assignment)
            approved = True
            comment = "AI_APPROVED"
            
            if 0 <= prob < self.threshold:
                logger.info(f"AI Filter VETO: Sig={base_signal.direction} Prob={prob:.2f} < {self.threshold}")
                approved = False
                comment = "AI_VETO"
                
            # 4. Institutional Sentiment Vetting (NVIDIA NIM Reasoning)
            if approved and self.sentiment_vetter.enabled:
                vet_result = self.sentiment_vetter.vet_signal(
                    base_signal.symbol if hasattr(base_signal, 'symbol') else "UNKNOWN", 
                    base_signal.direction, 
                    news_events
                )
                if not vet_result["approved"]:
                    approved = False
                    comment = f"SENTIMENT_VETO: {vet_result.get('reason', 'Macro Mismatch')}"
                else:
                    # Blend confidence scores (Weighted average: configurable)
                    prob = (self.model_weight * prob) + (self.sentiment_weight * float(vet_result.get("confidence", 0.5)))
                
            return FilteredSignal(original=base_signal, approved=approved, confidence=prob, comment=comment)
            
        except Exception as e:
            # INSTITUTIONAL FAIL-CLOSED: Any crash in the AI/Sentiment pipeline
            # MUST block trading. Silent approval on error is a critical vulnerability
            # that could allow unvetted trades through during system instability.
            logger.error(f"AI Filter FAIL-CLOSED: {e}", exc_info=True)
            return FilteredSignal(original=base_signal, approved=False, confidence=0.0, comment="EVAL_ERR_SYSTEM_HALT")
