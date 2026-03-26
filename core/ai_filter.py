import random

class AIFilter:
    def __init__(self, threshold: float = 0.75):
        self.threshold = threshold

    def get_probability_score(self, signal_data: dict) -> float:
        """
        Simulate AI filtering. In a real system, this would call a model.
        For this task, we simulate a score that favors high-confidence strategy signals.
        """
        base_score = random.uniform(0.4, 0.9)
        # In a real implementation, this would involve feature extraction and model inference.
        # Here we just provide a placeholder that adheres to the 0-1 range.
        return base_score

    def filter_signal(self, signal_data: dict) -> bool:
        score = self.get_probability_score(signal_data)
        return score >= self.threshold, score
