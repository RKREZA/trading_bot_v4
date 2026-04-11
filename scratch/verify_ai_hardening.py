import os
import json
import logging
from core.ai.sentiment import SentimentVetter
from core.ai.predictor import AIPredictor
from core.common.types import TradeSignal
from dotenv import load_dotenv

load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("VERIFY_AI")

def verify_ai_logic():
    print("\n[1] SENTIMENT VETTING TEST (NVIDIA NIM)")
    print("-" * 50)
    
    with open("config/config.json", "r") as f:
        config = json.load(f)
    
    vetter = SentimentVetter(config)
    
    # Mock data
    news = [
        {"title": "US CPI Higher Than Expected", "country": "USD", "impact": "High"},
        {"title": "Fed Official Hints at Rate Hike", "country": "USD", "impact": "High"}
    ]
    
    # Test a Buy against Hawkish US news (Should be contradicted or neutral with caution)
    result = vetter.vet_signal("XAUUSDm", "BUY", news)
    print(f"Signal: BUY XAUUSDm")
    print(f"Alignment: {result.get('bias')}")
    print(f"Approved:  {result.get('approved')}")
    print(f"Reason:    {result.get('reason')}")

    print("\n[2] PREDICTOR INTEGRATION TEST")
    print("-" * 50)
    predictor = AIPredictor(config)
    
    # Mock Candle Data (simplified)
    class MockCandles:
        def __init__(self):
            self.close = [1.0] * 50
            self.open = [1.0] * 50
            self.high = [1.05] * 50
            self.low = [0.95] * 50
            self.time = [1670000000] * 50
            self.spread = [0.0001] * 50
        def get_indicator(self, name):
            return [25.0] * 50 # Mock ADX
            
    candles = MockCandles()
    base_sig = TradeSignal(direction="BUY", price=4750.0)
    
    # This calls the full chain: Features -> ML Prob -> Sentiment Vetting
    filtered = predictor.filter_signal(base_sig, candles, 50.0, news_events=news)
    
    print(f"Final Approval: {filtered.approved}")
    print(f"Final Confidence: {filtered.confidence:.2f}")
    print(f"Comment: {filtered.comment}")

if __name__ == "__main__":
    if not os.getenv("NVIDIA_API_KEY"):
        print("SKIP: NVIDIA_API_KEY not found in .env. Cannot verify NIM logic.")
    else:
        verify_ai_logic()
