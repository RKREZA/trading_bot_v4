import logging
import requests
import os
import json
from datetime import datetime
from typing import Dict, Any, Optional

logger = logging.getLogger("trading_bot.ai.sentiment")

class SentimentVetter:
    """
    Institutional Macro Reasoning Layer.
    Utilizes NVIDIA NIM (Inference Microservices) to vet trade direction 
    against high-impact news sentiment.
    """
    
    def __init__(self, config: dict):
        self.config = config.get("ai_layer", {}).get("sentiment_vetting", {})
        self.enabled = self.config.get("enabled", True)
        self.api_key = os.getenv("NVIDIA_API_KEY")
        self.model = self.config.get("model", "meta/llama3-70b-instruct")
        self.endpoint = "https://integrate.api.nvidia.com/v1/chat/completions"
        self.strict_mode = self.config.get("mode", "advisory") == "strict"
        
    def vet_signal(self, symbol: str, direction: str, news_events: list) -> Dict[str, Any]:
        """
        Queries NVIDIA NIM to determine if a signal (BUY/SELL) is macro-consistent.
        """
        if not self.enabled or not self.api_key:
            return {"approved": True, "confidence": 1.0, "reason": "DISABLED"}
            
        if not news_events:
            return {"approved": True, "confidence": 0.5, "reason": "NO_NEWS_DATA"}

        try:
            # 1. Prepare Prompt
            news_summary = "\n".join([
                f"- {e.get('title')} ({e.get('country')}) Impact: {e.get('impact')}" 
                for e in news_events[:5]
            ])
            
            prompt = f"""
            As an institutional macro analyst, evaluate a trade signal for {symbol}.
            
            SIGNAL: {direction}
            RECENT/UPCOMING NEWS:
            {news_summary}
            
            Determine if the {direction} signal is consistent with the macro sentiment or if high-impact news suggests caution.
            
            Format: JSON
            {{
                "alignment": "CONFIRMED" | "CONTRADICTED" | "NEUTRAL",
                "bias_score": 0.0 to 1.0,
                "reasoning": "brief explanation"
            }}
            """

            # 2. Call NVIDIA NIM
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.2,
                "max_tokens": 150,
                "response_format": {"type": "json_object"}
            }

            response = requests.post(self.endpoint, headers=headers, json=payload, timeout=10)
            
            if response.status_code != 200:
                logger.warning(f"NVIDIA NIM Error: {response.status_code} - {response.text}")
                return {"approved": True, "confidence": 0.5, "reason": "API_ERROR"}

            data = response.json()
            analysis_raw = data['choices'][0]['message']['content']
            analysis = json.loads(analysis_raw)
            
            # 3. Decision Logic
            bias = analysis.get("alignment", "NEUTRAL")
            score = float(analysis.get("bias_score", 0.5))
            
            approved = True
            if self.strict_mode and bias == "CONTRADICTED":
                approved = False
            
            logger.info(f"AI Sentiment Vetting: {symbol} {direction} -> {bias} (Score: {score})")
            
            return {
                "approved": approved,
                "confidence": score,
                "reason": analysis.get("reasoning", "NONE"),
                "bias": bias
            }

        except Exception as e:
            logger.error(f"Sentiment Vetting Exception: {e}")
            return {"approved": True, "confidence": 0.5, "reason": "RUNTIME_ERROR"}
