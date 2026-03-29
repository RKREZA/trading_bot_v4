import logging
import os
import json
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Tuple, Optional
import openai

logger = logging.getLogger("trading_bot.ai_filter")

class AIFilter:
    def __init__(self, config: Optional[dict] = None, threshold: Optional[float] = None):
        if config is None: config = {}
        # Handle both nested and flat config
        self.config_data = config.get("ai_advisor", {}) if "ai_advisor" in config else config
        self.enabled = self.config_data.get("enabled", False)
        self.threshold = threshold if threshold is not None else self.config_data.get("min_confidence", 0.7)
        self.backtest_mode = False # Set to True during backtests
        
        self.base_url = self.config_data.get("base_url", "https://integrate.api.nvidia.com/v1")
        self.api_key = self.config_data.get("api_key") or os.getenv("NVIDIA_API_KEY")
        self.model = self.config_data.get("model", "deepseek-ai/deepseek-v3.2")
        
        # Cache for news events to avoid per-signal API calls
        self.high_impact_events = []
        self.last_news_fetch = None
        self.silent = False

    def _fetch_news_context(self) -> List[str]:
        """Fetch high-impact news times for the current day."""
        if not self.api_key: return []
        
        now = datetime.now(timezone.utc)
        if self.last_news_fetch and (now - self.last_news_fetch) < timedelta(hours=4):
            return self.high_impact_events
            
        try:
            client = openai.OpenAI(api_key=self.api_key, base_url=self.base_url)
            prompt = "List high-impact economic news times for XAUUSD (Gold) today in UTC. Format: HH:MM - Event Name."
            
            response = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}]
            )
            content = response.choices[0].message.content
            events = []
            for line in content.split("\n"):
                if ":" in line:
                    try:
                        time_part = line.split("-")[0].strip()
                        if ":" in time_part: # Ensure it looks like HH:MM
                            events.append(time_part)
                    except: continue
            
            self.high_impact_events = events
            self.last_news_fetch = now
            return events
        except Exception as e:
            logger.error(f"AI News Fetch Error: {e}")
            return []

    def is_news_blocked(self, timestamp: datetime) -> bool:
        """Check if the current time is within +/- 30 mins of a high-impact event."""
        if not self.enabled or self.backtest_mode: return False
        events = self._fetch_news_context()
        if not events: return False
        
        current_time_str = timestamp.strftime("%H:%M")
        curr_h, curr_m = map(int, current_time_str.split(":"))
        curr_total = curr_h * 60 + curr_m
        
        for ev in events:
            try:
                ev_h, ev_m = map(int, ev.split(":"))
                ev_total = ev_h * 60 + ev_m
                if abs(curr_total - ev_total) <= 30:
                    return True
            except: continue
        return False

    def filter_signal(self, signal_data: dict) -> Tuple[bool, float, float]:
        """
        Evaluate a signal and return (is_valid, confidence_score, sl_buffer_adjustment).
        """
        if not self.enabled:
            return True, 1.0, 0.0

        timestamp = signal_data.get("timestamp", datetime.now(timezone.utc))
        if isinstance(timestamp, (int, float)):
            timestamp = datetime.fromtimestamp(timestamp, tz=timezone.utc)
            
        if self.is_news_blocked(timestamp):
            if not self.silent: logger.warning(f"AI Blocked trade @ {timestamp} due to news proximity.")
            return False, 0.0, 0.0

        # Simulate per-signal evaluation (placeholder for real LLM scoring)
        score = 0.8 # Default baseline
        sl_buffer = 0.0
        
        if signal_data.get("regime") == "TRENDING" and signal_data.get("atr", 0) > 1.5:
            sl_buffer = 0.2 # Add 0.2 * ATR
            
        return score >= self.threshold, score, sl_buffer
