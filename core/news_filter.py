import json
import logging
from datetime import datetime, timezone
from typing import List, Dict

logger = logging.getLogger("trading_bot.news")

class SimpleNewsFilter:
    """
    V4-ULTRA News Persistence Layer.
    Blocks trading activities during high-impact economic events.
    Fills the 'Institutional News Filter' gap identified in the audit.
    """
    
    def __init__(self, news_file: str = "config/news_events.json"):
        self.news_file = news_file
        self.high_impact_events = self._load_news()

    def _load_news(self) -> List[int]:
        """Loads high-impact news timestamps."""
        try:
            with open(self.news_file, 'r') as f:
                data = json.load(f)
                # We expect a list of Unix timestamps
                return [int(ts) for ts in data.get("high_impact", [])]
        except (FileNotFoundError, json.JSONDecodeError):
            logger.warning(f"News: {self.news_file} not found. Operating without news guard.")
            return []

    def is_blocked(self, current_time: float, buffer_minutes: int = 30) -> bool:
        """
        Returns True if the current_time is within the buffer of a high-impact event.
        Default: Blocks 30 mins before and after the event.
        """
        if not self.high_impact_events:
            return False
            
        buffer_seconds = buffer_minutes * 60
        for event_ts in self.high_impact_events:
            if abs(current_time - event_ts) <= buffer_seconds:
                return True
        return False
