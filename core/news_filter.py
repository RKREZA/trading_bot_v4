import json
import logging
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional

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

    def _load_news(self) -> List[Dict[str, Any]]:
        """Loads high-impact news events (timestamp + title)."""
        try:
            with open(self.news_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data.get("high_impact", [])
        except (FileNotFoundError, json.JSONDecodeError):
            logger.warning(f"News: {self.news_file} not found. Operating without news guard.")
            return []

    def is_blocked(self, current_time: float, buffer_minutes: int = 30) -> bool:
        """Checks if current_time is within the buffer of any event."""
        if not self.high_impact_events:
            return False
            
        buffer_seconds = buffer_minutes * 60
        for event in self.high_impact_events:
            event_ts = int(event.get("timestamp", 0))
            if abs(current_time - event_ts) <= buffer_seconds:
                return True
        return False

    def get_next_event(self, current_time: float) -> Optional[Dict[str, Any]]:
        """Returns the next upcoming event from the schedule."""
        events = self.get_all_upcoming_events(current_time)
        return events[0] if events else None

    def get_all_upcoming_events(self, current_time: float) -> List[Dict[str, Any]]:
        """Returns all upcoming events sorted by proximity."""
        future_events = [e for e in self.high_impact_events if int(e.get("timestamp", 0)) > current_time - 1800]
        if not future_events:
            return []
            
        future_events.sort(key=lambda x: int(x.get("timestamp", 0)))
        
        results = []
        for ev in future_events:
            ts = int(ev.get("timestamp", 0))
            diff = ts - current_time
            is_active = abs(diff) <= 1800
            
            results.append({
                "title": ev.get("title", "Unknown"),
                "time": "ACTIVE NOW" if is_active else self._format_diff(diff),
                "is_active": is_active,
                "timestamp": ts
            })
        return results

    def _format_diff(self, seconds: float) -> str:
        if seconds < 0: return "Passed"
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"in {hours}h {minutes}m"
