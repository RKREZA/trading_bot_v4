import json
import logging
import time
import os
import requests
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any, Optional

logger = logging.getLogger("trading_bot.news")

class InstitutionalNewsFilter:
    """
    V4-ULTRA Institutional News Protection System.
    Dynamically fetches high-impact economic events and enforces trading blocks.
    """
    
    def __init__(self, config: dict):
        self.config = config
        self.news_cfg = config.get("news_filter", {})
        self.enabled = self.news_cfg.get("enabled", True)
        self.cache_file = self.news_cfg.get("cache_file", "config/news_cache.json")
        self.source_url = self.news_cfg.get("source_url", "https://nfs.forexfactory.com/ff_calendar_thisweek.json")
        self.impact_levels = self.news_cfg.get("impact_levels", ["High"])
        self.buffer_before = self.news_cfg.get("buffer_before_min", 30)
        self.buffer_after = self.news_cfg.get("buffer_after_min", 15)
        
        self.events: List[Dict[str, Any]] = []
        self._load_events()

    def _load_events(self):
        """Loads events from cache or fetches new ones if expired."""
        if not self.enabled:
            return

        should_fetch = True
        if os.path.exists(self.cache_file):
            mtime = os.path.getmtime(self.cache_file)
            # If cache is less than 24 hours old
            if time.time() - mtime < 86400:
                try:
                    with open(self.cache_file, 'r') as f:
                        self.events = json.load(f)
                    logger.info(f"News: Loaded {len(self.events)} events from cache.")
                    should_fetch = False
                except Exception as e:
                    logger.warning(f"News: Failed to load cache: {e}")

        if should_fetch:
            self.fetch_news()

    def fetch_news(self):
        """Fetches news from the dynamic source."""
        try:
            logger.info(f"News: Fetching from {self.source_url}")
            response = requests.get(self.source_url, timeout=10, headers={'User-Agent': 'Mozilla/5.0'})
            response.raise_for_status()
            raw_data = response.json()
            
            # Filter for High Impact and relevant currencies
            filtered_events = []
            for event in raw_data:
                if event.get("impact") in self.impact_levels:
                    # Parse date: "2024-03-07T08:30:00-05:00"
                    # We convert everything to UTC timestamp
                    try:
                        dt = datetime.fromisoformat(event["date"])
                        utc_ts = dt.timestamp()
                        
                        filtered_events.append({
                            "title": event.get("title"),
                            "country": event.get("country"),
                            "impact": event.get("impact"),
                            "timestamp": utc_ts
                        })
                    except Exception as e:
                        logger.error(f"News: Error parsing date {event.get('date')}: {e}")
            
            self.events = filtered_events
            
            # Save to cache
            os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
            with open(self.cache_file, 'w') as f:
                json.dump(self.events, f, indent=2)
                
            logger.info(f"News: Successfully fetched and cached {len(self.events)} high-impact events.")
            
        except Exception as e:
            logger.error(f"News: Fetch failed: {e}. Using stale cache if available.")

    def is_blocked(self, symbol: str, current_time: float) -> Optional[str]:
        """
        Checks if trading is blocked for a specific symbol.
        Returns the event name if blocked, else None.
        """
        if not self.enabled or not self.events:
            return None
            
        # Extract currencies from symbol (e.g., XAUUSD -> XAU, USD)
        affected_currencies = []
        if len(symbol) >= 6:
            # Common patterns: EURUSD, XAUUSDm, GBPJPY.pro
            # We take first 3 and next 3
            affected_currencies = [symbol[0:3], symbol[3:6]]
        
        # Buffer range
        before_sec = self.buffer_before * 60
        after_sec = self.buffer_after * 60
        
        for event in self.events:
            event_ts = event["timestamp"]
            event_country = event["country"]
            
            # Check if event affects our symbol's currencies
            if event_country in affected_currencies or event_country == "ALL":
                if (current_time >= event_ts - before_sec) and (current_time <= event_ts + after_sec):
                    return event["title"]
                    
        return None

    def get_upcoming_events(self, current_time: float, window_hours: int = 24) -> List[Dict[str, Any]]:
        """Returns events happening in the next X hours."""
        window_sec = window_hours * 3600
        upcoming = [
            e for e in self.events 
            if current_time <= e["timestamp"] <= current_time + window_sec
        ]
        return sorted(upcoming, key=lambda x: x["timestamp"])

    def get_auto_close_targets(self, current_time: float) -> List[str]:
        """
        Returns a list of currencies for which positions should be closed 
        due to proximity to a high-impact event.
        """
        auto_close_buffer = self.config.get("news_filter", {}).get("auto_close_before_min", 5) * 60
        targets = []
        
        for event in self.events:
            event_ts = event["timestamp"]
            # If event is in the next 'auto_close_before_min' minutes
            if 0 < (event_ts - current_time) <= auto_close_buffer:
                targets.append(event["country"])
                
        return list(set(targets))
