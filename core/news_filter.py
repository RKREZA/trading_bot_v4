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
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept-Language': 'en-US,en;q=0.9',
            'Referer': 'https://www.forexfactory.com/calendar'
        })
        
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
        """Fetches news with advanced spoofing, DailyFX fallback, and manual overrides."""
        # 1. Check for Manual Override
        manual_file = "config/news_manual.json"
        if os.path.exists(manual_file):
            try:
                with open(manual_file, 'r') as f:
                    self.events = json.load(f)
                logger.info(f"News: [MANUAL OVERRIDE] Loaded {len(self.events)} events from {manual_file}.")
                return
            except Exception as e:
                logger.error(f"News: Failed to load manual override: {e}")

        # 2. DailyFX Institutional Fallback (Highly Stable)
        if self.fetch_from_dailyfx():
            return

        # 3. ForexFactory Mirror Strategy (Advanced Spoofing)
        urls = [
            "https://www.forexfactory.com/ff_calendar_thisweek.json",
            "https://nfs.forexfactory.com/ff_calendar_thisweek.json",
            "https://cdn-ffc.forexfactory.com/ff_calendar_thisweek.json"
        ]
        
        last_error = ""
        for url in urls:
            try:
                logger.info(f"News: Attempting {url}...")
                response = self.session.get(url, timeout=12)
                response.raise_for_status()
                
                # Verify JSON integrity
                try:
                    raw_data = response.json()
                except:
                    logger.warning(f"News: {url} returned non-JSON content. (Likely HTML block)")
                    continue

                # Filter for High Impact and relevant currencies
                filtered_events = []
                for event in raw_data:
                    if event.get("impact") in self.impact_levels:
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
                            logger.debug(f"News: Skip parsing malformed event: {e}")
                
                self.events = filtered_events
                
                # Save to cache
                os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
                with open(self.cache_file, 'w') as f:
                    json.dump(self.events, f, indent=2)
                    
                logger.info(f"News: Successfully fetched {len(self.events)} events using {url}")
                return # Exit on success
                
            except Exception as e:
                last_error = str(e)
                # Suppress DNS errors to trace (too noisy for dashboard)
                if "getaddrinfo failed" in last_error:
                    logger.debug(f"News: DNS resolve failed for {url}")
                else:
                    logger.warning(f"News: Source {url} failed: {last_error}")
        
        logger.error(f"News: All automated sources failed. System using 'Persistence Mode' (Stale cache).")

    def fetch_from_dailyfx(self) -> bool:
        """Fetches high-impact news from the DailyFX API (Stable Institutional Source)."""
        try:
            # Fetch current and next week
            url = "https://www.dailyfx.com/api/v1/calendar/events"
            logger.info("News: Attempting fetch from DailyFX...")
            response = self.session.get(url, timeout=12)
            response.raise_for_status()
            data = response.json()
            
            new_events = []
            for item in data:
                if item.get("importance") == "high":
                    # DailyFX timestamp is in milliseconds
                    ts = item.get("date", 0) / 1000.0
                    new_events.append({
                        "title": item.get("title"),
                        "country": item.get("countryCode", "").upper(),
                        "impact": "High",
                        "timestamp": ts
                    })
            
            if new_events:
                self.events = new_events
                self._save_cache()
                logger.info(f"News: Successfully fetched {len(new_events)} events from DailyFX.")
                return True
        except Exception as e:
            logger.warning(f"News: DailyFX source failed: {e}")
        return False

    def _save_cache(self):
        """Internal helper to persist events."""
        try:
            os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
            with open(self.cache_file, 'w') as f:
                json.dump(self.events, f, indent=2)
        except Exception as e:
            logger.error(f"News: Cache save failed: {e}")

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
