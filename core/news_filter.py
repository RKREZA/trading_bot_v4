import json
import logging
import time
import os
import requests
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any, Optional, Set
from enum import Enum
from collections import defaultdict
from dataclasses import dataclass, field

logger = logging.getLogger("trading_bot.news")

class ImpactLevel(Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

class NewsResilience(Enum):
    HEALTHY = "HEALTHY"
    DEGRADED = "DEGRADED"
    STALE = "STALE"
    LOCKOUT = "LOCKOUT"

@dataclass
class NewsEvent:
    title: str
    country: str
    impact: str
    timestamp: float
    currency: str = ""
    category: str = ""
    actual: Optional[float] = None
    forecast: Optional[float] = None
    previous: Optional[float] = None
    is_actual: bool = False

class InstitutionalNewsFilter:
    """
    V6-INSIGNIA Institutional News Protection System.
    Enhanced with:
    - Multi-source fallbacks with health scoring
    - Symbol-specific filtering
    - Impact categorization
    - Proactive position flattening
    - Historical event analysis
    """
    
    _CURRENCY_MAP = {
        "EUR": ["EUR", "USD", "GBP", "JPY", "CHF"],
        "USD": ["USD", "EUR", "GBP", "JPY", "CAD", "AUD"],
        "GBP": ["GBP", "EUR", "USD", "JPY"],
        "JPY": ["JPY", "USD", "EUR", "GBP"],
        "CHF": ["CHF", "EUR", "USD"],
        "AUD": ["AUD", "USD", "NZD", "JPY"],
        "CAD": ["CAD", "USD"],
        "NZD": ["NZD", "USD", "AUD"],
    }

    def __init__(self, config: dict):
        self.config = config
        self.news_cfg = config.get("news_filter", {})
        self.enabled = self.news_cfg.get("enabled", True)
        self.cache_file = self.news_cfg.get("cache_file", "config/news_cache.json")
        self.source_url = self.news_cfg.get("source_url", "https://nfs.faireconomy.media/ff_calendar_thisweek.json")
        self.impact_levels = self.news_cfg.get("impact_levels", ["High"])
        self.buffer_before = self.news_cfg.get("buffer_before_min", 30)
        self.buffer_after = self.news_cfg.get("buffer_after_min", 15)
        self.resilience_mode = NewsResilience.HEALTHY
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36',
            'Accept': 'application/json, text/javascript, */*; q=0.01',
            'Accept-Language': 'en-US,en;q=0.9',
            'Referer': 'https://www.forexfactory.com/',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Sec-Fetch-Dest': 'empty',
            'Sec-Fetch-Mode': 'cors',
            'Sec-Fetch-Site': 'same-origin'
        })
        
        self.events: List[Dict[str, Any]] = []
        self._history: Dict[str, List[float]] = defaultdict(list)
        self._last_fetch = 0.0
        self._refresh_count = 0
        self._error_count = 0
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
        """Fetches news with local calendar priority, then web scraping fallback."""
        # 1. Check for Local Calendar (Primary - No web dependency)
        local_calendar = "config/news_calendar.json"
        if os.path.exists(local_calendar):
            try:
                with open(local_calendar, 'r') as f:
                    data = json.load(f)
                    events = data.get("events", data if isinstance(data, list) else [])
                    local_events = self._normalize_timestamps(events)
                    
                # FIX: Only return if we actually found local events!
                if len(local_events) > 0:
                    self.events = local_events
                    logger.info(f"News: Loaded {len(self.events)} events from local calendar (no web dependency).")
                    return
                else:
                    logger.info("News: Local calendar is empty. Proceeding to web sources.")
            except Exception as e:
                logger.warning(f"News: Failed to load local calendar: {e}. Falling back to web sources.")

        # 2. Legacy Manual Override (for backward compatibility)
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
            self.source_url, # Prioritize the configured source (faireconomy.media)
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
                            # 1. Institutional Timezone Normalization (Audit Bug #1 Fixed)
                            dt = datetime.fromisoformat(event["date"])
                            
                            # If it's already aware (standard in newer FF API responses), just convert to UTC
                            if dt.tzinfo is not None:
                                utc_ts = dt.timestamp()
                            else:
                                # Fallback for naive dates: assume US/Eastern as per FF legacy
                                import pytz
                                est_tz = pytz.timezone("US/Eastern")
                                localized_dt = est_tz.localize(dt)
                                utc_ts = localized_dt.timestamp()
                            
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
        self.resilience_mode = NewsResilience.DEGRADED

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

    def _normalize_timestamps(self, events: List[Dict]) -> List[Dict]:
        """Normalizes timestamps from ISO strings to Unix floats."""
        normalized = []
        for event in events:
            ts = event.get("timestamp")
            if isinstance(ts, str):
                try:
                    dt = datetime.fromisoformat(ts.replace('Z', '+00:00'))
                    ts = dt.timestamp()
                except Exception:
                    logger.debug(f"News: Skipping event with invalid timestamp: {ts}")
                    continue
            elif isinstance(ts, (int, float)):
                if ts > 1e12:  # milliseconds
                    ts = ts / 1000.0
            else:
                continue
                
            normalized.append({
                "title": event.get("title", "Unknown"),
                "country": event.get("country", "ALL"),
                "impact": event.get("impact", "Medium"),
                "timestamp": ts
            })
        return normalized

    def is_blocked(self, symbol: str, current_time: float = None) -> Optional[str]:
        if not self.enabled:
            return None

        if current_time is None:
            current_time = time.time()

        if self._is_data_stale():
            self.resilience_mode = NewsResilience.STALE
            logger.critical("NEWS ALERT: Data is STALE (Refreshed >24h ago). Global Trading LOCKOUT active.")
            return "STALE_DATA_LOCKOUT"
        
        if not self.events and self.resilience_mode == NewsResilience.HEALTHY:
             self.resilience_mode = NewsResilience.DEGRADED

        if not self.events:
            return None
            
        # Extract base and quote currencies
        base_ccy, quote_ccy = "ALL", "ALL" # Defaults
        if len(symbol) >= 6:
            base_ccy = symbol[0:3].upper()
            quote_ccy = symbol[3:6].upper()
        
        # Build the contagion list using the previously unused _CURRENCY_MAP
        affected_currencies = {base_ccy, quote_ccy, "ALL"}
        if base_ccy in self._CURRENCY_MAP:
            affected_currencies.update(self._CURRENCY_MAP[base_ccy])
        if quote_ccy in self._CURRENCY_MAP:
            affected_currencies.update(self._CURRENCY_MAP[quote_ccy])
        
        # Buffer range
        before_sec = self.buffer_before * 60
        after_sec = self.buffer_after * 60
        
        for event in self.events:
            event_ts = event["timestamp"]
            event_country = event["country"].upper()
            
            # Check if event country is in our mapped contagion list
            if event_country in affected_currencies or event_country == "USD": # USD affects everything
                if (current_time >= event_ts - before_sec) and (current_time <= event_ts + after_sec):
                    return event["title"]
                    
        return None

    def get_upcoming_events(self, current_time: float = None, window_hours: int = 24) -> List[Dict[str, Any]]:
        """Returns events happening in the next X hours."""
        if current_time is None:
            current_time = time.time()
            
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
    def _is_data_stale(self) -> bool:
        """Checks if the cached news data is older than 24 hours."""
        if not os.path.exists(self.cache_file):
            return True
        
        mtime = os.path.getmtime(self.cache_file)
        return (time.time() - mtime) > 86400  # 24 hours

    def get_resilience_multiplier(self) -> float:
        """Returns a lot size multiplier based on feed health."""
        if self.resilience_mode == NewsResilience.HEALTHY:
            return 1.0
        elif self.resilience_mode == NewsResilience.DEGRADED:
            return 0.5
        else:
            return 0.0

    def get_health_status(self) -> Dict[str, Any]:
        """Returns comprehensive health metrics."""
        return {
            "mode": self.resilience_mode.value,
            "event_count": len(self.events),
            "last_fetch": self._last_fetch,
            "refresh_count": self._refresh_count,
            "error_count": self._error_count,
            "upcoming_24h": len(self.get_upcoming_events(window_hours=24))
        }

    def categorize_event(self, event: Dict) -> str:
        """Categorizes an event by type."""
        title = event.get("title", "").lower()
        
        categories = {
            "inflation": ["cpi", "pce", "inflation", "consumer price"],
            "employment": ["nonfarm", "payroll", "unemployment", "jobs"],
            "interest": ["rate", "fed", "ecb", "boj", "boe", "monetary"],
            "gdp": ["gdp", "growth", "gdp"],
            "trade": ["trade balance", "current account", "exports", "imports"],
            "consumer": ["retail", "consumer confidence", "confidence"],
            "manufacturing": ["pmi", "ism", "manufacturing", "industrial"],
        }
        
        for cat, keywords in categories.items():
            if any(kw in title for kw in keywords):
                return cat
        return "other"

    def get_impact_score(self, event: Dict) -> int:
        """Calculates impact score for an event."""
        impact = event.get("impact", "Medium").upper()
        title = event.get("title", "").lower()
        score = 0
        
        if impact == "HIGH":
            score = 3
        elif impact == "MEDIUM":
            score = 2
        else:
            score = 1
        
        if any(t in title for t in ["rate", "fed", "ecb", "employment", "gdp", "cpi"]):
            score += 1
            
        return min(score, 5)

    def get_significant_events(self, window_hours: int = 24, min_score: int = 3) -> List[Dict]:
        """Returns high-impact events with score filtering."""
        events = self.get_upcoming_events(window_hours=window_hours)
        return [e for e in events if self.get_impact_score(e) >= min_score]

    def get_symbol_events(self, symbol: str, window_hours: int = 24) -> List[Dict]:
        """Returns events specifically relevant to a symbol."""
        if len(symbol) < 6:
            return []
            
        ccy1 = symbol[0:3].upper()
        ccy2 = symbol[3:6].upper()
        
        events = self.get_upcoming_events(window_hours=window_hours)
        relevant = []
        
        for event in events:
            event_ccy = event.get("country", "").upper()
            if event_ccy in [ccy1, ccy2, "ALL", "GLOBAL"]:
                relevant.append(event)
                
        return relevant

    def should_flatten_position(self, symbol: str, current_time: float) -> bool:
        """Determines if a position should be closed proactively before news hits."""
        auto_close_min = self.config.get("news_filter", {}).get("auto_close_before_min", 5)
        # FIX: Look forward into the future by ADDING time, simulating if we hit the block window
        simulated_future_time = current_time + (auto_close_min * 60)
        
        # Check if trading will be blocked in 'auto_close_min' minutes
        return self.is_blocked(symbol, simulated_future_time) is not None

    def record_actual(self, event: Dict, actual: float):
        """Records actual vs forecast for post-event analysis."""
        event["actual"] = actual
        event["is_actual"] = True
        event["surprise"] = actual - event.get("forecast", 0) if event.get("forecast") else None

    def refresh_if_needed(self):
        """Forces refresh if cache is expired."""
        if self._is_data_stale():
            logger.info("News: Forcing refresh due to stale data.")
            self.fetch_news()
