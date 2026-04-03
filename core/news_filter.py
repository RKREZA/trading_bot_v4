"""
TRADING BOT V3 — ForexFactory Economic Calendar Filter
Monitors High-Impact news events and provides a ±30 min trading blockade.
"""

import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
import logging
from typing import List, Dict, Tuple, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .broker_clock import BrokerClock

logger = logging.getLogger("trading_bot.news")

# ForexFactory Weekly XML Feed
FF_XML_URL = "https://nfs.faireconomy.media/ff_calendar_thisweek.xml"

class NewsFilter:
    def __init__(self, block_minutes: int = 30, broker_clock: 'BrokerClock' = None):
        """
        Args:
            block_minutes (int): Number of minutes before and after a high-impact event to block trading.
            broker_clock: BrokerClock instance for authoritative time.
        """
        self.block_minutes = block_minutes
        self.events: List[Dict] = []
        self.last_fetch_time: Optional[datetime] = None
        self._cache_duration_hours = 6
        self._broker_clock = broker_clock

    def _now(self) -> datetime:
        """Get current time from broker clock or fallback to UTC."""
        if self._broker_clock:
            return self._broker_clock.now()
        return datetime.now(timezone.utc)

    def fetch_news(self) -> bool:
        """
        Fetches the weekly economic calendar from ForexFactory XML.
        Runs once every 6 hours minimum to prevent spamming their simple API.
        """
        # Return True if cache is still valid
        if self.last_fetch_time:
            seconds_since = (self._now() - self.last_fetch_time).total_seconds()
            if seconds_since < (self._cache_duration_hours * 3600):
                return True

        try:
            req = urllib.request.Request(FF_XML_URL, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=10) as response:
                xml_data = response.read()

            root = ET.fromstring(xml_data)
            parsed_events = []

            for event in root.findall("event"):
                title = event.findtext("title", "")
                country = event.findtext("country", "")
                date_str = event.findtext("date", "")
                time_str = event.findtext("time", "")
                impact = event.findtext("impact", "")
                
                # We only care about high impact and non-tentative stuff
                if impact != "High" or not time_str or time_str.lower() == "all day" or time_str.lower() == "tentative":
                    continue
                
                # Format: Date="10-06-2025" Time="8:30am" 
                # Note: FF timezone inside the XML depends on settings, but NFS XML is typically EST (UTC-5)
                # Let's parse it securely.
                try:
                    dt_str = f"{date_str} {time_str}"
                    # e.g., 10-06-2025 8:30am 
                    # NFS feed is fixed at EST (UTC-5 consistently standard time ignoring daylight)
                    raw_dt = datetime.strptime(dt_str, "%m-%d-%Y %I:%M%p")
                    # Bind EST (-5 hours) to UTC
                    event_dt_utc = raw_dt.replace(tzinfo=timezone(timedelta(hours=-5))).astimezone(timezone.utc)
                    
                    parsed_events.append({
                        "title": title,
                        "currency": country.upper(),
                        "time_utc": event_dt_utc,
                        "impact": "High"
                    })
                except Exception as e:
                    logger.debug(f"Failed to parse news time: {dt_str} - {e}")
                    pass

            # Sort events chronologically
            self.events = sorted(parsed_events, key=lambda x: x["time_utc"])
            self.last_fetch_time = self._now()
            logger.info(f"Successfully fetched and cached {len(self.events)} High-Impact news events.")
            return True

        except Exception as e:
            logger.error(f"Failed to fetch ForexFactory news grid: {e}")
            # On failure, apply a 2-minute cooldown before trying again
            self.last_fetch_time = self._now() - timedelta(hours=self._cache_duration_hours) + timedelta(minutes=2)
            return False

    def get_upcoming_news(self, symbol: str) -> List[Dict]:
        """
        Get upcoming high-impact news for the currencies in the active symbol.
        """
        if not self.events:
            self.fetch_news()
            
        now = self._now()
        target_currencies = []
        
        # e.g. "XAUUSDm" -> XAU, USD
        if "USD" in symbol: target_currencies.append("USD")
        if "EUR" in symbol: target_currencies.append("EUR")
        if "GBP" in symbol: target_currencies.append("GBP")
        if "JPY" in symbol: target_currencies.append("JPY")
        if "CAD" in symbol: target_currencies.append("CAD")
        if "AUD" in symbol: target_currencies.append("AUD")
        if "NZD" in symbol: target_currencies.append("NZD")
        if "CHF" in symbol: target_currencies.append("CHF")
            
        upcoming = []
        for e in self.events:
            if e["currency"] in target_currencies:
                # Add it if it happens in the future, or recently passed (within today)
                # We specifically want to show upcoming, not fully finished past ones.
                if e["time_utc"] > (now - timedelta(minutes=self.block_minutes)):
                    upcoming.append(e)
                    
        return upcoming[:3] # Return top 3 upcoming

    def is_trading_blocked(self, symbol: str) -> Tuple[bool, str]:
        """
        Check if trading operations should be halted due to proximity to news.
        Returns: (is_blocked, reason_string)
        """
        upcoming = self.get_upcoming_news(symbol)
        now = self._now()

        for event in upcoming:
            time_diff = (event["time_utc"] - now).total_seconds() / 60.0
            
            # If time_diff is within the block window ([-30, +30])
            if -abs(self.block_minutes) <= time_diff <= self.block_minutes:
                # E.g., time_diff = 15 -> "15 mins until" ; time_diff = -15 -> "15 mins ago"
                direction = "until" if time_diff >= 0 else "ago"
                return True, f"Blocked: {event['title']} ({abs(int(time_diff))}m {direction})"
                
        return False, "Clear"
