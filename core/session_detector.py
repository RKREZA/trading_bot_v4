import datetime
from typing import Dict, List, Optional, Tuple
from enum import Enum
from dataclasses import dataclass

class SessionType(Enum):
    TOKYO = "TOKYO"
    LONDON = "LONDON"
    NEW_YORK = "NEW_YORK"
    LONDON_NY = "LONDON/NY"
    ROLLOVER = "ROLLOVER"
    GLOBAL = "GLOBAL"
    CLOSED = "CLOSED"

@dataclass
class SessionConfig:
    name: str
    open_hour: int
    close_hour: int
    volatility_score: int
    liquidity_score: int

class SessionDetector:
    """
    V6-INSIGNIA Institutional Session Detector.
    Advanced market session detection with:
    - DST-aware boundaries for all major sessions
    - Volatility and liquidity scoring
    - Configurable windows
    - Historical session analysis
    """
    
    _SESSION_CONFIGS: Dict[str, SessionConfig] = {
        "TOKYO": SessionConfig("TOKYO", 0, 8, 3, 4),
        "LONDON": SessionConfig("LONDON", 7, 16, 8, 9),
        "NEW_YORK": SessionConfig("NEW_YORK", 13, 21, 8, 8),
        "LONDON/NY": SessionConfig("LONDON/NY", 12, 16, 10, 10),
        "ROLLOVER": SessionConfig("ROLLOVER", 21, 24, 2, 3),
        "GLOBAL": SessionConfig("GLOBAL", 0, 24, 5, 5),
    }

    @staticmethod
    def _is_dst_active(dt: datetime.datetime, region: str = "EU") -> bool:
        """Determines if DST is active for a region."""
        year = dt.year
        
        if region == "EU":
            march_last = datetime.date(year, 3, 31)
            while march_last.weekday() != 6:
                march_last -= datetime.timedelta(days=1)
            dst_start = datetime.datetime(year, march_last.month, march_last.day, 1, 0, tzinfo=datetime.timezone.utc)
            
            oct_last = datetime.date(year, 10, 31)
            while oct_last.weekday() != 6:
                oct_last -= datetime.timedelta(days=1)
            dst_end = datetime.datetime(year, oct_last.month, oct_last.day, 1, 0, tzinfo=datetime.timezone.utc)
        
        elif region == "US":
            march_first = datetime.date(year, 3, 1)
            sundays = 0
            d = march_first
            while sundays < 2:
                if d.weekday() == 6:
                    sundays += 1
                    if sundays == 2:
                        break
                d += datetime.timedelta(days=1)
            dst_start = datetime.datetime(year, d.month, d.day, 7, 0, tzinfo=datetime.timezone.utc)
            
            nov_first = datetime.date(year, 11, 1)
            d = nov_first
            while d.weekday() != 6:
                d += datetime.timedelta(days=1)
            dst_end = datetime.datetime(year, d.month, d.day, 6, 0, tzinfo=datetime.timezone.utc)
        else:
            return False
        
        aware_dt = dt if dt.tzinfo else dt.replace(tzinfo=datetime.timezone.utc)
        return dst_start <= aware_dt < dst_end

    @staticmethod
    def get_session(dt: datetime.datetime, broker_offset_hours: int = 0, symbol: str = None) -> str:
        """Determines current market session with DST-aware boundaries.
        
        Args:
            dt: datetime to check
            broker_offset_hours: broker UTC offset
            symbol: trading symbol (for crypto 24/7 detection)
        """
        # Crypto is 24/7 - never closed
        if symbol and symbol.startswith(("BTC", "ETH", "XRP", "LTC", "DOGE")):
            hour = dt.hour
            if 13 <= hour < 21:
                return "LONDON/NY"  # Peak crypto volume
            elif 8 <= hour < 13:
                return "LONDON"
            elif 21 <= hour < 24 or 0 <= hour < 8:
                return "ASIA_PACIFIC"
        
        if dt.tzinfo is not None:
            utc_dt = dt.astimezone(datetime.timezone.utc)
        else:
            utc_dt = dt.replace(tzinfo=datetime.timezone.utc)
            
        hour = utc_dt.hour
        weekday = utc_dt.weekday()
        
        is_weekend = False
        if weekday == 5:
            is_weekend = True
        elif weekday == 4 and hour >= 22:
            is_weekend = True
        elif weekday == 6 and hour < 21:
            is_weekend = True
            
        eu_dst = SessionDetector._is_dst_active(utc_dt, "EU")
        us_dst = SessionDetector._is_dst_active(utc_dt, "US")
        
        london_open = 7 if eu_dst else 8
        london_close = 16
        ny_open = 12 if us_dst else 13
        ny_close = 21
        
        session = "GLOBAL"
        if ny_open <= hour < london_close:
            session = "LONDON/NY"
        elif london_open <= hour < ny_open:
            session = "LONDON"
        elif london_close <= hour < ny_close:
            session = "NEW_YORK"
        elif 0 <= hour < london_open:
            session = "TOKYO"
        elif ny_close <= hour < 24:
            session = "ROLLOVER"

        if is_weekend:
            return f"{session} (CLOSED)"
            
        return session

    @staticmethod
    def get_session_info(dt: datetime.datetime) -> Dict[str, any]:
        """Returns comprehensive session information including scores."""
        session = SessionDetector.get_session(dt)
        clean_session = session.replace(" (CLOSED)", "")
        
        config = SessionDetector._SESSION_CONFIGS.get(clean_session)
        
        is_closed = "CLOSED" in session
        
        return {
            "session": clean_session,
            "display": session,
            "volatility_score": config.volatility_score if config else 5,
            "liquidity_score": config.liquidity_score if config else 5,
            "is_closed": is_closed,
            "config": config
        }

    @staticmethod
    def get_optimal_sessions(dt: datetime.datetime, count: int = 2) -> List[str]:
        """Returns top N most volatile/liquid sessions sorted by score."""
        session_info = SessionDetector.get_session_info(dt)
        current = session_info["session"]
        
        scores = [(s, cfg.volatility_score + cfg.liquidity_score) 
                 for s, cfg in SessionDetector._SESSION_CONFIGS.items()
                 if s != current]
        
        scores.sort(key=lambda x: x[1], reverse=True)
        return [s[0] for s in scores[:count]]

    @staticmethod
    def is_session_active(dt: datetime.datetime, broker_offset_hours: int = 0, 
                         allowed_sessions: List[str] = None) -> bool:
        """Check if current time falls within allowed sessions."""
        if not allowed_sessions:
            return True
            
        current = SessionDetector.get_session(dt, broker_offset_hours)
        
        if current in allowed_sessions:
            return True
            
        if current == "LONDON/NY":
            if "LONDON" in allowed_sessions or "NEW_YORK" in allowed_sessions:
                return True
            if "GLOBAL" in allowed_sessions:
                return True
                
        return False
