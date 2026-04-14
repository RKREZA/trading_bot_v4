import datetime

class SessionDetector:
    """
    Utility for detecting market sessions based on Broker Time (Server Time).
    Standard MT5 Broker Time is usually UTC+2 winter / UTC+3 summer.
    
    INSTITUTIONAL HARDENING (v6):
    - DST-aware session boundaries for London/NY
    - Configurable session windows
    """
    
    # DST-AWARE SESSION BOUNDARIES (UTC hours)
    # London: Summer (last Sun Mar → last Sun Oct) opens 07:00, Winter opens 08:00
    # New York: Summer opens 12:00, Winter opens 13:00
    # Overlap adjusts accordingly
    
    @staticmethod
    def _is_dst_active(dt: datetime.datetime, region: str = "EU") -> bool:
        """
        Determines if Daylight Saving Time is active for a region.
        EU DST: Last Sunday of March 01:00 UTC → Last Sunday of October 01:00 UTC.
        US DST: Second Sunday of March 02:00 → First Sunday of November 02:00.
        """
        year = dt.year
        
        if region == "EU":
            # Last Sunday of March
            march_last = datetime.date(year, 3, 31)
            while march_last.weekday() != 6:  # Sunday = 6
                march_last -= datetime.timedelta(days=1)
            dst_start = datetime.datetime(year, march_last.month, march_last.day, 1, 0, tzinfo=datetime.timezone.utc)
            
            # Last Sunday of October
            oct_last = datetime.date(year, 10, 31)
            while oct_last.weekday() != 6:
                oct_last -= datetime.timedelta(days=1)
            dst_end = datetime.datetime(year, oct_last.month, oct_last.day, 1, 0, tzinfo=datetime.timezone.utc)
        
        elif region == "US":
            # Second Sunday of March
            march_first = datetime.date(year, 3, 1)
            sundays = 0
            d = march_first
            while sundays < 2:
                if d.weekday() == 6:
                    sundays += 1
                    if sundays == 2:
                        break
                d += datetime.timedelta(days=1)
            dst_start = datetime.datetime(year, d.month, d.day, 7, 0, tzinfo=datetime.timezone.utc)  # 2AM EST = 7AM UTC
            
            # First Sunday of November
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
    def get_session(dt: datetime.datetime, broker_offset_hours: int = 0) -> str:
        """
        Determines the current market session with DST-aware boundaries.
        Returns '(CLOSED)' suffix on weekends to avoid confusion.
        """
        # Ensure we are working with the UTC hour
        if dt.tzinfo is not None:
            utc_dt = dt.astimezone(datetime.timezone.utc)
        else:
            utc_dt = dt.replace(tzinfo=datetime.timezone.utc)
            
        hour = utc_dt.hour
        weekday = utc_dt.weekday()
        
        # 1. Weekend Detection (Institutional Protocol)
        is_weekend = False
        if weekday == 5: # Saturday
            is_weekend = True
        elif weekday == 4 and hour >= 22: # Friday Close
            is_weekend = True
        elif weekday == 6 and hour < 21: # Sunday before open (10PM UTC-ish)
            is_weekend = True
            
        # 2. DST-Aware Session Boundaries
        eu_dst = SessionDetector._is_dst_active(utc_dt, "EU")
        us_dst = SessionDetector._is_dst_active(utc_dt, "US")
        
        # London: Opens 07:00 (summer) or 08:00 (winter), closes ~16:30
        london_open = 7 if eu_dst else 8
        london_close = 16 if eu_dst else 16
        
        # New York: Opens 12:00 (summer DST) or 13:00 (winter)
        ny_open = 12 if us_dst else 13
        ny_close = 21
        
        # Overlap: Between NY open and London quasi-close
        overlap_start = ny_open
        overlap_end = london_close
        
        # 3. Session Classification
        session = "GLOBAL"
        if overlap_start <= hour < overlap_end:
            session = "LONDON/NY"
        elif london_open <= hour < overlap_start:
            session = "LONDON"
        elif overlap_end <= hour < ny_close:
            session = "NEW_YORK"
        elif 0 <= hour < london_open:
            session = "TOKYO"
        elif ny_close <= hour < 24:
            session = "ROLLOVER"

        if is_weekend:
            return f"{session} (CLOSED)"
            
        return session
    
    @staticmethod
    def is_session_active(dt: datetime.datetime, broker_offset_hours: int = 0, 
                         allowed_sessions: list = None) -> bool:
        """
        Check if the current time falls within any of the allowed sessions.
        Handles LONDON/NY overlap by checking both sessions.
        """
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
