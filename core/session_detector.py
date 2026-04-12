import datetime

class SessionDetector:
    """
    Utility for detecting market sessions based on Broker Time (Server Time).
    Standard MT5 Broker Time is usually UTC+2 winter / UTC+3 summer.
    """
    
    @staticmethod
    def get_session(dt: datetime.datetime, broker_offset_hours: int = 0) -> str:
        """
        Determines the current market session.
        Returns '(CLOSED)' suffix on weekends to avoid confusion.
        """
        # Ensure we are working with the UTC hour
        if dt.tzinfo is not None:
            utc_dt = dt.astimezone(datetime.timezone.utc)
        else:
            # Safety: If passed naive time (like some broker feeds), 
            # we assume it's already aligned with the requested context.
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
            
        # 2. Mapping (UTC Hours)
        session = "GLOBAL"
        if 13 <= hour < 16:
            session = "LONDON/NY"
        elif 8 <= hour < 13:
            session = "LONDON"
        elif 16 <= hour < 21:
            session = "NEW_YORK"
        elif 0 <= hour < 8:
            session = "TOKYO"
        elif 21 <= hour < 24:
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
