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
        
        Institutional Standard: All internal logic uses UTC. 
        Sessions are defined in UTC hours:
          - TOKYO: 00:00 - 08:00 UTC
          - LONDON: 08:00 - 16:00 UTC
          - NEW YORK: 13:00 - 21:00 UTC
        """
        # Ensure we are working with the UTC hour
        # If dt is naive, we assume it's UTC. If aware, we convert to UTC.
        if dt.tzinfo is not None:
            utc_dt = dt.astimezone(datetime.timezone.utc)
        else:
            utc_dt = dt
            
        hour = utc_dt.hour
        
        # Mapping (UTC Hours)
        if 13 <= hour < 16:
            return "LONDON/NY"
        
        if 8 <= hour < 13:
            return "LONDON"
            
        if 16 <= hour < 21:
            return "NEW_YORK"
            
        if 0 <= hour < 8:
            return "TOKYO"
            
        # Institutional Dead Zone: Broker Rollover Period
        if 21 <= hour < 24:
            return "ROLLOVER"
            
        return "GLOBAL"
    
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
