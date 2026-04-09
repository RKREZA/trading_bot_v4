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
        If 'dt' is UTC, we apply the broker_offset to align with the Session Map.
        Standard MT5 Broker Time (EET) is UTC+2 (Winter) / UTC+3 (Summer).
        """
        broker_time = dt + datetime.timedelta(hours=broker_offset_hours)
        hour = broker_time.hour
        
        if 13 <= hour < 16:
            return "LONDON/NY"
        
        if 8 <= hour < 13:
            return "LONDON"
            
        if 16 <= hour < 21:
            return "NEW_YORK"
            
        if 0 <= hour < 8:
            return "TOKYO"
            
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
                
        return False
