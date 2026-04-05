import datetime

class SessionDetector:
    """
    Utility for detecting market sessions based on Broker Time (Server Time).
    Standard MT5 Broker Time is usually UTC+2 winter / UTC+3 summer.
    """
    
    @staticmethod
    def get_session(dt: datetime.datetime) -> str:
        """
        Determines the current market session based on the hour of the day.
        
        Session Map (Broker Time):
        - TOKYO: 00:00 - 08:00
        - LONDON: 08:00 - 16:00
        - NEW_YORK: 13:00 - 21:00
        - LONDON/NY OVERLAP: 13:00 - 16:00 (Prioritized)
        """
        hour = dt.hour
        
        # Priority: Overlap
        if 13 <= hour < 16:
            return "LONDON/NY"
        
        if 8 <= hour < 13:
            return "LONDON"
            
        if 16 <= hour < 21:
            return "NEW_YORK"
            
        if 0 <= hour < 8:
            return "TOKYO"
            
        return "GLOBAL"
