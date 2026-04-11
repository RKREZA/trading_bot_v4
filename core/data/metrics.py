import numpy as np
import logging
from typing import List, Dict, Any
from datetime import datetime, timedelta

logger = logging.getLogger("trading_bot.metrics")

class MetricEngine:
    """
    V6-LIVE: Institutional Metric Calculation Engine.
    Provides real-time VaR, Correlation, and Drawdown analytics.
    """
    
    @staticmethod
    def calculate_var(equity_history: List[float], confidence: float = 0.95, window: int = 50) -> float:
        """
        Calculates 1-Day Value at Risk using the Historical Simulation method.
        Standard institutional window: 50-100 samples.
        """
        if len(equity_history) < 5:
            return 0.0
            
        recent_equity = equity_history[-window:]
        returns = np.diff(recent_equity) / recent_equity[:-1]
        
        if len(returns) < 1:
            return 0.0
            
        # Standard Historical VaR
        var_pct = np.percentile(returns, (1 - confidence) * 100)
        return abs(var_pct) * 100 # Returns % VaR

    @staticmethod
    def calculate_drawdown(equity_history: List[float]) -> Dict[str, float]:
        """Calculates current and max drawdown metrics."""
        if not equity_history:
            return {"current": 0.0, "max": 0.0}
            
        peak = -np.inf
        max_dd = 0.0
        current_eq = equity_history[-1]
        
        for eq in equity_history:
            if eq > peak:
                peak = eq
            dd = (peak - eq) / peak if peak > 0 else 0
            if dd > max_dd:
                max_dd = dd
                
        current_dd = (peak - current_eq) / peak if peak > 0 else 0
        
        return {
            "current": current_dd * 100,
            "max": max_dd * 100
        }

    @staticmethod
    def get_exposure_heatmap(positions: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Calculates net exposure per currency basket.
        Example: {'USD': 4.5, 'JPY': -2.0, 'GOLD': 1.0}
        """
        exposure = {}
        for pos in positions:
            symbol = pos.get('symbol', '').upper()
            lots = pos.get('volume', 0.0)
            direction = 1 if "BUY" in pos.get('type_text', '').upper() else -1
            
            # Simplified basket aggregation (XAUUSD -> GOLD, USD)
            if "XAU" in symbol:
                exposure['GOLD'] = exposure.get('GOLD', 0.0) + (lots * direction)
                exposure['USD'] = exposure.get('USD', 0.0) - (lots * direction) # USD is the quote
            elif "JPY" in symbol:
                exposure['JPY'] = exposure.get('JPY', 0.0) + (lots * direction)
                # ... other pairs ...
            elif "GBP" in symbol:
                exposure['GBP'] = exposure.get('GBP', 0.0) + (lots * direction)
            elif "USD" in symbol:
                # If USD is base (USDJPY) -> positive
                # If USD is quote (EURUSD) -> negative
                if symbol.startswith("USD"):
                    exposure['USD'] = exposure.get('USD', 0.0) + (lots * direction)
                else:
                    exposure['USD'] = exposure.get('USD', 0.0) - (lots * direction)
        
        return exposure
