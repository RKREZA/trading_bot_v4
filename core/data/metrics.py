import numpy as np
import logging
from typing import List, Dict, Any, Tuple
from datetime import datetime, timedelta

logger = logging.getLogger("trading_bot.metrics")

class MetricEngine:
    """
    V6-LIVE: Institutional Metric Calculation Engine.
    Provides real-time VaR, Correlation, Drawdown, and Performance analytics.
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
    def calculate_sharpe_ratio(returns: List[float], risk_free_rate: float = 0.02) -> float:
        """
        Calculates the Sharpe ratio (annualized).
        Assumes returns are daily excess returns.
        """
        if len(returns) < 2:
            return 0.0
            
        returns_array = np.array(returns)
        excess_returns = returns_array - risk_free_rate/252  # Daily risk-free rate
        
        if np.std(excess_returns) == 0:
            return 0.0
            
        sharpe = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)  # Annualized
        return sharpe

    @staticmethod
    def calculate_sortino_ratio(returns: List[float], risk_free_rate: float = 0.02) -> float:
        """
        Calculates the Sortino ratio (annualized).
        Measures return against downside deviation.
        """
        if len(returns) < 2:
            return 0.0
            
        returns_array = np.array(returns)
        excess_returns = returns_array - risk_free_rate/252  # Daily risk-free rate
        
        # Downside deviation (only negative returns)
        downside_returns = np.minimum(excess_returns, 0)
        downside_deviation = np.sqrt(np.mean(np.square(downside_returns)))
        
        if downside_deviation == 0:
            return 0.0 if np.mean(excess_returns) <= 0 else float('inf')
            
        sortino = np.mean(excess_returns) / downside_deviation * np.sqrt(252)  # Annualized
        return sortino

    @staticmethod
    def calculate_win_rate(profits: List[float]) -> float:
        """
        Calculates the win rate percentage.
        """
        if not profits:
            return 0.0
            
        wins = sum(1 for p in profits if p > 0)
        return (wins / len(profits)) * 100

    @staticmethod
    def calculate_profit_factor(profits: List[float]) -> float:
        """
        Calculates the profit factor (gross profit / gross loss).
        """
        if not profits:
            return 0.0
            
        gross_profit = sum(p for p in profits if p > 0)
        gross_loss = abs(sum(p for p in profits if p < 0))
        
        if gross_loss == 0:
            return float('inf') if gross_profit > 0 else 0.0
            
        return gross_profit / gross_loss

    @staticmethod
    def calculate_max_consecutive_losses(profits: List[float]) -> int:
        """
        Calculates the maximum number of consecutive losing trades.
        """
        if not profits:
            return 0
            
        max_consecutive = 0
        current_consecutive = 0
        
        for p in profits:
            if p <= 0:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0
                
        return max_consecutive

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

    @staticmethod
    def calculate_performance_metrics(trade_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculates comprehensive performance metrics from trade history.
        """
        if not trade_history:
            return {
                "total_trades": 0,
                "win_rate": 0.0,
                "profit_factor": 0.0,
                "sharpe_ratio": 0.0,
                "sortino_ratio": 0.0,
                "max_consecutive_losses": 0,
                "avg_profit": 0.0,
                "avg_loss": 0.0,
                "largest_win": 0.0,
                "largest_loss": 0.0
            }
        
        # Extract profits from trade history
        profits = [trade.get('profit', 0.0) for trade in trade_history]
        
        # Calculate returns for Sharpe/Sortino (assuming each trade is a period)
        returns = [p / 10000.0 for p in profits]  # Normalize to approximate returns
        
        # Separate wins and losses
        wins = [p for p in profits if p > 0]
        losses = [p for p in profits if p < 0]
        
        metrics = {
            "total_trades": len(profits),
            "win_rate": MetricEngine.calculate_win_rate(profits),
            "profit_factor": MetricEngine.calculate_profit_factor(profits),
            "sharpe_ratio": MetricEngine.calculate_sharpe_ratio(returns),
            "sortino_ratio": MetricEngine.calculate_sortino_ratio(returns),
            "max_consecutive_losses": MetricEngine.calculate_max_consecutive_losses(profits),
            "avg_profit": np.mean(wins) if wins else 0.0,
            "avg_loss": np.mean(losses) if losses else 0.0,
            "largest_win": max(wins) if wins else 0.0,
            "largest_loss": min(losses) if losses else 0.0
        }
        
        return metrics
