import pandas as pd
import numpy as np
from typing import List, Dict

class PerformanceTracker:
    """
    Institutional Performance Metrics Tracker.
    Calculates Sharpe, Sortino, Drawdown, and Expectancy.
    """

    @staticmethod
    def calculate_metrics(history: List[Dict], initial_balance: float = 1000.0) -> Dict:
        if not history:
            return {"status": "No trades executed"}

        df = pd.DataFrame(history)
        
        df['cumulative_pnl'] = df['pnl'].cumsum()
        df['equity'] = initial_balance + df['cumulative_pnl']
        
        net_profit = df['pnl'].sum()
        
        wins = df[df['pnl'] > 0]
        losses = df[df['pnl'] <= 0]
        win_rate = len(wins) / len(df) * 100 if len(df) > 0 else 0
        
        df['peak'] = df['equity'].cummax()
        df['drawdown'] = (df['peak'] - df['equity']) / df['peak'] * 100 if not df.empty else 0
        max_drawdown = df['drawdown'].max()
        
        returns = df['pnl'] / (df['equity'].shift(1).fillna(initial_balance))
        sharpe = (returns.mean() / returns.std() * np.sqrt(252)) if returns.std() > 0 else 0
        
        downside_returns = returns[returns < 0]
        sortino = (returns.mean() / downside_returns.std() * np.sqrt(252)) if len(downside_returns) > 0 and downside_returns.std() > 0 else 0
        
        avg_win = wins['pnl'].mean() if len(wins) > 0 else 0
        avg_loss = abs(losses['pnl'].mean()) if len(losses) > 0 else 0
        expectancy = (avg_win * (win_rate/100)) - (avg_loss * (1 - win_rate/100))
        
        rr_ratio = (avg_win / avg_loss) if avg_loss > 0 else 0

        return {
            "net_profit": round(net_profit, 2),
            "win_rate": f"{win_rate:.2f}%",
            "max_drawdown": f"{max_drawdown:.2f}%",
            "sharpe_ratio": round(sharpe, 2),
            "sortino_ratio": round(sortino, 2),
            "expectancy": round(expectancy, 2),
            "rr_ratio": round(rr_ratio, 2),
            "total_trades": len(df)
        }

    @staticmethod
    def calculate_per_strategy(history: List[Dict], initial_balance: float = 1000.0) -> Dict:
        df = pd.DataFrame(history)
        if df.empty: return {}
        
        results = {}
        for sid in df['strategy_id'].unique():
            strat_history = df[df['strategy_id'] == sid].to_dict('records')
            results[sid] = PerformanceTracker.calculate_metrics(strat_history, initial_balance)
            
        return results

    @staticmethod
    def calculate_per_session(history: List[Dict], initial_balance: float = 1000.0) -> Dict:
        df = pd.DataFrame(history)
        if df.empty: return {}
        
        results = {}
        for session in df['session'].unique():
            sess_history = df[df['session'] == session].to_dict('records')
            results[session] = PerformanceTracker.calculate_metrics(sess_history, initial_balance)
            
        return results
