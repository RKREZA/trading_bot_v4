import numpy as np
import pandas as pd
from typing import List, Dict

class PerformanceMetrics:
    @staticmethod
    def calculate_metrics(trades: List[Dict], initial_balance: float) -> Dict:
        if not trades:
            return {
                "net_profit": 0,
                "profit_factor": 0,
                "max_drawdown": 0,
                "sharpe_ratio": 0,
                "expectancy": 0,
                "win_rate": 0
            }

        df = pd.DataFrame(trades)
        net_profit = df['pnl'].sum()
        wins = df[df['pnl'] > 0]
        losses = df[df['pnl'] <= 0]
        
        profit_factor = abs(wins['pnl'].sum() / losses['pnl'].sum()) if not losses.empty else float('inf')
        win_rate = len(wins) / len(df) * 100
        
        # Max Drawdown
        balance = initial_balance
        equity_curve = [balance]
        for pnl in df['pnl']:
            balance += pnl
            equity_curve.append(balance)
        
        equity_series = pd.Series(equity_curve)
        rolling_max = equity_series.cummax()
        drawdown = (rolling_max - equity_series) / rolling_max * 100
        max_drawdown = drawdown.max()
        
        # Sharpe Ratio (Daily)
        df['date'] = pd.to_datetime(df['time']).dt.date
        daily_pnl = df.groupby('date')['pnl'].sum()
        if len(daily_pnl) > 1:
            sharpe = (daily_pnl.mean() / daily_pnl.std()) * np.sqrt(252) if daily_pnl.std() > 0 else 0
        else:
            sharpe = 0
            
        # Expectancy
        avg_win = wins['pnl'].mean() if not wins.empty else 0
        avg_loss = abs(losses['pnl'].mean()) if not losses.empty else 0
        expectancy = (win_rate / 100 * avg_win) - ((1 - win_rate / 100) * avg_loss)
        
        return {
            "initial_balance": round(initial_balance, 2),
            "final_balance": round(balance, 2),
            "net_profit": round(net_profit, 2),
            "profit_factor": round(profit_factor, 2),
            "max_drawdown": round(max_drawdown, 2),
            "sharpe_ratio": round(sharpe, 2),
            "expectancy": round(expectancy, 2),
            "win_rate": round(win_rate, 2),
            "total_trades": len(df),
            "equity_curve": equity_curve
        }
