import numpy as np
import pandas as pd
from typing import List, Dict

class PerformanceMetrics:
    """
    Static utility for calculating post-trade performance analytics.
    Calculates standard institutional metrics like Sharpe Ratio, Profit Factor, and Max Drawdown.
    """
    @staticmethod
    def calculate_metrics(trades: List[Dict], initial_balance: float) -> Dict:
        """
        Processes a list of trade dictionaries to produce a summary metric report.
        
        Args:
            trades (List[Dict]): List of trade objects with 'pnl' and 'time' keys.
            initial_balance (float): Starting balance for the evaluation period.
            
        Returns:
            Dict: Comprehensive report containing Profit Factor, Win Rate, DD, Sharpe, etc.
        """
        if not trades:
            return {
                "initial_balance": initial_balance,
                "final_balance": initial_balance,
                "net_profit": 0,
                "profit_factor": 0,
                "max_drawdown_pct": 0,
                "max_drawdown_abs": 0,
                "recovery_factor": 0,
                "sharpe_ratio": 0,
                "sortino_ratio": 0,
                "calmar_ratio": 0,
                "expectancy": 0,
                "win_rate": 0,
                "total_trades": 0,
                "equity_curve": [initial_balance]
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
        # Drawdown and Recovery Factor
        equity_series = pd.Series(equity_curve)
        rolling_max = equity_series.cummax()
        drawdown_abs = (rolling_max - equity_series)
        drawdown_pct = drawdown_abs / rolling_max * 100
        max_drawdown_pct = drawdown_pct.max()
        max_drawdown_abs = drawdown_abs.max()
        recovery_factor = net_profit / max_drawdown_abs if max_drawdown_abs > 0 else net_profit
        
        # Sharpe & Sortino Ratio (Daily Percentage Returns based on exit time)
        if 'exit_time' in df.columns:
            df['date'] = pd.to_datetime(df['exit_time']).dt.date
        else:
            df['date'] = pd.to_datetime(df['time']).dt.date
        daily_pnl = df.groupby('date')['pnl'].sum()
        
        # Reconstruct daily balance
        daily_balances = [initial_balance]
        for pnl in daily_pnl:
            daily_balances.append(daily_balances[-1] + pnl)
        
        daily_bal_series = pd.Series(daily_balances)
        daily_ret = daily_bal_series.pct_change().dropna()
        
        sharpe = 0; sortino = 0; calmar = 0
        if len(daily_ret) > 1:
            # Sharpe
            std = daily_ret.std()
            sharpe = (daily_ret.mean() / std) * np.sqrt(252) if std > 0 else 0
            
            # Sortino (Downside deviation only)
            downside_ret = daily_ret[daily_ret < 0]
            downside_std = downside_ret.std()
            sortino = (daily_ret.mean() / downside_std) * np.sqrt(252) if downside_std > 0 else sharpe
            
            # Calmar (Annualized Return / Max Drawdown)
            total_days = (df.iloc[-1]['date'] - df.iloc[0]['date']).days if len(df) > 1 else 1
            annualized_ret = (balance / initial_balance) ** (365 / max(1, total_days)) - 1
            calmar = annualized_ret / (max_drawdown_pct / 100) if max_drawdown_pct > 0 else 0

        # Expectancy
        avg_win = wins['pnl'].mean() if not wins.empty else 0
        avg_loss = abs(losses['pnl'].mean()) if not losses.empty else 0
        expectancy = (win_rate / 100 * avg_win) - ((1 - win_rate / 100) * avg_loss)
        
        return {
            "initial_balance": round(initial_balance, 2),
            "final_balance": round(balance, 2),
            "net_profit": round(net_profit, 2),
            "profit_factor": round(profit_factor, 2),
            "max_drawdown_pct": round(max_drawdown_pct, 2),
            "max_drawdown_abs": round(max_drawdown_abs, 2),
            "recovery_factor": round(recovery_factor, 2),
            "sharpe_ratio": round(sharpe, 2),
            "sortino_ratio": round(sortino, 2),
            "calmar_ratio": round(calmar, 2),
            "expectancy": round(expectancy, 2),
            "win_rate": round(win_rate, 2),
            "total_trades": len(df),
            "equity_curve": equity_curve
        }
