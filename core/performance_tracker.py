import pandas as pd
import numpy as np
from typing import List, Dict

class PerformanceTracker:
    """
    Institutional Performance Metrics Tracker.
    Calculates Sharpe, Sortino, Calmar, Drawdown, Expectancy, and Robustness.
    """

    @staticmethod
    def calculate_metrics(history: List[Dict], initial_balance: float = 1000.0, equity_curve: List[float] = None) -> Dict:
        """
        Institutional Grade Metric Calculation Engine.
        Uses Time-Series Log Returns for Ratios.
        """
        if not history:
            return {"status": "NO_TRADES"}

        df = pd.DataFrame(history)
        net_profit = df['pnl'].sum()
        total_pnl_percent = (net_profit / initial_balance) * 100
        
        wins = df[df['pnl'] > 0]
        losses = df[df['pnl'] <= 0]
        win_rate = len(wins) / len(df) * 100 if len(df) > 0 else 0
        
        # ── Institutional Drawdown ──
        # Balance-based Drawdown
        df['equity'] = initial_balance + df['pnl'].cumsum()
        df['peak'] = df['equity'].cummax()
        df['drawdown'] = (df['peak'] - df['equity']) / df['peak'] * 100
        max_drawdown = df['drawdown'].max()
        
        # Equity-based (Intra-candle)
        max_equity_drawdown = max_drawdown
        if equity_curve:
            eq_series = pd.Series(equity_curve)
            peak = eq_series.cummax()
            eq_dd = (peak - eq_series) / peak * 100
            max_equity_drawdown = eq_dd.max()

        # ── Statistical Ratios (Time-Series Based) ──
        returns = df['pnl'] / initial_balance
        avg_ret = returns.mean() if not returns.empty else 0
        std_ret = returns.std() if len(returns) > 1 else 0
        
        ann_factor = np.sqrt(252)
        sharpe = (avg_ret / std_ret * ann_factor) if std_ret > 0 else 0
        
        downside_returns = returns[returns < 0]
        downside_std = downside_returns.std() if len(downside_returns) > 1 else 0
        sortino = (avg_ret / downside_std * ann_factor) if downside_std > 0 else 0
        
        # SQN: Expectancy / Std(PnL) * sqrt(trades)
        avg_win = wins['pnl'].mean() if not wins.empty else 0
        avg_loss = losses['pnl'].mean() if not losses.empty else 0
        expectancy = (avg_win * (win_rate/100)) - (abs(avg_loss) * (1 - win_rate/100))
        
        pnl_std = df['pnl'].std() if len(df) > 1 else 0
        sqn = (expectancy / pnl_std * np.sqrt(len(df))) if pnl_std > 0 else 0
        
        loss_sum = abs(losses['pnl'].sum())
        profit_factor = wins['pnl'].sum() / loss_sum if loss_sum > 0 else float('inf')
        rr_ratio = (avg_win / abs(avg_loss)) if avg_loss != 0 else 0

        return {
            "net_profit": round(float(net_profit), 2),
            "net_profit_pct": f"{total_pnl_percent:.2f}%",
            "win_rate": f"{win_rate:.2f}%",
            "max_drawdown": f"{max_drawdown:.2f}%",
            "max_equity_drawdown": f"{max_equity_drawdown:.2f}%",
            "profit_factor": round(float(profit_factor), 2) if profit_factor != float('inf') else "INF",
            "sharpe_ratio": round(float(sharpe), 2),
            "sortino_ratio": round(float(sortino), 2),
            "expectancy": round(float(expectancy), 2),
            "rr_ratio": round(float(rr_ratio), 2),
            "sqn": round(float(sqn), 2),
            "total_trades": int(len(df))
        }

    @staticmethod
    def generate_professional_dashboard(portfolio_results: Dict) -> str:
        """
        Creates a Professional High-RRR Institutional Dashboard.
        Uses clear visual indicators and formatted tables.
        """
        lines = []
        lines.append("=" * 80)
        lines.append(" INSTITUTIONAL TRADING SYSTEM — BACKTEST DASHBOARD (V4-PRO) ")
        lines.append("=" * 80)
        
        # 1. Summary Block
        s = portfolio_results.get('portfolio', {})
        lines.append(f" PORTFOLIO PERFORMANCE SUMMARY:")
        lines.append(f" - Net Profit:     ${s.get('net_profit', 0):<15}   Profit Factor: {s.get('profit_factor', 0)}")
        lines.append(f" - Win Rate:       {s.get('win_rate', '0%'):<15}   Total Trades:  {s.get('total_trades', 0)}")
        lines.append(f" - Max Drawdown:   {s.get('max_drawdown', '0%'):<15}   Sharpe Ratio:  {s.get('sharpe_ratio', 0)}")
        lines.append(f" - SQN (Quality):  {s.get('sqn', 0):<15}   Expectancy:    ${s.get('expectancy', 0)}")
        lines.append("-" * 80)
        
        # 2. Strategy Leaderboard
        lines.append(f"{'STRATEGY ID':<20} | {'PROFIT':<10} | {'WIN %':<10} | {'MAX DD':<10} | {'SHARPE':<10} | {'SQN':<6}")
        lines.append("-" * 80)
        
        strategies = portfolio_results.get('strategies', {})
        for sid, res in sorted(strategies.items(), key=lambda x: x[1].get('sharpe_ratio', 0), reverse=True):
            lines.append(f"{sid:<20} | ${res.get('net_profit', 0):<9.2f} | {res.get('win_rate', '0%'):<10} | {res.get('max_drawdown', '0%'):<10} | {res.get('sharpe_ratio', 0):<10} | {res.get('sqn', 0):<6}")
        
        lines.append("-" * 80)
        
        # 3. Session Heatmap
        sessions = portfolio_results.get('sessions', {})
        lines.append(f" SESSION PERFORMANCE HEATMAP:")
        for name, res in sessions.items():
            lines.append(f" - {name:<15} Profit: ${res.get('net_profit', 0):<12} WR: {res.get('win_rate', '0%')}")
            
        lines.append("=" * 80)
        return "\n".join(lines)

    @staticmethod
    def save_audit_pack(history: List[Dict], results: Dict, output_dir: str):
        """
        Institutional Audit Pack Generator (Step 14: Output Requirements).
        Exports CSV artifacts for trades, strategy performance, and curves.
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. Trade Logs (CSV)
        df_trades = pd.DataFrame(history)
        if not df_trades.empty:
            df_trades.to_csv(f"{output_dir}/trades.csv", index=False)
        
        # 2. Strategy-wise Performance (CSV)
        strategies = results.get('strategies', {})
        if strategies:
            df_strats = pd.DataFrame(strategies).T
            df_strats.to_csv(f"{output_dir}/strategy_performance.csv")
        
        # 3. Equity and Drawdown Curves (CSV)
        if not df_trades.empty:
            initial_balance = results.get('portfolio', {}).get('initial_balance', 1000.0)
            df_trades['equity_curve'] = initial_balance + df_trades['pnl'].cumsum()
            df_trades['peak'] = df_trades['equity_curve'].cummax()
            df_trades['drawdown_curve'] = (df_trades['peak'] - df_trades['equity_curve']) / df_trades['peak'] * 100
            
            curves_df = df_trades[['timestamp', 'equity_curve', 'drawdown_curve']]
            curves_df.to_csv(f"{output_dir}/curves.csv", index=False)
            
        # 4. Professional Summary (MD)
        dashboard = PerformanceTracker.generate_professional_dashboard(results)
        with open(f"{output_dir}/summary.md", "w") as f:
            f.write(dashboard)
            
        return output_dir

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
