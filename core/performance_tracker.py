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
        
        # Monthly Performance Breakdown
        # Ensure timestamp is datetime
        df['dt'] = pd.to_datetime(df['timestamp'], unit='s')
        df['month_year'] = df['dt'].dt.to_period('M').astype(str)
        monthly_stats = df.groupby('month_year')['pnl'].agg(['sum', 'count']).to_dict('index')
        
        # Equity-based (Intra-candle)
        max_equity_drawdown = max_drawdown
        if equity_curve:
            if isinstance(equity_curve[0], dict) and "time" in equity_curve[0]:
                eq_df = pd.DataFrame(equity_curve)
                if 'strategy_id' in eq_df.columns:
                    eq_series = eq_df.groupby('time')['equity'].sum().reset_index(drop=True)
                else:
                    eq_series = eq_df['equity']
            else:
                eq_series = pd.Series(equity_curve)
                
            peak = eq_series.cummax()
            eq_dd = (peak - eq_series) / peak * 100
            max_equity_drawdown = eq_dd.max()

        # ── Statistical Ratios (Time-Series Based) ──
        returns = df['pnl'] / initial_balance
        avg_ret = returns.mean() if not returns.empty else 0
        std_ret = returns.std() if len(returns) > 1 else 0
        
        # FX/Gold trades 5d/week including overnight ≈ 260 annual trading days.
        # Using 252 (equity market convention) understates annualised volatility for FX pairs.
        ann_factor = 260.0
        ann_sqrt = np.sqrt(ann_factor)
        sharpe = (avg_ret / std_ret * ann_sqrt) if std_ret > 0 else 0
        
        downside_returns = returns[returns < 0]
        downside_std = downside_returns.std() if len(downside_returns) > 1 else 0
        sortino = (avg_ret / downside_std * ann_sqrt) if downside_std > 0 else 0
        
        # SQN: Expectancy / Std(PnL) * sqrt(trades)
        avg_win = wins['pnl'].mean() if not wins.empty else 0
        avg_loss = losses['pnl'].mean() if not losses.empty else 0
        expectancy = (avg_win * (win_rate/100)) - (abs(avg_loss) * (1 - win_rate/100))
        
        pnl_std = df['pnl'].std() if len(df) > 1 else 0
        sqn = (expectancy / pnl_std * np.sqrt(len(df))) if pnl_std > 0 else 0
        
        loss_sum = abs(losses['pnl'].sum())
        profit_factor = wins['pnl'].sum() / loss_sum if loss_sum > 0 else float('inf')
        rr_ratio = (avg_win / abs(avg_loss)) if avg_loss != 0 else 0

        # ── CAGR & Calmar Ratio ──
        # CAGR = (final_balance / initial_balance) ^ (365 / holding_days) - 1
        cagr = 0.0
        calmar = 0.0
        try:
            if 'timestamp' in df.columns and len(df) >= 2:
                t_start = df['timestamp'].min()
                t_end = df['timestamp'].max()
                holding_days = (t_end - t_start) / 86400.0
                if holding_days > 0:
                    final_balance = initial_balance + net_profit
                    ratio = final_balance / initial_balance if initial_balance > 0 else 1.0
                    if ratio > 0 and holding_days > 0:
                        try:
                            power = 365.0 / holding_days
                            # Safety cap for extreme math results during account wipeouts
                            if power < 1000:
                                cagr = (ratio ** power - 1.0) * 100  # as %
                        except (OverflowError, RuntimeWarning):
                            cagr = 0.0
                    # Calmar = CAGR / Max Drawdown%  (higher is better)
                    calmar = (cagr / max_drawdown) if max_drawdown > 0 else 0.0
        except Exception:
            pass  # Non-fatal; timestamps may be missing in synthetic/test data

        return {
            "initial_balance": round(float(initial_balance), 2),
            "final_balance": round(float(initial_balance + net_profit), 2),
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
            "cagr": f"{cagr:.2f}%",
            "calmar_ratio": round(float(calmar), 2),
            "total_trades": int(len(df)),
            "monthly_stats": monthly_stats
        }

    @staticmethod
    def generate_professional_dashboard(portfolio_results: Dict) -> str:
        """
        Creates a Professional High-RRR Institutional Dashboard (V5-INSIGNIA).
        Uses clear visual indicators and formatted tables.
        """
        lines = []
        lines.append("=" * 80)
        lines.append(" INSTITUTIONAL TRADING SYSTEM — BACKTEST DASHBOARD (V5-INSIGNIA) ")
        lines.append(f" Symbol: {portfolio_results.get('symbol', 'N/A')} | Range: {portfolio_results.get('start_date', 'N/A')} to {portfolio_results.get('end_date', 'N/A')}")
        lines.append("=" * 80)
        
        # 1. Summary Block
        s = portfolio_results.get('portfolio', {})
        if not s or s.get("status") == "NO_TRADES":
            lines.append(" !!! AUDIT NOTICE: NO TRADES EXECUTED DURING THIS PERIOD !!!")
            lines.append(" Reason: Strategy criteria not met, or insufficient volatility in the requested range.")
            lines.append("-" * 80)
            return "\n".join(lines)

        lines.append(f" PORTFOLIO PERFORMANCE SUMMARY:")
        lines.append(f" - Initial Balance: ${s.get('initial_balance', 0):<15}   Final Balance: ${s.get('final_balance', 0)}")
        lines.append(f" - Net Profit:      ${s.get('net_profit', 0):<15}   Profit Factor: {s.get('profit_factor', 0)}")
        lines.append(f" - Win Rate:        {s.get('win_rate', '0%'):<15}   Total Trades:  {s.get('total_trades', 0)}")
        lines.append(f" - Max Drawdown:    {s.get('max_drawdown', '0%'):<15}   Sharpe Ratio:  {s.get('sharpe_ratio', 0)}")
        lines.append(f" - SQN (Quality):   {s.get('sqn', 0):<15}   Expectancy:    ${s.get('expectancy', 0)}")
        lines.append("-" * 80)
        
        # 1.5 Monthly Performance
        monthly = s.get("monthly_stats", {})
        if monthly:
            lines.append(f" MONTHLY PERFORMANCE BREAKDOWN:")
            lines.append(f" {'MONTH':<10} | {'PROFIT':<12} | {'TRADES':<8} | {'STATUS'}")
            lines.append("-" * 50)
            balance_at_month_start = s.get('initial_balance', 0)
            for month in sorted(monthly.keys()):
                m_data = monthly[month]
                pnl = m_data['sum']
                cnt = m_data['count']
                status = "PROFIT" if pnl > 0 else "LOSS"
                lines.append(f" {month:<10} | ${pnl:<11.2f} | {cnt:<8} | {status}")
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
            
            # Use 'timestamp' if available, otherwise use index
            time_col = 'timestamp' if 'timestamp' in df_trades.columns else None
            if time_col:
                curves_df = df_trades[[time_col, 'equity_curve', 'drawdown_curve']]
            else:
                curves_df = df_trades[['equity_curve', 'drawdown_curve']]
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
