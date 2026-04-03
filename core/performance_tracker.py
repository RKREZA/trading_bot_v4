"""
TRADING BOT V3 — Per-Strategy Performance Tracker
Live-updating performance metrics with zero shared state.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional

logger = logging.getLogger("trading_bot.performance_tracker")


class PerformanceTracker:
    """
    Maintains running performance metrics for a single strategy.
    
    Provides:
        - Real-time PnL / Win Rate / Drawdown tracking
        - Equity curve construction
        - Post-hoc institutional metrics (Sharpe, Sortino, Profit Factor)
    
    Each StrategyRuntime gets its own independent instance.
    """

    def __init__(self, strategy_id: str, initial_balance: float = 0.0):
        """
        Args:
            strategy_id: Owning strategy's unique identifier
            initial_balance: Starting balance for equity tracking
        """
        self.strategy_id = strategy_id
        self.initial_balance = initial_balance
        self.balance = initial_balance

        # Running counters
        self.total_trades = 0
        self.win_count = 0
        self.loss_count = 0
        self.gross_profit = 0.0
        self.gross_loss = 0.0
        self.daily_pnl = 0.0
        self.daily_trades = 0
        self.consecutive_losses = 0
        self.max_consecutive_losses = 0

        # Equity tracking
        self.equity_curve: List[float] = [initial_balance]
        self.peak_balance = initial_balance
        self.max_drawdown_abs = 0.0
        self.max_drawdown_pct = 0.0

        # Trade log (for post-hoc analysis)
        self._trades: List[dict] = []
        self._daily_equity_history: List[float] = [initial_balance]

    @property
    def win_rate(self) -> float:
        """Current win rate as percentage."""
        if self.total_trades == 0:
            return 0.0
        return (self.win_count / self.total_trades) * 100.0

    @property
    def net_pnl(self) -> float:
        return self.gross_profit + self.gross_loss  # gross_loss is negative

    @property
    def profit_factor(self) -> float:
        if self.gross_loss == 0:
            return float('inf') if self.gross_profit > 0 else 0.0
        return abs(self.gross_profit / self.gross_loss)

    @property
    def expectancy(self) -> float:
        if self.total_trades == 0:
            return 0.0
        avg_win = (self.gross_profit / self.win_count) if self.win_count > 0 else 0.0
        avg_loss = abs(self.gross_loss / self.loss_count) if self.loss_count > 0 else 0.0
        wr = self.win_rate / 100.0
        return (wr * avg_win) - ((1 - wr) * avg_loss)

    def record_trade(self, trade_record: dict) -> None:
        """
        Process a completed trade and update all running metrics.
        
        Args:
            trade_record: Dict with keys: ticket, pnl, result, session, etc.
        """
        pnl = trade_record.get("pnl", 0.0)
        self.total_trades += 1
        self.daily_trades += 1
        self.daily_pnl += pnl
        self.balance += pnl

        if pnl >= 0:
            self.win_count += 1
            self.gross_profit += pnl
            self.consecutive_losses = 0
        else:
            self.loss_count += 1
            self.gross_loss += pnl  # pnl is negative
            self.consecutive_losses += 1
            self.max_consecutive_losses = max(
                self.max_consecutive_losses, self.consecutive_losses
            )

        # Update equity curve and drawdown
        self.equity_curve.append(self.balance)
        if self.balance > self.peak_balance:
            self.peak_balance = self.balance
        
        current_dd_abs = self.peak_balance - self.balance
        if current_dd_abs > self.max_drawdown_abs:
            self.max_drawdown_abs = current_dd_abs
        
        current_dd_pct = (current_dd_abs / self.peak_balance * 100) if self.peak_balance > 0 else 0
        if current_dd_pct > self.max_drawdown_pct:
            self.max_drawdown_pct = current_dd_pct

        self._trades.append(trade_record)

    def reset_daily_stats(self) -> None:
        """Called at the start of each new trading day."""
        self._daily_equity_history.append(self.balance)
        # Keep rolling window
        if len(self._daily_equity_history) > 100:
            self._daily_equity_history = self._daily_equity_history[-100:]
        self.daily_pnl = 0.0
        self.daily_trades = 0

    def record_daily_close(self, equity: float) -> None:
        """Record end-of-day equity for risk calculations."""
        self._daily_equity_history.append(equity)
        if len(self._daily_equity_history) > 100:
            self._daily_equity_history = self._daily_equity_history[-100:]

    def get_daily_equity_history(self) -> List[float]:
        return list(self._daily_equity_history)

    def finalize(self) -> Dict:
        """
        Calculate full institutional-grade metrics from the trade log.
        Called at the end of a backtest or reporting period.
        
        Returns:
            Dict with comprehensive performance metrics
        """
        if not self._trades:
            return {
                "strategy_id": self.strategy_id,
                "initial_balance": round(self.initial_balance, 2),
                "final_balance": round(self.initial_balance, 2),
                "net_profit": 0, "profit_factor": 0,
                "max_drawdown_pct": 0, "max_drawdown_abs": 0,
                "recovery_factor": 0, "sharpe_ratio": 0,
                "sortino_ratio": 0, "calmar_ratio": 0,
                "expectancy": 0, "win_rate": 0,
                "total_trades": 0, "equity_curve": [self.initial_balance],
                "max_consecutive_losses": 0,
            }

        df = pd.DataFrame(self._trades)
        net_profit = df['pnl'].sum()
        wins = df[df['pnl'] > 0]
        losses = df[df['pnl'] <= 0]

        pf = abs(wins['pnl'].sum() / losses['pnl'].sum()) if not losses.empty and losses['pnl'].sum() != 0 else (
            float('inf') if not wins.empty else 0
        )
        wr = len(wins) / len(df) * 100

        # Equity series
        eq = pd.Series(self.equity_curve)
        rolling_max = eq.cummax()
        dd_abs = (rolling_max - eq)
        dd_pct = dd_abs / rolling_max * 100
        max_dd_pct = dd_pct.max()
        max_dd_abs = dd_abs.max()
        recovery = net_profit / max_dd_abs if max_dd_abs > 0 else net_profit

        # Sharpe & Sortino (daily returns)
        sharpe = 0.0
        sortino = 0.0
        calmar = 0.0

        if 'exit_time' in df.columns:
            df['date'] = pd.to_datetime(df['exit_time']).dt.date
        elif 'time' in df.columns:
            df['date'] = pd.to_datetime(df['time']).dt.date
        else:
            df['date'] = range(len(df))

        daily_pnl = df.groupby('date')['pnl'].sum()
        daily_bals = [self.initial_balance]
        for p in daily_pnl:
            daily_bals.append(daily_bals[-1] + p)

        daily_ret = pd.Series(daily_bals).pct_change().dropna()

        if len(daily_ret) > 1:
            std = daily_ret.std()
            sharpe = (daily_ret.mean() / std) * np.sqrt(252) if std > 0 else 0

            downside = daily_ret[daily_ret < 0]
            ds_std = downside.std()
            sortino = (daily_ret.mean() / ds_std) * np.sqrt(252) if ds_std > 0 else sharpe

            total_days = max(1, len(daily_pnl))
            ann_ret = (self.balance / self.initial_balance) ** (365 / total_days) - 1
            calmar = ann_ret / (max_dd_pct / 100) if max_dd_pct > 0 else 0

        avg_win = wins['pnl'].mean() if not wins.empty else 0
        avg_loss = abs(losses['pnl'].mean()) if not losses.empty else 0
        expectancy_val = (wr / 100 * avg_win) - ((1 - wr / 100) * avg_loss)

        return {
            "strategy_id": self.strategy_id,
            "initial_balance": round(self.initial_balance, 2),
            "final_balance": round(self.balance, 2),
            "net_profit": round(net_profit, 2),
            "profit_factor": round(pf, 2),
            "max_drawdown_pct": round(max_dd_pct, 2),
            "max_drawdown_abs": round(max_dd_abs, 2),
            "recovery_factor": round(recovery, 2),
            "sharpe_ratio": round(sharpe, 2),
            "sortino_ratio": round(sortino, 2),
            "calmar_ratio": round(calmar, 2),
            "expectancy": round(expectancy_val, 2),
            "win_rate": round(wr, 2),
            "total_trades": len(df),
            "equity_curve": self.equity_curve,
            "max_consecutive_losses": self.max_consecutive_losses,
            "trades": self._trades,
        }

    def get_summary(self) -> Dict:
        """Quick summary of current performance (for dashboards)."""
        return {
            "strategy_id": self.strategy_id,
            "balance": round(self.balance, 2),
            "net_pnl": round(self.net_pnl, 2),
            "daily_pnl": round(self.daily_pnl, 2),
            "total_trades": self.total_trades,
            "daily_trades": self.daily_trades,
            "win_rate": round(self.win_rate, 2),
            "profit_factor": round(self.profit_factor, 2),
            "max_drawdown_pct": round(self.max_drawdown_pct, 2),
            "consecutive_losses": self.consecutive_losses,
        }
