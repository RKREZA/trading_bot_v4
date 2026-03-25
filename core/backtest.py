"""
TRADING BOT V3 - Backtest Engine
Simulates trading strategy on historical data with proper bias handling.
"""

import os
import sys
import logging
from datetime import datetime
from typing import List, Optional
import bisect
import csv
import numpy as np

# Add the project root to sys.path so Python and linters can find 'dashboard' and 'core'
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from dashboard import BacktestDashboard
from core.strategy_engine import StrategyEngine, TradeSignal

logger = logging.getLogger("trading_bot.backtest")


class BacktestEngine:
    """
    Backtesting engine with:
    - Fixed look-ahead bias (same-candle SL/TP ordering uses open direction)
    - Config-based lot sizes (not hardcoded)
    - Proper P/L calculation with commission and slippage
    - Sharpe ratio calculation
    """

    def __init__(self, config: dict, strategy: StrategyEngine):
        self.config = config
        self.strategy = strategy

    @staticmethod
    def _find_last_index(candles, time_threshold):
        """Return index of last candle with time < time_threshold using binary search."""
        times = [c["time"] for c in candles]
        idx = bisect.bisect_left(times, time_threshold) - 1
        return idx

    def run(self, symbol: str, h4_candles: List[dict], m30_candles: List[dict],
            m15_candles: List[dict], quiet: bool = False) -> Optional[dict]:
        """
        Run backtest on historical data.
        If quiet=True, no dashboard output, returns results dict.
        """
        if not h4_candles or not m30_candles or not m15_candles:
            logger.error("Insufficient data for backtest")
            return None

        logger.info("H4: %d candles, M30: %d candles, M15: %d candles",
                     len(h4_candles), len(m30_candles), len(m15_candles))

        # Get symbol config
        symbol_config = self.config.get("symbols_config", {}).get(symbol, {})
        point = symbol_config.get("point", 0.01)
        lot = symbol_config.get("lot", 0.01)
        contract_size = symbol_config.get("contract_size", 1)
        spread_pips = self.config.get("backtest", {}).get("spread_pips", {}).get(symbol, 50)
        spread = spread_pips * point
        commission_per_lot = self.config.get("backtest", {}).get("commission_usd", {}).get(symbol, 0)
        slippage_pips = self.config.get("backtest", {}).get("slippage_pips", {}).get(symbol, 0)

        balance = self.config.get("backtest", {}).get("initial_balance", 1000)
        initial_balance = balance

        logger.info("Symbol Config — Point: %s, Spread: %s pips (%s), Contract: %s, Lot: %s, Commission: $%s/lot, Slippage: %s pips",
                     point, spread_pips, spread, contract_size, lot, commission_per_lot, slippage_pips)

        trades = []
        signal_count = 0
        last_trade_idx = -999
        cooldown = self.config.get("strategy", {}).get("cooldown_candles", 3)
        bt_dashboard = BacktestDashboard(self.config) if not quiet else None
        total_candles = len(m30_candles) - 110

        logger.info("Running backtest on %d candles...", total_candles)

        # Pre‑build time arrays for binary search
        h4_times = [c["time"] for c in h4_candles]
        m15_times = [c["time"] for c in m15_candles]

        for i in range(100, len(m30_candles) - 10):
            if not quiet and bt_dashboard:
                current_time = datetime.fromtimestamp(m30_candles[i]["time"]).strftime("%Y-%m-%d %H:%M")
                bt_dashboard.show_progress(i - 100, total_candles, current_time, signal_count, len(trades))

            if i - last_trade_idx < cooldown:
                continue

            # Use the *next* candle's open as entry price to avoid look‑ahead
            if i + 1 >= len(m30_candles):
                continue
            entry_price = m30_candles[i+1]["open"]

            # For indicators, use data only up to current candle i (strictly before entry)
            h4_last_idx = self._find_last_index(h4_candles, m30_candles[i]["time"])
            if h4_last_idx < 49:
                continue
            h4_data = h4_candles[:h4_last_idx+1]

            m30_data = m30_candles[:i+1]

            m15_last_idx = self._find_last_index(m15_candles, m30_candles[i]["time"])
            if m15_last_idx < 99:
                continue
            m15_data = m15_candles[:m15_last_idx+1]

            # Analyze — unpack (signal, h4_trend) tuple
            signal, _ = self.strategy.analyze(symbol, h4_data, m30_data, m15_data, entry_price)
            if signal:
                signal_count += 1
                if signal.confidence >= self.strategy.min_confidence:
                    last_trade_idx = i

                    # Apply spread and slippage
                    entry = entry_price + spread if signal.direction == "BUY" else entry_price - spread
                    slippage = slippage_pips * point
                    if signal.direction == "BUY":
                        entry += slippage
                    else:
                        entry -= slippage

                    # Compute ATR from the m30 candles leading up to this bar
                    atr = self.strategy._calculate_atr(m30_candles[:i+1])

                    # Simulate trade using future candles starting from i+2
                    window = self.config.get("backtest", {}).get("simulation_window", 300)
                    future_candles = m30_candles[i+2:i+window]
                    outcome = self._simulate_trade(signal, future_candles, entry, atr)

                    # Calculate P/L with commission
                    if outcome == "WIN":
                        pip_profit = abs(signal.take_profit - entry) / point
                        gross_pnl = pip_profit * point * contract_size * lot
                        commission = commission_per_lot * lot
                        pnl = gross_pnl - commission
                        balance += pnl
                        result = "TP"
                    elif outcome == "LOSS":
                        pip_loss = abs(signal.stop_loss - entry) / point
                        gross_pnl = -pip_loss * point * contract_size * lot
                        commission = commission_per_lot * lot
                        pnl = gross_pnl - commission
                        balance += pnl
                        result = "SL"
                    else:
                        pnl = 0
                        result = "OPEN"

                    trades.append({
                        "time": datetime.fromtimestamp(m30_candles[i]["time"]).strftime("%Y-%m-%d %H:%M"),
                        "direction": signal.direction,
                        "entry": entry,
                        "sl": signal.stop_loss,
                        "tp": signal.take_profit,
                        "lot": lot,
                        "result": result,
                        "pnl": pnl,
                        "rr": signal.rr_ratio,
                    })
                    # OPEN trades (neither SL nor TP hit in window) are stored
                    # but excluded from win/loss stats so they don't deflate win rate.

        # Compile results — only count closed trades (TP/SL) in win/loss stats
        closed_trades = [t for t in trades if t["result"] in ("TP", "SL")]
        open_trades   = [t for t in trades if t["result"] == "OPEN"]
        wins   = [t for t in closed_trades if t["result"] == "TP"]
        losses = [t for t in closed_trades if t["result"] == "SL"]
        total_profit = sum(t["pnl"] for t in wins)
        total_loss   = abs(sum(t["pnl"] for t in losses))

        # Calculate Sharpe ratio on trading-days only (skip weekend zero-return days)
        daily_pnl: dict = {}
        for t in closed_trades:
            day = t["time"].split()[0]
            daily_pnl[day] = daily_pnl.get(day, 0) + t["pnl"]
        # Filter out zero-return days (weekends / non-trading days)
        returns = [v for v in daily_pnl.values() if v != 0]
        if len(returns) > 1:
            mean_return = np.mean(returns)
            std_return  = np.std(returns)
            sharpe = mean_return / std_return * np.sqrt(252) if std_return > 0 else 0
        else:
            sharpe = 0

        results = {
            "symbol": symbol,
            "start_date": datetime.fromtimestamp(m30_candles[100]["time"]).strftime("%Y-%m-%d"),
            "end_date": datetime.fromtimestamp(m30_candles[-10]["time"]).strftime("%Y-%m-%d"),
            "initial_balance": initial_balance,
            "final_balance": balance,
            "return_pct": (balance - initial_balance) / initial_balance * 100,
            "total_trades": len(closed_trades),  # closed only
            "open_trades": len(open_trades),      # informational
            "winning_trades": len(wins),
            "losing_trades": len(losses),
            "win_rate": len(wins) / len(closed_trades) * 100 if closed_trades else 0,
            "profit_factor": total_profit / total_loss if total_loss > 0 else 0,
            "total_profit": total_profit,
            "total_loss": total_loss,
            "avg_win": total_profit / len(wins) if wins else 0,
            "avg_loss": total_loss / len(losses) if losses else 0,
            "rr_ratio": (total_profit / len(wins)) / (total_loss / len(losses)) if wins and losses else 0,
            "max_drawdown": self._calc_drawdown(trades, initial_balance),
            "max_win_streak": self._calc_streak(trades, "TP"),
            "max_loss_streak": self._calc_streak(trades, "SL"),
            "sharpe_ratio": sharpe,
            "trades": trades,  # full list including OPEN (for CSV export)
        }

        if not quiet:
            # Save trades to CSV
            results_dir = os.path.join(project_root, "backtest_results")
            os.makedirs(results_dir, exist_ok=True)
            filename = f"{symbol}_trades_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            filepath = os.path.join(results_dir, filename)

            if trades:
                try:
                    with open(filepath, mode='w', newline='') as f:
                        writer = csv.DictWriter(f, fieldnames=trades[0].keys())
                        writer.writeheader()
                        writer.writerows(trades)
                    logger.info("Saved %d trades to %s", len(trades), filepath)
                except Exception as e:
                    logger.error("Failed to save trades to CSV: %s", e)

            bt_dashboard.show_results(results)

        return results

    def _simulate_trade(self, signal: TradeSignal, future_candles: List[dict],
                         entry: float, atr: float) -> str:
        """
        Simulate trade outcome with professional 3-phase trailing SL.

        Phase 0 (initial): SL stays at the original stop-loss level.
        Phase 1 (1R  profit): SL moves to breakeven (entry price).
        Phase 2 (1.5R profit): SL trails 1.5× ATR behind the best price seen.
        Phase 3 (2R+ profit): SL tightens to 1.0× ATR behind best price.

        This mirrors what a professional manual trader would do: protect the
        trade as soon as it is meaningfully in profit, then ride the trend
        with a trailing stop rather than waiting passively for fixed TP.
        """
        if not future_candles:
            return "OPEN"

        sl = signal.stop_loss
        tp = signal.take_profit
        risk = abs(entry - sl)       # 1R distance
        trail_phase = 0              # 0 = initial, 1 = BE, 2 = trail, 3 = tight
        best_price = entry           # highest high (BUY) or lowest low (SELL) seen

        for candle in future_candles:
            if signal.direction == "BUY":
                # Update best price seen
                best_price = max(best_price, candle["high"])
                profit    = best_price - entry

                # Phase transitions (only advance, never retreat)
                if trail_phase < 3 and profit >= risk * 2.0:
                    trail_phase = 3
                elif trail_phase < 2 and profit >= risk * 1.5:
                    trail_phase = 2
                elif trail_phase < 1 and profit >= risk * 1.0:
                    trail_phase = 1

                # Advance SL based on phase (SL can only move up for BUY)
                if trail_phase == 1:
                    sl = max(sl, entry)                        # breakeven
                elif trail_phase == 2:
                    sl = max(sl, best_price - atr * 1.5)      # 1.5 ATR trail
                elif trail_phase == 3:
                    sl = max(sl, best_price - atr * 1.0)      # tighter 1 ATR trail

                if candle["high"] >= tp and candle["low"] <= sl:
                    return "WIN" if candle["open"] < candle["close"] else "LOSS"
                if candle["high"] >= tp:
                    return "WIN"
                if candle["low"] <= sl:
                    return "LOSS" if sl <= signal.stop_loss else "WIN"  # trailed SL = partial win

            else:  # SELL
                best_price = min(best_price, candle["low"])
                profit     = entry - best_price

                if trail_phase < 3 and profit >= risk * 2.0:
                    trail_phase = 3
                elif trail_phase < 2 and profit >= risk * 1.5:
                    trail_phase = 2
                elif trail_phase < 1 and profit >= risk * 1.0:
                    trail_phase = 1

                if trail_phase == 1:
                    sl = min(sl, entry)                        # breakeven
                elif trail_phase == 2:
                    sl = min(sl, best_price + atr * 1.5)      # 1.5 ATR trail
                elif trail_phase == 3:
                    sl = min(sl, best_price + atr * 1.0)      # tighter 1 ATR trail

                if candle["low"] <= tp and candle["high"] >= sl:
                    return "WIN" if candle["open"] > candle["close"] else "LOSS"
                if candle["low"] <= tp:
                    return "WIN"
                if candle["high"] >= sl:
                    return "LOSS" if sl >= signal.stop_loss else "WIN"  # trailed SL = partial win

        return "OPEN"

    @staticmethod
    def _calc_drawdown(trades: list, initial: float) -> float:
        """Calculate maximum drawdown percentage."""
        if not trades:
            return 0.0
        balance = initial
        peak = initial
        max_dd = 0.0
        for t in trades:
            balance += t["pnl"]
            if balance > peak:
                peak = balance
            dd = (peak - balance) / peak * 100 if peak > 0 else 0
            if dd > max_dd:
                max_dd = dd
        return max_dd

    @staticmethod
    def _calc_streak(trades: list, result_type: str) -> int:
        """Calculate maximum consecutive streak of a result type."""
        if not trades:
            return 0
        max_s, cur = 0, 0
        for t in trades:
            if t["result"] == result_type:
                cur += 1
                max_s = max(max_s, cur)
            else:
                cur = 0
        return max_s