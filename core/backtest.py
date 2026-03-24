"""
TRADING BOT V3 - Backtest Engine
Simulates trading strategy on historical data with proper bias handling.
"""

import os
import sys
import logging
from datetime import datetime
from typing import List, Optional

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
    - Proper P/L calculation
    """

    def __init__(self, config: dict, strategy: StrategyEngine):
        self.config = config
        self.strategy = strategy

    def run(self, symbol: str, h4_candles: List[dict], m30_candles: List[dict],
            m15_candles: List[dict]):
        """Run backtest on historical data."""
        if not h4_candles or not m30_candles or not m15_candles:
            logger.error("Insufficient data for backtest")
            return

        logger.info("H4: %d candles, M30: %d candles, M15: %d candles",
                     len(h4_candles), len(m30_candles), len(m15_candles))

        # Get symbol config from config file (not hardcoded)
        symbol_config = self.config.get("symbols_config", {}).get(symbol, {})
        point = symbol_config.get("point", 0.01)
        lot = symbol_config.get("lot", 0.01)
        contract_size = symbol_config.get("contract_size", 1)
        spread_pips = self.config.get("backtest", {}).get("spread_pips", {}).get(symbol, 50)
        spread = spread_pips * point

        balance = self.config.get("backtest", {}).get("initial_balance", 1000)
        initial_balance = balance

        logger.info("Symbol Config — Point: %s, Spread: %s pips (%s), Contract: %s, Lot: %s",
                     point, spread_pips, spread, contract_size, lot)

        trades = []
        signal_count = 0
        last_trade_idx = -999
        cooldown = self.config.get("strategy", {}).get("cooldown_candles", 3)
        bt_dashboard = BacktestDashboard(self.config)
        total_candles = len(m30_candles) - 110

        logger.info("Running backtest on %d candles...", total_candles)

        for i in range(100, len(m30_candles) - 10):
            current_time = datetime.fromtimestamp(m30_candles[i]["time"]).strftime("%Y-%m-%d %H:%M")
            bt_dashboard.show_progress(i - 100, total_candles, current_time, signal_count, len(trades))

            if i - last_trade_idx < cooldown:
                continue

            current_price = m30_candles[i]["close"]
            h4_data = [c for c in h4_candles if c["time"] < m30_candles[i]["time"]]
            m30_data = m30_candles[:i + 1]
            m15_data = [c for c in m15_candles if c["time"] <= m30_candles[i]["time"]]

            if len(h4_data) < 50 or len(m30_data) < 100 or len(m15_data) < 100:
                continue

            signal = self.strategy.analyze(symbol, h4_data, m30_data, m15_data, current_price)
            if signal:
                signal_count += 1
                if signal.confidence >= self.strategy.min_confidence:
                    last_trade_idx = i

                    # Apply spread
                    entry = signal.entry_price + spread if signal.direction == "BUY" else signal.entry_price - spread

                    # Simulate trade (with bias fix)
                    # Increased window to 300 candles (150 hours) to prevent premature OPEN status
                    window = self.config.get("backtest", {}).get("simulation_window", 300)
                    future_candles = m30_candles[i + 1:i + window]
                    outcome = self._simulate_trade(signal, future_candles, entry)

                    # Calculate P/L
                    if outcome == "WIN":
                        pip_profit = abs(signal.take_profit - entry) / point
                        pnl = pip_profit * point * contract_size * lot
                        balance += pnl
                        result = "TP"
                    elif outcome == "LOSS":
                        pip_loss = abs(signal.stop_loss - entry) / point
                        pnl = -pip_loss * point * contract_size * lot
                        balance += pnl
                        result = "SL"
                    else:
                        pnl = 0
                        result = "OPEN"

                    trades.append({
                        "time": current_time,
                        "direction": signal.direction,
                        "entry": entry,
                        "sl": signal.stop_loss,
                        "tp": signal.take_profit,
                        "lot": lot,
                        "result": result,
                        "pnl": pnl,
                        "rr": signal.rr_ratio,
                    })

        # Compile results
        wins = [t for t in trades if t["result"] == "TP"]
        losses = [t for t in trades if t["result"] == "SL"]
        total_profit = sum(t["pnl"] for t in wins)
        total_loss = abs(sum(t["pnl"] for t in losses))

        results = {
            "symbol": symbol,
            "start_date": datetime.fromtimestamp(m30_candles[100]["time"]).strftime("%Y-%m-%d"),
            "end_date": datetime.fromtimestamp(m30_candles[-10]["time"]).strftime("%Y-%m-%d"),
            "initial_balance": initial_balance,
            "final_balance": balance,
            "return_pct": (balance - initial_balance) / initial_balance * 100,
            "total_trades": len(trades),
            "winning_trades": len(wins),
            "losing_trades": len(losses),
            "win_rate": len(wins) / len(trades) * 100 if trades else 0,
            "profit_factor": total_profit / total_loss if total_loss > 0 else 0,
            "total_profit": total_profit,
            "total_loss": total_loss,
            "avg_win": total_profit / len(wins) if wins else 0,
            "avg_loss": total_loss / len(losses) if losses else 0,
            "rr_ratio": (total_profit / len(wins)) / (total_loss / len(losses)) if wins and losses else 0,
            "max_drawdown": self._calc_drawdown(trades, initial_balance),
            "max_win_streak": self._calc_streak(trades, "TP"),
            "max_loss_streak": self._calc_streak(trades, "SL"),
            "trades": trades,
        }

        # Store all trades in a separate CSV file
        import csv
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

    def _simulate_trade(self, signal: TradeSignal, future_candles: List[dict], entry: float) -> str:
        """
        Simulate trade outcome with look-ahead bias fix.

        When both SL and TP could be hit in the same candle, we use the candle's
        open-to-close direction to determine which was likely hit first:
        - If candle opens moving TOWARD SL first → LOSS
        - If candle opens moving TOWARD TP first → WIN
        """
        if not future_candles:
            return "OPEN"

        for candle in future_candles:
            if signal.direction == "BUY":
                tp_hit = candle["high"] >= signal.take_profit
                sl_hit = candle["low"] <= signal.stop_loss

                if tp_hit and sl_hit:
                    # Both hit in same candle — use open direction to resolve
                    # If candle opened going down first (open > close or open near high),
                    # SL was likely hit first
                    if candle["open"] > candle["close"]:
                        return "LOSS"  # Bearish candle → SL hit first
                    else:
                        return "WIN"   # Bullish candle → TP hit first

                if tp_hit:
                    return "WIN"
                if sl_hit:
                    return "LOSS"

            else:  # SELL
                tp_hit = candle["low"] <= signal.take_profit
                sl_hit = candle["high"] >= signal.stop_loss

                if tp_hit and sl_hit:
                    # Both hit — use open direction
                    if candle["open"] < candle["close"]:
                        return "LOSS"  # Bullish candle → SL hit first for SELL
                    else:
                        return "WIN"   # Bearish candle → TP hit first for SELL

                if tp_hit:
                    return "WIN"
                if sl_hit:
                    return "LOSS"

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
