"""
TRADING BOT V3 - Bulletproof Backtest Engine

Professional-grade backtesting with zero tolerance for bias.

Key guarantees (matching live trading):
  1. One position at a time — no overlapping trades
  2. Risk-based lot sizing — lot scales with balance (compounding)
  3. Conservative same-candle resolution — SL assumed hit first when ambiguous
  4. Honest trailing stop P/L — actual exit price, not original SL/TP
  5. Daily trade limit & daily loss circuit breaker
  6. Max drawdown halt
  7. Gap slippage — fills at gap-open, not at SL/TP level
  8. Accurate P/L: (exit - entry) × direction × contract × lot - commission
"""

import os
import sys
import logging
from datetime import datetime, timezone, timedelta
from typing import List, Optional, Dict
import bisect
import csv
import math
import numpy as np

# Add the project root to sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from dashboard import BacktestDashboard
from core.strategy_engine import StrategyEngine, TradeSignal

logger = logging.getLogger("trading_bot.backtest")


# ---------------------------------------------------------------------------
# Open-trade state tracker
# ---------------------------------------------------------------------------

class _OpenTrade:
    """Tracks the state of a single open trade during simulation."""

    __slots__ = (
        "signal", "entry_price", "lot", "sl", "tp",
        "trail_phase", "best_price", "entry_index",
        "entry_time", "atr",
    )

    def __init__(self, signal: TradeSignal, entry_price: float, lot: float,
                 entry_index: int, entry_time: int, atr: float):
        self.signal = signal
        self.entry_price = entry_price
        self.lot = lot
        self.sl = signal.stop_loss
        self.tp = signal.take_profit
        self.trail_phase = 0
        self.best_price = entry_price
        self.entry_index = entry_index
        self.entry_time = entry_time
        self.atr = atr


# ---------------------------------------------------------------------------
# Backtest Engine
# ---------------------------------------------------------------------------

class BacktestEngine:
    """
    Bulletproof backtesting engine.

    Mirrors live-trading constraints:
    - Single position at a time
    - Risk-based lot sizing
    - Daily trade limits + loss circuit breaker + drawdown halt
    - Conservative candle resolution
    - Gap slippage
    - Accurate P/L from actual exit price
    """

    def __init__(self, config: dict, strategy: StrategyEngine):
        self.config = config
        self.strategy = strategy

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _find_last_index(candles, time_threshold):
        """Return index of last candle with time < time_threshold using binary search."""
        times = [c["time"] for c in candles]
        idx = bisect.bisect_left(times, time_threshold) - 1
        return idx

    @staticmethod
    def _utc_date_from_ts(ts: int) -> str:
        """Return 'YYYY-MM-DD' string for a unix timestamp in UTC."""
        return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d")

    def _calc_lot_size(self, balance: float, entry: float, sl: float,
                       point: float, contract_size: float) -> float:
        """
        Risk-based lot sizing — mirrors PositionManager.calculate_lot_size.

        lot = risk_amount / (risk_distance_in_points × point_value)
        """
        risk_pct = self.config.get("risk_per_trade", 1.0)
        risk_amount = balance * (risk_pct / 100.0)

        risk_distance = abs(entry - sl) / point
        if risk_distance <= 0:
            return 0.01

        point_value = contract_size * point
        lot = risk_amount / (risk_distance * point_value)

        # Round to 0.01 step, clamp to [0.01, max_lot_size]
        step = 0.01
        lot = math.floor(lot / step) * step
        max_lot = self.config.get("max_lot_size", 5.0)
        lot = max(0.01, min(lot, max_lot))
        return lot

    # ------------------------------------------------------------------
    # Trade simulation on a single candle
    # ------------------------------------------------------------------

    def _process_candle_for_trade(self, trade: _OpenTrade, candle: dict) -> Optional[dict]:
        """
        Process one candle against an open trade.
        Returns a trade-result dict if the trade closes, else None.

        Resolution rules (conservative):
          1. Gap check: if candle open is beyond SL or TP, fill at open.
          2. For BUY: check SL (low) before TP (high). For SELL: check SL (high) before TP (low).
          3. Same-candle ambiguity: if both levels breached, SL wins (worst case).
          4. Trail phases advance using candle-close (conservative), not high/low.
        """
        o, h, l, c = candle["open"], candle["high"], candle["low"], candle["close"]
        direction = trade.signal.direction
        sl = trade.sl
        tp = trade.tp
        entry = trade.entry_price

        # Trailing stop config
        ts_cfg = self.config.get("trailing_stop", {})
        ts_enabled = ts_cfg.get("enabled", True)
        be_rr = ts_cfg.get("breakeven_at_rr", 1.0)
        p2_rr = ts_cfg.get("trail_phase2_at_rr", 1.5)
        p3_rr = ts_cfg.get("trail_phase3_at_rr", 2.0)
        p2_mult = ts_cfg.get("trail_atr_multiplier", 1.5)
        p3_mult = ts_cfg.get("trail_tight_atr_multiplier", 1.0)

        risk = abs(entry - trade.signal.stop_loss)  # 1R = original risk distance
        if risk <= 0:
            risk = trade.atr  # fallback

        def _make_result(exit_price: float, result_type: str) -> dict:
            """Build a trade result dict with actual P/L."""
            d = 1.0 if direction == "BUY" else -1.0
            symbol_cfg = self._symbol_config
            gross = (exit_price - entry) * d * symbol_cfg["contract_size"] * trade.lot
            comm = symbol_cfg["commission"] * trade.lot
            pnl = gross - comm
            return {
                "time": datetime.fromtimestamp(trade.entry_time, tz=timezone.utc).strftime("%Y-%m-%d %H:%M"),
                "direction": direction,
                "entry": entry,
                "exit": round(exit_price, 5),
                "sl": trade.signal.stop_loss,
                "tp": trade.signal.take_profit,
                "lot": trade.lot,
                "result": result_type,
                "pnl": pnl,
                "rr": trade.signal.rr_ratio,
            }

        if direction == "BUY":
            # --- Gap check at open ---
            if o <= sl:
                return _make_result(o, "SL")  # gap through SL, fill at open
            if o >= tp:
                return _make_result(o, "TP")  # gap through TP, fill at open

            # --- Check SL first (conservative) ---
            sl_hit = l <= sl
            tp_hit = h >= tp

            if sl_hit and tp_hit:
                # Ambiguous: ALWAYS assume SL hit first (worst case)
                return _make_result(sl, "SL")
            if sl_hit:
                exit_price = sl
                # If SL was trailed above entry, label TSL
                result_type = "TSL" if sl > trade.signal.stop_loss else "SL"
                return _make_result(exit_price, result_type)
            if tp_hit:
                return _make_result(tp, "TP")

            # --- Trail update using candle close (conservative) ---
            if ts_enabled:
                trade.best_price = max(trade.best_price, c)  # use close, not high
                profit = trade.best_price - entry
                if trade.trail_phase < 3 and profit >= risk * p3_rr:
                    trade.trail_phase = 3
                elif trade.trail_phase < 2 and profit >= risk * p2_rr:
                    trade.trail_phase = 2
                elif trade.trail_phase < 1 and profit >= risk * be_rr:
                    trade.trail_phase = 1

                if trade.trail_phase == 1:
                    trade.sl = max(trade.sl, entry)  # breakeven
                elif trade.trail_phase == 2:
                    trade.sl = max(trade.sl, trade.best_price - trade.atr * p2_mult)
                elif trade.trail_phase == 3:
                    trade.sl = max(trade.sl, trade.best_price - trade.atr * p3_mult)

        else:  # SELL
            # --- Gap check at open ---
            if o >= sl:
                return _make_result(o, "SL")
            if o <= tp:
                return _make_result(o, "TP")

            # --- Check SL first (conservative) ---
            sl_hit = h >= sl
            tp_hit = l <= tp

            if sl_hit and tp_hit:
                return _make_result(sl, "SL")  # worst case
            if sl_hit:
                exit_price = sl
                result_type = "TSL" if sl < trade.signal.stop_loss else "SL"
                return _make_result(exit_price, result_type)
            if tp_hit:
                return _make_result(tp, "TP")

            # --- Trail update using candle close (conservative) ---
            if ts_enabled:
                trade.best_price = min(trade.best_price, c)  # use close, not low
                profit = entry - trade.best_price
                if trade.trail_phase < 3 and profit >= risk * p3_rr:
                    trade.trail_phase = 3
                elif trade.trail_phase < 2 and profit >= risk * p2_rr:
                    trade.trail_phase = 2
                elif trade.trail_phase < 1 and profit >= risk * be_rr:
                    trade.trail_phase = 1

                if trade.trail_phase == 1:
                    trade.sl = min(trade.sl, entry)
                elif trade.trail_phase == 2:
                    trade.sl = min(trade.sl, trade.best_price + trade.atr * p2_mult)
                elif trade.trail_phase == 3:
                    trade.sl = min(trade.sl, trade.best_price + trade.atr * p3_mult)

        return None  # trade still open

    # ------------------------------------------------------------------
    # Main run loop
    # ------------------------------------------------------------------

    def run(self, symbol: str, h4_candles: List[dict], m30_candles: List[dict],
            m15_candles: List[dict], quiet: bool = False) -> Optional[dict]:
        """Run bulletproof backtest."""
        if not h4_candles or not m30_candles or not m15_candles:
            logger.error("Insufficient data for backtest")
            return None

        logger.info("H4: %d candles, M30: %d candles, M15: %d candles",
                     len(h4_candles), len(m30_candles), len(m15_candles))

        # --- Symbol config ---
        symbol_config = self.config.get("symbols_config", {}).get(symbol, {})
        point = symbol_config.get("point", 0.01)
        contract_size = symbol_config.get("contract_size", 1)
        spread_pips = self.config.get("backtest", {}).get("spread_pips", {}).get(symbol, 50)
        spread = spread_pips * point
        commission_per_lot = self.config.get("backtest", {}).get("commission_usd", {}).get(symbol, 0)
        slippage_pips = self.config.get("backtest", {}).get("slippage_pips", {}).get(symbol, 0)
        slippage = slippage_pips * point

        # Store for _process_candle_for_trade
        self._symbol_config = {
            "point": point,
            "contract_size": contract_size,
            "commission": commission_per_lot,
        }

        logger.info("Symbol Config — Point: %s, Spread: %s pips, Contract: %s, "
                     "Commission: $%s/lot, Slippage: %s pips",
                     point, spread_pips, contract_size, commission_per_lot, slippage_pips)

        # --- State ---
        balance = self.config.get("backtest", {}).get("initial_balance", 1000)
        initial_balance = balance
        peak_balance = balance

        trades: List[dict] = []
        open_trade: Optional[_OpenTrade] = None
        signal_count = 0
        last_signal_idx = -999
        cooldown = self.config.get("strategy", {}).get("cooldown_candles", 3)

        # Daily limits (matching live)
        max_daily_trades = self.config.get("max_daily_trades", 5)
        max_daily_loss_pct = self.config.get("risk", {}).get("max_daily_loss_percent", 10)
        max_drawdown_pct = self.config.get("risk", {}).get("max_drawdown_percent", 30)
        daily_trade_count: Dict[str, int] = {}
        daily_pnl: Dict[str, float] = {}
        daily_start_balance: Dict[str, float] = {}
        halted = False
        daily_limit_hits = 0

        bt_dashboard = BacktestDashboard(self.config) if not quiet else None
        total_candles = len(m30_candles) - 110

        logger.info("Running bulletproof backtest on %d candles...", total_candles)

        # Pre-build time arrays for binary search
        h4_times = [c["time"] for c in h4_candles]
        m15_times = [c["time"] for c in m15_candles]

        for i in range(100, len(m30_candles) - 10):
            if halted:
                break

            candle_time = m30_candles[i]["time"]
            candle_date = self._utc_date_from_ts(candle_time)

            # Initialize daily tracking
            if candle_date not in daily_trade_count:
                daily_trade_count[candle_date] = 0
                daily_pnl[candle_date] = 0.0
                daily_start_balance[candle_date] = balance

            # Dashboard progress
            if not quiet and bt_dashboard:
                current_time = datetime.fromtimestamp(candle_time, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")
                bt_dashboard.show_progress(i - 100, total_candles, current_time, signal_count, len(trades))

            # ============================================================
            # PHASE 1: If a trade is open, try to close it on this candle
            # ============================================================
            if open_trade is not None:
                result = self._process_candle_for_trade(open_trade, m30_candles[i])
                if result is not None:
                    # Trade closed
                    trades.append(result)
                    balance += result["pnl"]

                    # Update daily P/L
                    trade_date = self._utc_date_from_ts(candle_time)
                    daily_pnl[trade_date] = daily_pnl.get(trade_date, 0.0) + result["pnl"]

                    # Peak balance / drawdown check
                    if balance > peak_balance:
                        peak_balance = balance
                    dd = (peak_balance - balance) / peak_balance * 100 if peak_balance > 0 else 0
                    if dd >= max_drawdown_pct:
                        logger.warning("DRAWDOWN HALT: %.1f%% >= %.1f%% — stopping backtest",
                                       dd, max_drawdown_pct)
                        halted = True

                    open_trade = None
                continue  # don't look for new signals while a trade was just processed

            # ============================================================
            # PHASE 2: No open trade — look for a new signal
            # ============================================================

            # Cooldown from last signal
            if i - last_signal_idx < cooldown:
                continue

            # Daily trade limit check
            if daily_trade_count.get(candle_date, 0) >= max_daily_trades:
                continue

            # Daily loss limit check
            day_start_bal = daily_start_balance.get(candle_date, balance)
            day_loss_limit = day_start_bal * (max_daily_loss_pct / 100.0)
            if daily_pnl.get(candle_date, 0.0) < -day_loss_limit:
                continue

            # Need at least one more candle for entry price
            if i + 1 >= len(m30_candles):
                continue

            # Entry at next candle's open (no look-ahead)
            entry_price = m30_candles[i + 1]["open"]

            # Historical data slices (only data known at candle i)
            h4_last_idx = self._find_last_index(h4_candles, candle_time)
            if h4_last_idx < 49:
                continue
            h4_data = h4_candles[:h4_last_idx + 1]

            m30_data = m30_candles[:i + 1]

            m15_last_idx = self._find_last_index(m15_candles, candle_time)
            if m15_last_idx < 99:
                continue
            m15_data = m15_candles[:m15_last_idx + 1]

            # Session filter
            utc_hour = datetime.fromtimestamp(candle_time, tz=timezone.utc).hour
            session = StrategyEngine.get_session_from_hour(utc_hour)

            # Run strategy
            signal, _ = self.strategy.analyze(symbol, h4_data, m30_data, m15_data,
                                              entry_price, session=session)
            if not signal:
                continue

            signal_count += 1

            if signal.confidence < self.strategy.min_confidence:
                continue

            last_signal_idx = i

            # --- Apply spread + slippage ---
            if signal.direction == "BUY":
                entry = entry_price + spread + slippage
            else:
                entry = entry_price - spread - slippage

            # ATR for trailing stop
            atr = self.strategy._calculate_atr(m30_candles[:i + 1])

            # --- Risk-based lot sizing ---
            lot = self._calc_lot_size(balance, entry, signal.stop_loss, point, contract_size)

            # Update daily trade count
            daily_trade_count[candle_date] = daily_trade_count.get(candle_date, 0) + 1

            # Open the trade (simulation starts from i+2 onward)
            open_trade = _OpenTrade(
                signal=signal,
                entry_price=entry,
                lot=lot,
                entry_index=i + 1,
                entry_time=m30_candles[i + 1]["time"],
                atr=atr,
            )

        # Close any remaining open trade as "OPEN" (not counted in stats)
        if open_trade is not None:
            d = 1.0 if open_trade.signal.direction == "BUY" else -1.0
            last_price = m30_candles[-1]["close"]
            gross = (last_price - open_trade.entry_price) * d * contract_size * open_trade.lot
            comm = commission_per_lot * open_trade.lot
            trades.append({
                "time": datetime.fromtimestamp(open_trade.entry_time, tz=timezone.utc).strftime("%Y-%m-%d %H:%M"),
                "direction": open_trade.signal.direction,
                "entry": open_trade.entry_price,
                "exit": last_price,
                "sl": open_trade.signal.stop_loss,
                "tp": open_trade.signal.take_profit,
                "lot": open_trade.lot,
                "result": "OPEN",
                "pnl": gross - comm,
                "rr": open_trade.signal.rr_ratio,
            })

        # ==================================================================
        # Compile results
        # ==================================================================
        closed_trades = [t for t in trades if t["result"] in ("TP", "SL", "TSL")]
        open_trades = [t for t in trades if t["result"] == "OPEN"]
        wins = [t for t in closed_trades if t["pnl"] > 0]
        losses = [t for t in closed_trades if t["pnl"] <= 0]
        tp_trades = [t for t in closed_trades if t["result"] == "TP"]
        sl_trades = [t for t in closed_trades if t["result"] == "SL"]
        tsl_trades = [t for t in closed_trades if t["result"] == "TSL"]

        total_profit = sum(t["pnl"] for t in wins)
        total_loss = abs(sum(t["pnl"] for t in losses))

        # Sharpe ratio (trading-days only, skip zero-return days)
        daily_returns: Dict[str, float] = {}
        for t in closed_trades:
            day = t["time"].split()[0]
            daily_returns[day] = daily_returns.get(day, 0) + t["pnl"]
        returns = [v for v in daily_returns.values() if v != 0]
        if len(returns) > 1:
            mean_r = np.mean(returns)
            std_r = np.std(returns)
            sharpe = mean_r / std_r * np.sqrt(252) if std_r > 0 else 0
        else:
            sharpe = 0

        # Count daily limit hits
        for day, count in daily_trade_count.items():
            if count >= max_daily_trades:
                daily_limit_hits += 1

        results = {
            "symbol": symbol,
            "start_date": datetime.fromtimestamp(m30_candles[100]["time"], tz=timezone.utc).strftime("%Y-%m-%d"),
            "end_date": datetime.fromtimestamp(m30_candles[-10]["time"], tz=timezone.utc).strftime("%Y-%m-%d"),
            "initial_balance": initial_balance,
            "final_balance": balance,
            "return_pct": (balance - initial_balance) / initial_balance * 100,
            "total_trades": len(closed_trades),
            "open_trades": len(open_trades),
            "tp_count": len(tp_trades),
            "sl_count": len(sl_trades),
            "tsl_count": len(tsl_trades),
            "winning_trades": len(wins),
            "losing_trades": len(losses),
            "win_rate": len(wins) / len(closed_trades) * 100 if closed_trades else 0,
            "profit_factor": total_profit / total_loss if total_loss > 0 else 0,
            "total_profit": total_profit,
            "total_loss": total_loss,
            "avg_win": total_profit / len(wins) if wins else 0,
            "avg_loss": total_loss / len(losses) if losses else 0,
            "rr_ratio": (total_profit / len(wins)) / (total_loss / len(losses)) if wins and losses else 0,
            "max_drawdown": self._calc_drawdown(closed_trades, initial_balance),
            "max_win_streak": self._calc_streak(closed_trades, True),
            "max_loss_streak": self._calc_streak(closed_trades, False),
            "sharpe_ratio": sharpe,
            "daily_limit_hits": daily_limit_hits,
            "halted": halted,
            "trades": trades,
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

    # ------------------------------------------------------------------
    # Stats helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _calc_drawdown(trades: list, initial: float) -> float:
        """Calculate maximum drawdown percentage from closed trades."""
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
    def _calc_streak(trades: list, winning: bool) -> int:
        """Calculate maximum consecutive win or loss streak."""
        if not trades:
            return 0
        max_s, cur = 0, 0
        for t in trades:
            if (winning and t["pnl"] > 0) or (not winning and t["pnl"] <= 0):
                cur += 1
                max_s = max(max_s, cur)
            else:
                cur = 0
        return max_s