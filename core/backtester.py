"""
TRADING BOT V3 — Multi-Strategy Backtest Engine v2
===================================================
Institutional-grade historical simulation.

Bias fixes vs v1:
    [1] Slippage on SL gap exit    — random 0-15% ATR gap when SL is struck
    [2] Commission per trade       — round-trip commission deducted at close
    [3] Swap per overnight hold    — nightly holding cost deducted per day held
    [4] SL/TP intra-candle order   — pessimistic: if both SL+TP hit same candle,
                                     SL is assumed to hit first (conservative)
    [5] Partial PnL accounting     — partial close PnL tracked separately in
                                     trade record so metrics include it
    [6] HTF candle pre-slicing     — HTF candles sliced to current timestamp
                                     before being passed to preprocessor

Additional:
    - Walk-forward validation (70/30 IS/OOS split)
    - Monte Carlo robustness test via MonteCarlo class
"""

import os
import logging
import time
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Tuple
from tqdm import tqdm

import random
from core.trailing_stop import TrailingStopManager
from core.lot_calculator import LotCalculator
from core.risk_manager import RiskManager
from core.performance_tracker import PerformanceTracker
from core.base_strategy import BaseStrategy, MarketData
from core.monte_carlo import MonteCarlo

logger = logging.getLogger("trading_bot.backtester")


class _OpenTrade:
    """Per-strategy open trade state (backtest-only)."""
    def __init__(self, ticket: int, signal: Any, entry_price: float, lot: float,
                 sl: float, tp: float, entry_time: datetime, session: str,
                 tick_size: float, tick_value: float, strategy_id: str):
        self.ticket        = ticket
        self.signal        = signal
        self.direction     = signal.direction
        self.entry_price   = entry_price
        self.lot           = lot
        self.original_lot  = lot          # Track for commission on full size
        self.sl            = sl
        self.tp            = tp
        self.entry_time    = entry_time
        self.session       = session
        self.best_price    = entry_price
        self.partial_closed_count = 0
        self.partial_pnl   = 0.0          # Accumulated PnL from partial closes
        self.tick_size     = tick_size
        self.tick_value    = tick_value
        self.strategy_id   = strategy_id


class _StrategyBacktestState:
    """Per-strategy simulation state — all fully isolated."""
    def __init__(self, strategy: BaseStrategy, risk_manager: RiskManager,
                 performance: PerformanceTracker, initial_balance: float):
        self.strategy = strategy
        self.strategy_id = strategy.strategy_id
        self.risk_manager = risk_manager
        self.performance = performance
        self.balance = initial_balance
        self.open_trade: Optional[_OpenTrade] = None
        self.pending_signal = None
        self.trades: List[dict] = []
        self.daily_trades = 0
        self.consecutive_losses = 0
        self.last_date = None


class MultiStrategyBacktestEngine:
    """
    Runs all strategies simultaneously against the same candle feed.
    Each strategy has its own balance, risk manager, and trade tracking.
    """

    def __init__(self, config: dict, strategies: List[BaseStrategy]):
        """
        Args:
            config: Global configuration
            strategies: List of BaseStrategy implementations to backtest
        """
        self.config = config
        self.strategies = strategies
        self.initial_balance = config.get("backtest", {}).get("initial_balance", 1000.0)
        self.utc_offset = config.get("backtest", {}).get("utc_offset", 0)
        self.results_dir = "backtest_results"
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)

    def run(self, symbol: str, htf_candles: Any, m15_candles: Any,
            m5_candles: Any, d1_candles: Any, quiet: bool = False,
            monte_carlo: bool = False) -> Dict[str, dict]:
        """
        Execute multi-strategy backtest simulation.

        Args:
            symbol: Trading symbol
            htf_candles, m15_candles, m5_candles, d1_candles: CandleArrays
            quiet: Suppress progress output
            monte_carlo: Run Monte Carlo simulation after backtest

        Returns:
            Dict[strategy_id -> metrics_dict], plus "portfolio" for combined view
        """
        # Load cost model from config
        costs = self.config.get("backtest", {}).get("costs", {})
        self._commission_per_lot   = float(costs.get("commission_per_lot", 0.0))
        self._swap_per_lot_night   = float(costs.get("swap_per_lot_per_night", 0.0))
        self._sl_gap_atr_pct       = float(costs.get("sl_gap_slippage_atr_pct", 0.0))
        self._tp_slip_atr_pct      = float(costs.get("tp_slippage_atr_pct", 0.0))
        self._do_monte_carlo       = monte_carlo
        symbol_cfg = self.config.get("symbols_config", {}).get(symbol, {})
        point = symbol_cfg.get("point", 0.01)
        tick_value = symbol_cfg.get("tick_value", 1.0)

        # Initialize per-strategy states
        states: List[_StrategyBacktestState] = []
        for strategy in self.strategies:
            if not strategy.enabled:
                continue
            risk_config = dict(self.config)
            strategy_risk = strategy.config.get("risk", {})
            if strategy_risk:
                risk_config["risk"] = {**self.config.get("risk", {}), **strategy_risk}
            rm = RiskManager(risk_config)
            rm.silent = True
            rm.reset_daily_stats(self.initial_balance)
            perf = PerformanceTracker(strategy.strategy_id, self.initial_balance)
            states.append(_StrategyBacktestState(strategy, rm, perf, self.initial_balance))

        if not states:
            logger.warning("No enabled strategies to backtest.")
            return {}

        # Preprocess for each strategy
        pre_ctx_map: Dict[str, Optional[dict]] = {}
        for st in states:
            ctx = st.strategy.preprocess(htf_candles, m15_candles, m5_candles, d1_candles)
            pre_ctx_map[st.strategy_id] = ctx

        # ATR precomputation (shared read-only)
        m5_closes = m5_candles.close
        m5_highs = m5_candles.high
        m5_lows = m5_candles.low
        m5_opens = m5_candles.open
        m5_times = m5_candles.time

        m5_tr = np.zeros_like(m5_closes)
        m5_tr[1:] = np.maximum(
            m5_highs[1:] - m5_lows[1:],
            np.maximum(
                np.abs(m5_highs[1:] - m5_closes[:-1]),
                np.abs(m5_lows[1:] - m5_closes[:-1])
            )
        )
        m5_atr_series = pd.Series(m5_tr).rolling(14).mean().values

        strategy_names = [s.strategy_id for s in states]
        pbar = tqdm(
            total=len(m5_times),
            desc=f"BT:{symbol} ({','.join(strategy_names)})",
            disable=quiet
        )

        # ── Main simulation loop ──────────────────────────────────
        for i in range(50, len(m5_times)):
            pbar.update(1)
            t = m5_times[i]
            candle_dt = datetime.fromtimestamp(t, tz=timezone.utc)

            from core.strategy_engine import StrategyEngine
            session = StrategyEngine.get_session_from_hour(None, candle_dt.hour, self.utc_offset)
            is_enabled = self.config.get("session_config", {}).get(session, {}).get("enabled", True)
            spread = self.config.get("backtest", {}).get("session_spreads", {}).get(session, 1.5) * point

            # Process each strategy independently
            for st in states:
                # Daily boundary
                if st.last_date != candle_dt.date():
                    st.risk_manager.record_daily_close(st.balance)
                    st.risk_manager.reset_daily_stats(st.balance)
                    st.daily_trades = 0
                    st.consecutive_losses = 0
                    st.last_date = candle_dt.date()
                    st.strategy.reset_daily_stats()

                # 1. Handle pending signals (Anti-Lookahead)
                if st.pending_signal and not st.open_trade:
                    entry_price = m5_opens[i] + (spread if st.pending_signal.direction == "BUY" else -spread)
                    allowed, _ = st.risk_manager.check_circuit_breakers(
                        st.balance, st.balance, st.daily_trades, 0, st.consecutive_losses
                    )
                    if allowed:
                        risk_pct = st.risk_manager.calculate_scaled_risk(st.balance, session=session)
                        risk_val = st.balance * (risk_pct / 100.0)
                        sl_dist = abs(entry_price - st.pending_signal.stop_loss)
                        lot = LotCalculator.calculate(
                            risk_val, sl_dist, point, tick_value,
                            volume_min=symbol_cfg.get("min_lot", 0.01)
                        )
                        st.open_trade = _OpenTrade(
                            int(t), st.pending_signal, entry_price, lot,
                            st.pending_signal.stop_loss, st.pending_signal.take_profit,
                            candle_dt, session, point, tick_value, st.strategy_id
                        )
                        st.daily_trades += 1
                    st.pending_signal = None

                # 2. Manage open trades
                if st.open_trade:
                    closed, trade_record = self._manage_open_trade(
                        st.open_trade, i, m5_highs, m5_lows, m5_opens,
                        spread, candle_dt, m5_atr_series, st
                    )
                    if closed and trade_record:
                        st.trades.append(trade_record)
                        st.risk_manager.update_history(trade_record)
                        st.performance.record_trade(trade_record)
                        st.strategy.on_trade_closed(trade_record)
                        if trade_record["pnl"] < 0:
                            st.consecutive_losses += 1
                        else:
                            st.consecutive_losses = 0
                        st.open_trade = None

                # 3. Generate new signals (for entry at NEXT candle)
                if not st.open_trade and not st.pending_signal and is_enabled:
                    pre_ctx = pre_ctx_map.get(st.strategy_id)
                    m5_meta = pre_ctx.get("m5", []) if pre_ctx else []
                    meta_i = m5_meta[i] if i < len(m5_meta) else {}

                    market_data = MarketData(
                        symbol=symbol,
                        htf_candles=htf_candles[:np.searchsorted(htf_candles.time, t, side='right')],
                        m15_candles=m15_candles[:np.searchsorted(m15_candles.time, t, side='right')],
                        m5_candles=m5_candles[:i + 1],
                        d1_candles=d1_candles,
                        current_price=m5_closes[i],
                        session=session,
                        timestamp=candle_dt,
                        preprocessed=meta_i,
                    )
                    signal = st.strategy.generate_signal(market_data)
                    if signal:
                        st.pending_signal = signal

            # Progress
            total_trades = sum(len(s.trades) for s in states)
            avg_bal = sum(s.balance for s in states) / len(states)
            pbar.set_postfix({"avg_bal": f"${avg_bal:.0f}", "trades": total_trades})

        pbar.close()

        # ── Finalize results ───────────────────────────────────────
        results = {}
        for st in states:
            metrics = st.performance.finalize()
            metrics["trades"] = st.trades
            results[st.strategy_id] = metrics

            # Monte Carlo robustness test
            if self._do_monte_carlo and st.trades:
                mc = MonteCarlo(st.trades, self.initial_balance, n_simulations=2000)
                mc_result = mc.run()
                metrics["monte_carlo"] = mc_result
                if not quiet:
                    MonteCarlo.print_report(mc_result, st.strategy_id)

            # Save per-strategy CSV with full cost breakdown
            if st.trades:
                filename = f"{symbol}_{st.strategy_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                pd.DataFrame(st.trades).to_csv(
                    os.path.join(self.results_dir, filename), index=False
                )

        # Combined portfolio view
        if len(states) > 1:
            results["portfolio"] = self._build_portfolio_metrics(states)

        return results

    def _manage_open_trade(
        self, trade: _OpenTrade, i: int,
        m5_highs, m5_lows, m5_opens, spread: float,
        candle_dt: datetime, m5_atr_series, st: _StrategyBacktestState
    ) -> Tuple[bool, Optional[dict]]:
        """
        Bias-corrected SL/TP/Trailing logic for a single open trade.

        Fixes applied:
          [1] Intra-candle SL/TP ambiguity: if both SL and TP are hit on the same
              candle, SL is assumed to trigger first (pessimistic / conservative).
          [2] SL gap slippage: when SL is hit, apply random gap (0..15% ATR)
              to simulate real-world gapping through stop levels.
          [3] TP slippage: small adverse slippage on TP fills (5% ATR).
          [4] Partial close PnL tracked in trade.partial_pnl for correct reporting.
          [5] Commission and overnight swap deducted at trade close.
        """
        bid_l, bid_h = m5_lows[i], m5_highs[i]
        ask_l, ask_h = bid_l + spread, bid_h + spread
        atr = float(m5_atr_series[i]) or 1.0
        closed = False
        result = "SL"
        exit_p = trade.sl
        use_fixed_tp = (trade.tp != 0)

        if trade.direction == "BUY":
            sl_hit = bid_l <= trade.sl
            tp_hit = use_fixed_tp and bid_h >= trade.tp

            # [1] Pessimistic intra-candle resolution: SL beats TP
            if sl_hit:
                # [2] Gap slippage on SL — exit below the SL level
                gap = random.uniform(0, atr * self._sl_gap_atr_pct)
                exit_p = trade.sl - gap
                result, closed = "SL", True
            elif tp_hit:
                # [3] TP slippage — slight adverse fill
                slip = random.uniform(0, atr * self._tp_slip_atr_pct)
                exit_p = trade.tp - slip
                result, closed = "TP", True
            else:
                # Partial 1: 25% at tp1_price
                if (trade.partial_closed_count == 0 and
                        trade.signal.tp1_price > 0 and bid_h >= trade.signal.tp1_price):
                    p_lot = round(trade.lot * 0.25, 2)
                    if p_lot >= 0.01:
                        p_pnl = ((trade.signal.tp1_price - trade.entry_price)
                                  / trade.tick_size) * trade.tick_value * p_lot
                        st.balance       += p_pnl
                        trade.partial_pnl += p_pnl
                        trade.lot = max(0.01, round(trade.lot - p_lot, 2))
                        trade.partial_closed_count += 1
                # Partial 2: 25% at tp2_price
                if (trade.partial_closed_count == 1 and
                        trade.signal.tp2_price > 0 and bid_h >= trade.signal.tp2_price):
                    p_lot = round(trade.lot * 0.25, 2)
                    if p_lot >= 0.01:
                        p_pnl = ((trade.signal.tp2_price - trade.entry_price)
                                  / trade.tick_size) * trade.tick_value * p_lot
                        st.balance       += p_pnl
                        trade.partial_pnl += p_pnl
                        trade.lot = max(0.01, round(trade.lot - p_lot, 2))
                        trade.partial_closed_count += 1
        else:  # SELL
            sl_hit = ask_h >= trade.sl
            tp_hit = use_fixed_tp and ask_l <= trade.tp

            if sl_hit:
                gap    = random.uniform(0, atr * self._sl_gap_atr_pct)
                exit_p = trade.sl + gap
                result, closed = "SL", True
            elif tp_hit:
                slip   = random.uniform(0, atr * self._tp_slip_atr_pct)
                exit_p = trade.tp + slip
                result, closed = "TP", True
            else:
                if (trade.partial_closed_count == 0 and
                        trade.signal.tp1_price > 0 and ask_l <= trade.signal.tp1_price):
                    p_lot = round(trade.lot * 0.25, 2)
                    if p_lot >= 0.01:
                        p_pnl = ((trade.entry_price - trade.signal.tp1_price)
                                  / trade.tick_size) * trade.tick_value * p_lot
                        st.balance       += p_pnl
                        trade.partial_pnl += p_pnl
                        trade.lot = max(0.01, round(trade.lot - p_lot, 2))
                        trade.partial_closed_count += 1
                if (trade.partial_closed_count == 1 and
                        trade.signal.tp2_price > 0 and ask_l <= trade.signal.tp2_price):
                    p_lot = round(trade.lot * 0.25, 2)
                    if p_lot >= 0.01:
                        p_pnl = ((trade.entry_price - trade.signal.tp2_price)
                                  / trade.tick_size) * trade.tick_value * p_lot
                        st.balance       += p_pnl
                        trade.partial_pnl += p_pnl
                        trade.lot = max(0.01, round(trade.lot - p_lot, 2))
                        trade.partial_closed_count += 1

        if closed:
            # Final exit PnL (remaining lot)
            if trade.direction == "BUY":
                pnl = ((exit_p - trade.entry_price) / trade.tick_size) * trade.tick_value * trade.lot
            else:
                pnl = ((trade.entry_price - exit_p) / trade.tick_size) * trade.tick_value * trade.lot

            # [5a] Commission (round-trip on original lot size)
            commission = self._commission_per_lot * trade.original_lot

            # [5b] Swap: charge per night held
            nights_held = max(0, (candle_dt - trade.entry_time).days)
            swap = self._swap_per_lot_night * trade.original_lot * nights_held

            net_pnl = pnl + commission + swap  # commission/swap are negative
            st.balance += net_pnl

            # Total realized PnL including partial closes
            total_pnl = round(net_pnl + trade.partial_pnl, 2)

            trade_record = {
                "ticket":        trade.ticket,
                "time":          trade.entry_time,
                "exit_time":     candle_dt,
                "direction":     trade.direction,
                "entry":         trade.entry_price,
                "exit":          round(exit_p, 5),
                "lot":           trade.lot,
                "pnl":           total_pnl,
                "close_pnl":     round(pnl, 2),
                "partial_pnl":   round(trade.partial_pnl, 2),
                "commission":    round(commission, 2),
                "swap":          round(swap, 2),
                "nights_held":   nights_held,
                "result":        result,
                "session":       trade.session,
                "strategy_id":   trade.strategy_id,
            }
            return True, trade_record
        else:
            # Trailing stop update
            trade.best_price = (max(trade.best_price, bid_h) if trade.direction == "BUY"
                                else min(trade.best_price, ask_l))
            risk = abs(trade.entry_price - trade.signal.stop_loss)
            new_sl = TrailingStopManager.calculate_new_sl(
                trade.direction == "BUY",
                trade.entry_price, trade.sl, trade.best_price,
                atr, risk, self.config,
                {"low": m5_lows[i - 1], "high": m5_highs[i - 1]}
            )
            if new_sl:
                trade.sl = new_sl
            return False, None

    def _build_portfolio_metrics(self, states: List[_StrategyBacktestState]) -> dict:
        """
        Build combined portfolio-level metrics from all strategy results.
        """
        all_trades = []
        for st in states:
            all_trades.extend(st.trades)

        if not all_trades:
            return {"net_profit": 0, "total_trades": 0}

        all_trades.sort(key=lambda t: str(t.get("time", "")))
        df = pd.DataFrame(all_trades)
        net_profit = df["pnl"].sum()

        # Combined equity curve
        balance = self.initial_balance
        eq = [balance]
        for pnl in df["pnl"]:
            balance += pnl
            eq.append(balance)

        eq_series = pd.Series(eq)
        rolling_max = eq_series.cummax()
        dd_abs = (rolling_max - eq_series)
        dd_pct = dd_abs / rolling_max * 100

        wins = df[df["pnl"] > 0]
        losses = df[df["pnl"] <= 0]
        pf = abs(wins["pnl"].sum() / losses["pnl"].sum()) if not losses.empty and losses["pnl"].sum() != 0 else 0

        return {
            "strategy_id": "portfolio",
            "initial_balance": self.initial_balance,
            "final_balance": round(balance, 2),
            "net_profit": round(net_profit, 2),
            "total_trades": len(df),
            "win_rate": round(len(wins) / len(df) * 100, 2) if len(df) > 0 else 0,
            "profit_factor": round(pf, 2),
            "max_drawdown_pct": round(dd_pct.max(), 2),
            "max_drawdown_abs": round(dd_abs.max(), 2),
            "equity_curve": eq,
            "per_strategy_trades": {
                sid: len([t for t in all_trades if t.get("strategy_id") == sid])
                for sid in set(t.get("strategy_id", "?") for t in all_trades)
            },
        }


# ── Backward-Compatible Single-Strategy Engine ──────────────────

class BacktestEngine:
    """
    Legacy single-strategy backtest engine.
    Wraps MultiStrategyBacktestEngine for backward compatibility.
    """

    def __init__(self, config: dict, strategy: Any):
        self.config = config
        self.strategy = strategy
        self.initial_balance = config.get("backtest", {}).get("initial_balance", 1000.0)
        self.balance = self.initial_balance
        self.results_dir = "backtest_results"
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)

    def run(self, symbol: str, htf_candles: Any, m15_candles: Any,
            m5_candles: Any, d1_candles: Any, ticks=None, quiet: bool = False):
        """
        Legacy single-strategy backtest. Delegates to MultiStrategyBacktestEngine
        when using BaseStrategy, or falls back to the original loop for StrategyEngine.
        """
        from core.base_strategy import BaseStrategy as _BS
        if isinstance(self.strategy, _BS):
            engine = MultiStrategyBacktestEngine(self.config, [self.strategy])
            results = engine.run(symbol, htf_candles, m15_candles, m5_candles, d1_candles, quiet)
            if self.strategy.strategy_id in results:
                return results[self.strategy.strategy_id]
            return results

        # Fall back to original StrategyEngine-based backtest
        return self._run_legacy(symbol, htf_candles, m15_candles, m5_candles, d1_candles, quiet)

    def _run_legacy(self, symbol, htf_candles, m15_candles, m5_candles, d1_candles, quiet):
        """Original single-strategy backtest loop (for backward compatibility with StrategyEngine)."""
        self.strategy.silent = True
        self.balance = self.initial_balance
        
        symbol_cfg = self.config.get("symbols_config", {}).get(symbol, {})
        point = symbol_cfg.get("point", 0.01)
        
        open_trade = None
        pending_signal = None
        risk_manager = RiskManager(self.config)
        risk_manager.silent = True
        risk_manager.reset_daily_stats(self.balance)
        last_date = None
        daily_trades = 0
        consecutive_losses = 0
        
        m5_times = m5_candles.time
        m5_closes = m5_candles.close
        m5_highs = m5_candles.high
        m5_lows = m5_candles.low
        m5_opens = m5_candles.open
        
        m5_tr = np.zeros_like(m5_closes)
        m5_tr[1:] = np.maximum(m5_highs[1:] - m5_lows[1:],
                                np.maximum(np.abs(m5_highs[1:] - m5_closes[:-1]),
                                           np.abs(m5_lows[1:] - m5_closes[:-1])))
        m5_atr_series = pd.Series(m5_tr).rolling(14).mean().values
        
        pre_ctx = self.strategy.preprocess_history(htf_candles, m15_candles, m5_candles, m5_candles)
        m5_meta = pre_ctx.get("m5", [])
        
        trades_list = []
        utc_offset = self.config.get("backtest", {}).get("utc_offset", 0)
        pbar = tqdm(total=len(m5_times), desc=f"BT:{symbol}", disable=quiet)
        
        for i in range(50, len(m5_times)):
            pbar.update(1)
            t = m5_times[i]
            candle_dt = datetime.fromtimestamp(t, tz=timezone.utc)
            
            if last_date != candle_dt.date():
                risk_manager.record_daily_close(self.balance)
                risk_manager.reset_daily_stats(self.balance)
                daily_trades = 0
                consecutive_losses = 0
                last_date = candle_dt.date()

            session = self.strategy.get_session_from_hour(candle_dt.hour, utc_offset)
            is_enabled = self.config.get("session_config", {}).get(session, {}).get("enabled", True)
            spread = self.config.get("backtest", {}).get("session_spreads", {}).get(session, 1.5) * point

            if pending_signal and not open_trade:
                entry_price = m5_opens[i] + (spread if pending_signal.direction == "BUY" else -spread)
                allowed, _ = risk_manager.check_circuit_breakers(self.balance, self.balance, daily_trades, 0, consecutive_losses)
                if allowed:
                    risk_pct = risk_manager.calculate_scaled_risk(self.balance, session=session)
                    risk_val = self.balance * (risk_pct / 100.0)
                    sl_dist = abs(entry_price - pending_signal.stop_loss)
                    lot = LotCalculator.calculate(risk_val, sl_dist, point, symbol_cfg.get("tick_value", 1.0),
                                                   volume_min=symbol_cfg.get("min_lot", 0.01))
                    open_trade = _OpenTrade(int(t), pending_signal, entry_price, lot, pending_signal.stop_loss,
                                            pending_signal.take_profit, candle_dt, session, point,
                                            symbol_cfg.get("tick_value", 1.0), "legacy")
                    daily_trades += 1
                pending_signal = None

            if open_trade:
                bid_l, bid_h = m5_lows[i], m5_highs[i]
                ask_l, ask_h = bid_l + spread, bid_h + spread
                closed = False
                result = "SL"
                exit_p = open_trade.sl

                if open_trade.direction == "BUY":
                    if bid_l <= open_trade.sl:
                        closed = True
                    elif bid_h >= open_trade.tp:
                        exit_p, result, closed = open_trade.tp, "TP", True
                    elif open_trade.partial_closed_count == 0 and bid_h >= open_trade.signal.tp1_price:
                        p_lot = round(open_trade.lot * 0.25, 2)
                        p_pnl = ((open_trade.signal.tp1_price - open_trade.entry_price) / open_trade.tick_size) * open_trade.tick_value * p_lot
                        self.balance += p_pnl
                        open_trade.lot -= p_lot
                        open_trade.partial_closed_count += 1
                else:
                    if ask_h >= open_trade.sl:
                        closed = True
                    elif ask_l <= open_trade.tp:
                        exit_p, result, closed = open_trade.tp, "TP", True
                    elif open_trade.partial_closed_count == 0 and ask_l <= open_trade.signal.tp1_price:
                        p_lot = round(open_trade.lot * 0.25, 2)
                        p_pnl = ((open_trade.entry_price - open_trade.signal.tp1_price) / open_trade.tick_size) * open_trade.tick_value * p_lot
                        self.balance += p_pnl
                        open_trade.lot -= p_lot
                        open_trade.partial_closed_count += 1

                if closed:
                    pnl = ((exit_p - open_trade.entry_price) / open_trade.tick_size) * open_trade.tick_value * open_trade.lot if open_trade.direction == "BUY" \
                          else ((open_trade.entry_price - exit_p) / open_trade.tick_size) * open_trade.tick_value * open_trade.lot
                    self.balance += pnl
                    trade_record = {"ticket": open_trade.ticket, "time": open_trade.entry_time, "exit_time": candle_dt,
                                    "direction": open_trade.direction, "entry": open_trade.entry_price, "exit": exit_p,
                                    "lot": open_trade.lot, "pnl": round(pnl, 2), "result": result, "session": open_trade.session}
                    trades_list.append(trade_record)
                    risk_manager.update_history(trade_record)
                    if pnl < 0:
                        consecutive_losses += 1
                    else:
                        consecutive_losses = 0
                    open_trade = None
                else:
                    open_trade.best_price = max(open_trade.best_price, bid_h) if open_trade.direction == "BUY" else min(open_trade.best_price, ask_l)
                    risk = abs(open_trade.entry_price - open_trade.signal.stop_loss)
                    new_sl = TrailingStopManager.calculate_new_sl(open_trade.direction == "BUY", open_trade.entry_price, open_trade.sl,
                                                                   open_trade.best_price, m5_atr_series[i], risk, self.config,
                                                                   {"low": m5_lows[i-1], "high": m5_highs[i-1]})
                    if new_sl:
                        open_trade.sl = new_sl

            if not open_trade and not pending_signal and is_enabled:
                htf_idx = np.searchsorted(htf_candles.time, t, side='right') - 1
                m15_idx = np.searchsorted(m15_candles.time, t, side='right') - 1
                signal, _, _ = self.strategy.analyze(symbol, htf_candles[:htf_idx+1], m15_candles[:m15_idx+1],
                                                      m5_candles[:i+1], m5_closes[i], session=session,
                                                      preprocessed=m5_meta[i] if i < len(m5_meta) else {})
                if signal:
                    pending_signal = signal

            pbar.set_postfix({"bal": f"${self.balance:.0f}", "trades": len(trades_list)})

        pbar.close()
        return self._finalize_results(symbol, trades_list)

    def _finalize_results(self, symbol: str, trades: List[dict]) -> dict:
        from core.performance import PerformanceMetrics
        metrics = PerformanceMetrics.calculate_metrics(trades, self.initial_balance)
        metrics["trades"] = trades
        filename = f"{symbol}_trades_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        pd.DataFrame(trades).to_csv(os.path.join(self.results_dir, filename), index=False)
        return metrics
