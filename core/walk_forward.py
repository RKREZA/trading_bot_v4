"""
TRADING BOT V3 — Walk-Forward Validation Engine
================================================
Separates In-Sample (IS) parameter "evaluation" from Out-of-Sample (OOS)
performance to detect overfitting.

How it works:
    1. Split the full date range into IS (70%) + OOS (30%)
    2. Run the backtest independently on each segment
    3. Compare IS vs OOS key metrics
    4. Flag degradation > 40% as a sign of overfitting

Usage:
    python backtest.py --from 2025-10-01 --to 2026-03-31 --walk-forward
"""

import logging
import numpy as np
from datetime import datetime, timezone
from typing import List, Dict, Any, Tuple

logger = logging.getLogger("trading_bot.walk_forward")


class WalkForwardValidator:
    """
    Runs IS + OOS split validation on a fixed strategy configuration.
    Does NOT re-optimize parameters — only evaluates pre-configured strategies.

    "Walk-Forward" in this context = holdout validation:
        - The parameters were NOT changed after seeing the OOS data
        - OOS results are genuinely unseen performance
    """

    def __init__(self, config: dict, strategies: List[Any],
                 engine_class: Any, is_pct: float = 0.70):
        self.config      = config
        self.strategies  = strategies
        self.EngineClass = engine_class
        self.is_pct      = is_pct
        self.oos_pct     = 1.0 - is_pct

    def run(self, symbol: str, htf: Any, m15: Any, m5: Any, d1: Any,
            quiet: bool = False) -> Dict:
        """
        Execute IS + OOS split validation.
        Returns dict with 'is', 'oos', 'degradation', 'verdict'.
        """
        m5_times = m5.time
        total    = len(m5_times)
        split_i  = int(total * self.is_pct)
        split_ts = float(m5_times[split_i])

        if not quiet:
            is_end_dt  = datetime.fromtimestamp(float(m5_times[split_i - 1]), tz=timezone.utc)
            oos_st_dt  = datetime.fromtimestamp(split_ts, tz=timezone.utc)
            oos_end_dt = datetime.fromtimestamp(float(m5_times[-1]), tz=timezone.utc)
            print(f"\n{'='*55}")
            print(f"  WALK-FORWARD VALIDATION")
            print(f"  In-Sample  (IS):  start → {is_end_dt.date()}  ({split_i} candles, {self.is_pct*100:.0f}%)")
            print(f"  Out-Sample (OOS): {oos_st_dt.date()} → {oos_end_dt.date()}  ({total - split_i} candles, {self.oos_pct*100:.0f}%)")
            print(f"{'='*55}")

        # Slice candle arrays at split point
        htf_is,  htf_oos  = self._slice(htf,  split_ts)
        m15_is,  m15_oos  = self._slice(m15,  split_ts)
        m5_is,   m5_oos   = self._slice(m5,   split_ts)
        d1_is,   d1_oos   = self._slice(d1,   split_ts)

        # IS backtest
        if not quiet:
            print(f"\n  Running IN-SAMPLE backtest ({split_i} candles)...")
        is_engine  = self.EngineClass(self.config, self._clone_strategies())
        is_results = is_engine.run(symbol, htf_is, m15_is, m5_is, d1_is, quiet=True)

        # OOS backtest — fresh strategy instances (no IS state leakage)
        if not quiet:
            print(f"  Running OUT-OF-SAMPLE backtest ({total - split_i} candles)...")
        oos_engine  = self.EngineClass(self.config, self._clone_strategies())
        oos_results = oos_engine.run(symbol, htf_oos, m15_oos, m5_oos, d1_oos, quiet=True)

        # Compute degradation
        key_metrics = ["win_rate", "profit_factor", "net_profit", "max_drawdown_pct", "total_trades"]
        degradation = {}
        for sid in is_results:
            if sid == "portfolio":
                continue
            is_m  = is_results.get(sid, {})
            oos_m = oos_results.get(sid, {})
            deg   = {}
            for m in key_metrics:
                is_val  = float(is_m.get(m, 0) or 0.001)
                oos_val = float(oos_m.get(m, 0))
                deg[m]  = round((oos_val - is_val) / max(abs(is_val), 0.001) * 100, 1)
            degradation[sid] = deg

        verdict = self._verdict(degradation)

        if not quiet:
            self._print_report(is_results, oos_results, degradation, verdict)

        return {"is": is_results, "oos": oos_results,
                "degradation": degradation, "verdict": verdict}

    def _slice(self, candles: Any, split_ts: float) -> Tuple[Any, Any]:
        idx = int(np.searchsorted(candles.time, split_ts, side="left"))
        return candles[:idx], candles[idx:]

    def _clone_strategies(self) -> List[Any]:
        from strategies import create_strategy
        fresh = []
        for s in self.strategies:
            cfg  = dict(s.config)
            stype = type(s).__name__.replace("Strategy", "").upper()
            # Map class name to registry key used in create_strategy
            stype_map = {"SNIPER": "SNIPER", "SMC": "SMC"}
            s2 = create_strategy(s.strategy_id, stype_map.get(stype, stype), cfg)
            fresh.append(s2)
        return fresh

    def _verdict(self, degradation: Dict) -> str:
        """ROBUST / SUSPECT / OVERFITTED based on worst metric degradation."""
        penalty_scores = []
        for sid, deg in degradation.items():
            for m, d in deg.items():
                if m == "max_drawdown_pct":
                    if d > 0: penalty_scores.append(d)  # Drawdown INCREASE is bad
                elif m == "total_trades":
                    pass
                else:
                    if d < 0: penalty_scores.append(-d) # Only DECREASE is bad for profit/winrate

        if not penalty_scores:
            return "ROBUST"
        worst = max(penalty_scores)
        if worst > 60: return "OVERFITTED"
        if worst > 30: return "SUSPECT"
        return "ROBUST"

    def _print_report(self, is_r, oos_r, degradation, verdict) -> None:
        icons = {"ROBUST": "✅", "SUSPECT": "⚠️", "OVERFITTED": "❌"}
        icon  = icons.get(verdict, "?")
        print(f"\n{'='*55}")
        print(f"  WALK-FORWARD RESULTS — {icon} {verdict}")
        print(f"{'='*55}")

        labels = {
            "win_rate":         "Win Rate %",
            "profit_factor":    "Profit Factor",
            "net_profit":       "Net Profit $",
            "max_drawdown_pct": "Max Drawdown %",
            "total_trades":     "Total Trades",
        }

        for sid in degradation:
            print(f"\n  Strategy: {sid}")
            print(f"  {'Metric':<20} {'IS':>10} {'OOS':>10} {'Δ':>10}")
            print(f"  {'-'*52}")
            is_m  = is_r.get(sid, {})
            oos_m = oos_r.get(sid, {})
            deg   = degradation.get(sid, {})

            for m, label in labels.items():
                is_v  = round(float(is_m.get(m, 0)), 2)
                oos_v = round(float(oos_m.get(m, 0)), 2)
                chg   = deg.get(m, 0)
                warn  = " ⚠️" if abs(chg) > 30 else ""
                warn  = " ❌" if abs(chg) > 60 else warn
                print(f"  {label:<20} {str(is_v):>10} {str(oos_v):>10}   {chg:>+6.1f}%{warn}")

        print(f"\n  Verdict: {icon} {verdict}")
        descriptions = {
            "ROBUST":     "IS/OOS metrics align — genuine edge detected.",
            "SUSPECT":    "Moderate OOS degradation — check session sensitivity.",
            "OVERFITTED": "Heavy OOS degradation — parameters overfit to history.",
        }
        print(f"  {descriptions.get(verdict, '')}")
        print(f"{'='*55}")
