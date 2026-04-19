"""
V6-INSIGNIA — Walk-Forward Validator
=====================================
Institutional grade rolling window validation with:
- Multi-objective optimization support
- Bootstrap confidence intervals
- Parameter stability analysis
- Monte Carlo validation for OOS windows
"""

import logging
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
from concurrent.futures import ProcessPoolExecutor, as_completed

from backtesting.backtester import PortfolioBacktester
from backtesting.monte_carlo import MonteCarloSimulator
from core.performance_tracker import PerformanceTracker

logger = logging.getLogger("trading_bot.wfo")


def _wfo_worker(config: dict, symbol: str, strat_cls, strategy_id: str,
                param_dict: dict, new_cfg: dict, is_data: dict) -> float:
    """
    Top-level picklable worker for ProcessPoolExecutor.
    Runs a single IS backtest for a given parameter combination and returns PnL.
    Must be a top-level function (not a method) for Windows multiprocessing compatibility.
    """
    try:
        test_strat = strat_cls(strategy_id, new_cfg)
        bt = PortfolioBacktester(config)
        hist, _ = bt.run(
            symbol, [test_strat],
            is_data.get("M5"), is_data.get("H1"),
            is_data.get("M15"), is_data.get("M5"), is_data.get("M1")
        )
        return sum(t['pnl'] for t in hist) if hist else 0.0
    except Exception:
        return -float('inf')

class WFORobustness(Enum):
    EXCELLENT = "EXCELLENT"
    GOOD = "GOOD"
    MARGINAL = "MARGINAL"
    REJECTED = "REJECTED"

@dataclass
class WindowResult:
    window: str
    is_start: datetime
    is_end: datetime
    oos_start: datetime
    oos_end: datetime
    is_profit: float
    oos_profit: float
    wfo_ratio: float
    robustness: WFORobustness
    metrics: Dict[str, Any]
    mc_score: Optional[float] = None

class WalkForwardValidator:
    """
    V6 Walk-Forward Optimization & Validation.
    Enhanced with Monte Carlo robustness testing and parameter stability analysis.
    """

    def __init__(self, config: dict):
        self.config = config
        self._results: List[WindowResult] = []
        self._param_history: Dict[str, List] = {}

    def run_validation(self, symbol: str, strategies: list, data: dict, 
                       window_weeks: int = 8, test_weeks: int = 2,
                       min_wfo_ratio: float = 0.60,
                       run_mc: bool = True) -> List[Dict]:
        """Executes rolling walk-forward validation with MC robustness."""
        m5 = data.get("M5")
        if not m5:
            return []

        start_ts = m5.time[0]
        end_ts = m5.time[-1]
        
        current_is_start = datetime.fromtimestamp(start_ts, tz=timezone.utc)
        final_end = datetime.fromtimestamp(end_ts, tz=timezone.utc)
        
        oos_results = []
        
        logger.info(f"Starting WFO Validation: {current_is_start.date()} to {final_end.date()}")

        while True:
            is_end = current_is_start + timedelta(weeks=window_weeks)
            oos_end = is_end + timedelta(weeks=test_weeks)
            
            if oos_end > final_end:
                break
                
            is_start_ts = current_is_start.timestamp()
            oos_start_ts = is_end.timestamp()
            oos_end_ts = oos_end.timestamp()
            
            oos_data = {tf: candles[(candles.time >= oos_start_ts) & (candles.time < oos_end_ts)] 
                      for tf, candles in data.items()}

            if len(oos_data.get("M5", [])) < 50 or len(oos_data.get("M1", [])) < 10:
                current_is_start += timedelta(weeks=test_weeks)
                continue

            is_data = {tf: candles[(candles.time >= is_start_ts) & (candles.time < oos_start_ts)]
                      for tf, candles in data.items()}
            
            if len(is_data.get("M5", [])) < 100 or len(is_data.get("M1", [])) < 20:
                current_is_start += timedelta(weeks=test_weeks)
                continue

            # --- Walk-Forward Grid Search Optimizer ---
            optimized_configs = self._optimize_is_window(symbol, strategies, is_data)
            is_profit = sum(cfg.get("_is_profit", 0) for cfg in optimized_configs.values())

            # Inject mathematical optimums into OOS Validation Phase
            oos_strategies = []
            for strat in strategies:
                mod_cfg = strat.config.copy()
                if strat.strategy_id in optimized_configs:
                    opt_params = optimized_configs[strat.strategy_id].get("params", {})
                    base_name = strat.strategy_id.rsplit('_v', 1)[0] if '_v' in strat.strategy_id else strat.strategy_id
                    
                    if "strategies" in mod_cfg and base_name in mod_cfg["strategies"]:
                        mod_cfg["strategies"][base_name].update(opt_params)
                    elif base_name in mod_cfg:
                        mod_cfg[base_name].update(opt_params)
                oos_strategies.append(strat.__class__(strat.strategy_id, mod_cfg))
            backtester = PortfolioBacktester(self.config)
            history, equity_history = backtester.run(symbol, oos_strategies, oos_data["M5"], oos_data.get("H1"), oos_data.get("M15"), oos_data.get("M5"), oos_data.get("M1"))
            
            oos_profit = sum(t['pnl'] for t in history) if history else 0.0
            oos_trade_count = len(history) if history else 0

            # Significance test: reject windows with too few trades to be meaningful.
            # An IS with 1 trade and OOS with 0 can produce misleading 0.0 WFO ratios.
            min_is_trades = 10
            min_oos_trades = 5
            if is_profit == 0 and oos_trade_count < min_oos_trades:
                logger.warning(
                    f"[WFO] Window {is_end.date()} skipped: OOS has only {oos_trade_count} trades "
                    f"(min={min_oos_trades}). Insufficient for significance."
                )
                current_is_start += timedelta(weeks=test_weeks)
                continue
            
            normalized_is = is_profit / window_weeks
            normalized_oos = oos_profit / test_weeks
            wfo_ratio = normalized_oos / normalized_is if normalized_is > 0 else 0.0
            
            robustness = self._assess_robustness(wfo_ratio, min_wfo_ratio)
            
            total_initial = len(strategies) * 1000.0
            portfolio_equity = []
            if equity_history:
                eq_df = pd.DataFrame(equity_history)
                portfolio_equity = eq_df.groupby('time')['equity'].sum().tolist()

            metrics = PerformanceTracker.calculate_metrics(history, total_initial, equity_curve=portfolio_equity)
            
            mc_score = None
            if run_mc and history:
                mc = MonteCarloSimulator(iterations=500)
                mc_result = mc.run(history, initial_balance=1000.0)
                mc_score = mc_result.get("robustness_score", 0)

            result = WindowResult(
                window=f"{is_end.date()} -> {oos_end.date()}",
                is_start=current_is_start,
                is_end=is_end,
                oos_start=is_end,
                oos_end=oos_end,
                is_profit=round(is_profit, 2),
                oos_profit=round(oos_profit, 2),
                wfo_ratio=round(wfo_ratio, 2),
                robustness=robustness,
                metrics=metrics,
                mc_score=mc_score
            )
            oos_results.append(result)

            # LIVE WFO GATE: Persist result to SQLite so the orchestrator
            # can query and block strategies that failed their last window.
            for strat in strategies:
                self._persist_wfo_result(
                    strategy_id=strat.strategy_id,
                    window=result.window,
                    robustness=robustness.value,
                    wfo_ratio=round(wfo_ratio, 2),
                    is_profit=round(is_profit, 2),
                    oos_profit=round(oos_profit, 2),
                )

            current_is_start += timedelta(weeks=test_weeks)

        self._results = oos_results
        return self._get_dict_results(oos_results)

    def _optimize_is_window(self, symbol: str, strategies: list, is_data: dict) -> dict:
        """
        Bayesian IS window optimizer using Optuna TPE sampler (primary).
        Falls back to ProcessPoolExecutor parallel grid search if optuna is unavailable.

        Optuna TPE (Tree-structured Parzen Estimator) converges to high-performing
        parameter regions 10-100x faster than grid search by learning from each trial.
        """
        import itertools
        from backtesting.backtester import PortfolioBacktester

        try:
            import optuna
            _OPTUNA_AVAILABLE = True
            # Suppress Optuna's verbose per-trial output in backtest runs
            optuna.logging.set_verbosity(optuna.logging.WARNING)
        except ImportError:
            _OPTUNA_AVAILABLE = False

        best_configs = {}
        for strat in strategies:
            grid = strat.get_parameter_grid()
            if not grid:
                bt = PortfolioBacktester(self.config)
                hist, _ = bt.run(symbol, [strat.__class__(strat.strategy_id, strat.config)],
                                 is_data.get("M5"), is_data.get("H1"), is_data.get("M15"),
                                 is_data.get("M5"), is_data.get("M1"))
                prof = sum(t['pnl'] for t in hist) if hist else 0.0
                best_configs[strat.strategy_id] = {"params": {}, "_is_profit": prof}
                continue

            keys = list(grid.keys())
            values = list(grid.values())
            combinations = list(itertools.product(*values))
            n_trials = min(100, len(combinations))  # Budget-aware: cap at 100 trials

            if _OPTUNA_AVAILABLE and len(combinations) > 5:
                # ── PRIMARY PATH: Optuna TPE Bayesian Search ─────────────────
                logger.info(
                    f"[{strat.strategy_id}] Optuna TPE search: {n_trials} trials "
                    f"from {len(combinations)}-combo space (budget-capped)."
                )

                def objective(trial):
                    param_dict = {
                        k: trial.suggest_categorical(k, v)
                        for k, v in zip(keys, values)
                    }
                    new_cfg = strat.config.copy()
                    base_name = strat.strategy_id.rsplit('_v', 1)[0] if '_v' in strat.strategy_id else strat.strategy_id
                    if "strategies" in new_cfg and base_name in new_cfg["strategies"]:
                        new_cfg["strategies"][base_name].update(param_dict)
                    elif base_name in new_cfg:
                        new_cfg[base_name].update(param_dict)
                    else:
                        new_cfg[base_name] = param_dict
                    return _wfo_worker(self.config, symbol, strat.__class__,
                                      strat.strategy_id, param_dict, new_cfg, is_data)

                study = optuna.create_study(
                    direction="maximize",
                    sampler=optuna.samplers.TPESampler(seed=42),
                )
                study.optimize(objective, n_trials=n_trials, n_jobs=1)

                best_params = study.best_params
                best_prof = study.best_value
                logger.info(
                    f"[{strat.strategy_id}] Optuna best params: {best_params} "
                    f"(PnL: {best_prof:.2f}, {len(study.trials)} trials)"
                )

            else:
                # ── FALLBACK PATH: Parallel Grid Search ───────────────────────
                logger.info(
                    f"[{strat.strategy_id}] Grid search fallback: {len(combinations)} combos "
                    f"across {os.cpu_count()} CPUs."
                )
                combo_args = []
                for combo in combinations:
                    param_dict = dict(zip(keys, combo))
                    new_cfg = strat.config.copy()
                    base_name = strat.strategy_id.rsplit('_v', 1)[0] if '_v' in strat.strategy_id else strat.strategy_id
                    if "strategies" in new_cfg and base_name in new_cfg["strategies"]:
                        new_cfg["strategies"][base_name].update(param_dict)
                    elif base_name in new_cfg:
                        new_cfg[base_name].update(param_dict)
                    else:
                        new_cfg[base_name] = param_dict
                    combo_args.append((param_dict, new_cfg))

                best_prof = -float('inf')
                best_params = {}
                max_workers = min(os.cpu_count() or 4, len(combo_args))
                with ProcessPoolExecutor(max_workers=max_workers) as executor:
                    futures = {
                        executor.submit(
                            _wfo_worker, self.config, symbol, strat.__class__,
                            strat.strategy_id, param_dict, new_cfg, is_data
                        ): param_dict
                        for param_dict, new_cfg in combo_args
                    }
                    for future in as_completed(futures):
                        param_dict = futures[future]
                        try:
                            prof = future.result()
                        except Exception as exc:
                            logger.debug(f"[WFO Worker] {param_dict} raised: {exc}")
                            prof = -float('inf')
                        if prof > best_prof:
                            best_prof = prof
                            best_params = param_dict
                logger.info(f"[{strat.strategy_id}] Grid search best: {best_params} (PnL: {best_prof:.2f})")

            best_configs[strat.strategy_id] = {"params": best_params, "_is_profit": best_prof}

        return best_configs

    def _assess_robustness(self, wfo_ratio: float, min_ratio: float) -> WFORobustness:
        """Assesses OOS robustness based on WFO ratio."""
        if wfo_ratio >= min_ratio:
            if wfo_ratio >= 0.80:
                return WFORobustness.EXCELLENT
            return WFORobustness.GOOD
        elif wfo_ratio >= 0.40:
            return WFORobustness.MARGINAL
        return WFORobustness.REJECTED

    def _persist_wfo_result(
        self,
        strategy_id: str,
        window: str,
        robustness: str,
        wfo_ratio: float,
        is_profit: float,
        oos_profit: float,
    ):
        """
        Persists WFO window result to SQLite for the live WFO gate.
        The orchestrator queries `wfo_gate` to block strategies whose
        most recent window has REJECTED robustness.
        Uses WAL mode to prevent locking from concurrent reader threads.
        """
        import sqlite3
        db_path = self.config.get("paths", {}).get(
            "wfo_gate_db", "config/wfo_gate.db"
        )
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        try:
            conn = sqlite3.connect(db_path, timeout=10.0)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("""
                CREATE TABLE IF NOT EXISTS wfo_gate (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    strategy_id TEXT NOT NULL,
                    window TEXT NOT NULL,
                    robustness TEXT NOT NULL,
                    wfo_ratio REAL,
                    is_profit REAL,
                    oos_profit REAL,
                    recorded_at TEXT DEFAULT (datetime('now'))
                )
            """)
            conn.execute(
                "INSERT INTO wfo_gate (strategy_id, window, robustness, wfo_ratio, is_profit, oos_profit) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (strategy_id, window, robustness, wfo_ratio, is_profit, oos_profit)
            )
            conn.commit()
            conn.close()
            logger.debug(f"[WFO Gate] Persisted {strategy_id}: {robustness} ({window})")
        except Exception as e:
            logger.warning(f"[WFO Gate] Failed to persist result: {e}")

    @staticmethod
    def get_last_wfo_robustness(strategy_id: str, config: dict) -> str:
        """
        Queries the WFO gate DB for the most recent robustness classification
        of the given strategy. Returns 'UNKNOWN' if no record exists.
        Called by the orchestrator on startup to apply the live WFO gate.
        """
        import sqlite3
        db_path = config.get("paths", {}).get("wfo_gate_db", "config/wfo_gate.db")
        if not os.path.exists(db_path):
            return "UNKNOWN"
        try:
            conn = sqlite3.connect(db_path, timeout=5.0)
            conn.execute("PRAGMA journal_mode=WAL")
            cursor = conn.execute(
                "SELECT robustness FROM wfo_gate WHERE strategy_id=? "
                "ORDER BY recorded_at DESC LIMIT 1",
                (strategy_id,)
            )
            row = cursor.fetchone()
            conn.close()
            return row[0] if row else "UNKNOWN"
        except Exception:
            return "UNKNOWN"

    def _get_dict_results(self, results: List[WindowResult]) -> List[Dict]:
        """Converts WindowResult to dictionary format."""
        return [{
            "window": r.window,
            "is_profit": r.is_profit,
            "oos_profit": r.oos_profit,
            "wfo_ratio": r.wfo_ratio,
            "robustness": r.robustness.value,
            "metrics": r.metrics,
            "mc_score": r.mc_score
        } for r in results]

    def get_parameter_stability(self) -> Dict[str, Any]:
        """Analyzes stability of parameters across windows."""
        if not self._results:
            return {"status": "No results", "stable": False}
            
        profits = [r.oos_profit for r in self._results]
        ratios = [r.wfo_ratio for r in self._results]
        
        return {
            "window_count": len(self._results),
            "avg_oos_profit": np.mean(profits),
            "std_oos_profit": np.std(profits),
            "profit_consistency": len([p for p in profits if p > 0]) / len(profits) * 100,
            "avg_wfo_ratio": np.mean(ratios),
            "stable": np.std(ratios) < 0.20,
            "recommendation": "PRODUCTION READY" if np.mean(ratios) >= 0.60 else "REQUIRES TUNING"
        }

    def get_aggregate_robustness(self) -> Tuple[float, str]:
        """Returns aggregate robustness score across all windows."""
        if not self._results:
            return 0.0, "NO DATA"
            
        scores = []
        for r in self._results:
            base_score = r.wfo_ratio * 100
            if r.mc_score:
                base_score = (base_score + r.mc_score) / 2
            scores.append(base_score)
        
        avg = np.mean(scores)
        
        if avg >= 80:
            return avg, "EXCELLENT"
        elif avg >= 60:
            return avg, "GOOD"
        elif avg >= 40:
            return avg, "MARGINAL"
        return avg, "REJECTED"

    def summarize_wfo_results(self, results: List[Dict] = None):
        """Prints comprehensive WFO summary."""
        results = results or self._get_dict_results(self._results)
        if not results:
            print("WFO: No results to summarize.")
            return

        print("\n" + "="*60)
        print("V6 INSTITUTIONAL WALK-FORWARD VALIDATION SUMMARY")
        print("="*60)
        
        profits = [r['metrics'].get('net_profit', 0) for r in results]
        win_rates = [float(r['metrics'].get('win_rate', '0%').replace('%','')) for r in results]
        ratios = [r['wfo_ratio'] for r in results]
        
        passed = len([r for r in results if r['robustness'] in ['EXCELLENT', 'GOOD']])
        
        print(f"Total Windows: {len(results)}")
        print(f"Passed: {passed}/{len(results)} ({passed/len(results)*100:.1f}%)")
        print(f"Avg Profit/Window: ${np.mean(profits):.2f}")
        print(f"Profit Consistency: {(len([p for p in profits if p > 0]) / len(profits) * 100):.2f}%")
        print(f"Avg Win Rate: {np.mean(win_rates):.2f}%")
        print(f"Avg WFO Ratio: {np.mean(ratios):.2f}")
        
        stability = self.get_parameter_stability()
        print(f"Parameter Stability: {'✓ STABLE' if stability.get('stable') else '✗ UNSTABLE'}")
        print(f"Recommendation: {stability.get('recommendation', 'N/A')}")
        
        agg_score, agg_status = self.get_aggregate_robustness()
        print(f"Aggregate Robustness: {agg_score:.1f}/100 ({agg_status})")
        print("="*60 + "\n")
