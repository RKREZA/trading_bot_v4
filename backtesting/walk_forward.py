"""
TRADING BOT V4 — Walk-Forward Validator
=======================================
Institutional grade rolling window validation for strategy robustness.
Splits data into 'In-Sample' (IS) for parameter tuning and 'Out-of-Sample' (OOS) for validation.
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Any

from backtesting.backtester import PortfolioBacktester
from core.performance_tracker import PerformanceTracker

logger = logging.getLogger("trading_bot.wfo")

class WalkForwardValidator:
    """
    V4 Walk-Forward Optimization & Validation.
    Ensures that strategy performance isn't a result of over-optimization (curve fitting).
    """

    def __init__(self, config: dict):
        self.config = config

    def run_validation(self, symbol: str, strategies: list, data: dict, 
                       window_weeks: int = 8, test_weeks: int = 2) -> List[Dict]:
        """
        Executes a rolling walk-forward validation across the provided dataset.
        
        Args:
            symbol (str): Trading pair.
            strategies (list): List of strategy instances.
            data (dict): Dict containing 'M1', 'M5', 'M15', 'H1' CandleArrays.
            window_weeks (int): Size of the 'In-Sample' training period.
            test_weeks (int): Size of the 'Out-of-Sample' validation period.
            
        Returns:
            List[Dict]: Performance metrics for each Out-Of-Sample window.
        """
        m5 = data.get("M5")
        if not m5:
            return []

        # Convert timestamps to datetime for window calculation
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
                
            logger.info(f"WFO Window: OOS Validation from {is_end.date()} to {oos_end.date()}")
            
            # Slicing Logic
            is_start_ts = current_is_start.timestamp()
            oos_start_ts = is_end.timestamp()
            oos_end_ts = oos_end.timestamp()
            
            # Extract OOS slices from all timeframes
            oos_data = {}
            for tf, candles in data.items():
                mask = (candles.time >= oos_start_ts) & (candles.time < oos_end_ts)
                oos_data[tf] = candles[mask]

            if len(oos_data.get("M5", [])) < 50:
                current_is_start += timedelta(weeks=test_weeks)
                continue

            # 1. Run In-Sample (Training) for context
            is_data = {}
            for tf, candles in data.items():
                mask = (candles.time >= is_start_ts) & (candles.time < oos_start_ts)
                is_data[tf] = candles[mask]
            
            is_backtester = PortfolioBacktester(self.config)
            is_history, _ = is_backtester.run(symbol, strategies, is_data["M5"], is_data.get("H1"), is_data.get("M15"), is_data.get("M5"), is_data.get("M1"))
            is_profit = sum(t['pnl'] for t in is_history) if is_history else 1.0

            # 2. Run Out-Of-Sample (Validation)
            backtester = PortfolioBacktester(self.config)
            history, equity_history = backtester.run(
                symbol, 
                strategies, 
                oos_data["M5"], 
                oos_data.get("H1"), 
                oos_data.get("M15"), 
                oos_data.get("M5"),
                oos_data.get("M1")
            )
            
            oos_profit = sum(t['pnl'] for t in history) if history else 0.0
            
            # WFO Efficiency Ratio (Step 8.1)
            # Ratio = OOS Profit / IS Profit (normalized by time)
            normalized_is = is_profit / window_weeks
            normalized_oos = oos_profit / test_weeks
            wfo_ratio = normalized_oos / normalized_is if normalized_is > 0 else 0.0

            # Aggregate equity curve
            total_initial = len(strategies) * 1000.0
            portfolio_equity = []
            if equity_history:
                eq_df = pd.DataFrame(equity_history)
                portfolio_equity = eq_df.groupby('time')['equity'].sum().tolist()

            metrics = PerformanceTracker.calculate_metrics(history, total_initial, equity_curve=portfolio_equity)
            
            oos_results.append({
                "window": f"{is_end.date()} -> {oos_end.date()}",
                "oos_profit": round(oos_profit, 2),
                "wfo_ratio": round(wfo_ratio, 2),
                "robustness": "PASSED" if wfo_ratio >= 0.60 else "REJECTED (Curve-Fitted)",
                "metrics": metrics
            })
            
            # Roll window forward
            current_is_start += timedelta(weeks=test_weeks)

        return oos_results

    def summarize_wfo_results(self, results: List[Dict]):
        """Prints a professional summary of WFO performance."""
        if not results:
            print("WFO: No results to summarize.")
            return

        print("\n" + "="*50)
        print("INSTITUTIONAL WALK-FORWARD VALIDATION SUMMARY")
        print("="*50)
        
        profits = [r['metrics'].get('net_profit', 0) for r in results]
        win_rates = [float(r['metrics'].get('win_rate', '0%').replace('%','')) for r in results]
        
        print(f"Total Windows: {len(results)}")
        print(f"Avg Profit/Window: ${np.mean(profits):.2f}")
        print(f"Profit Consistency: {(len([p for p in profits if p > 0]) / len(profits) * 100):.2f}%")
        print(f"Avg Win Rate: {np.mean(win_rates):.2f}%")
        print("="*50 + "\n")
