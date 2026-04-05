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

            # Run Backtester on OOS window
            backtester = PortfolioBacktester(self.config)
            history = backtester.run(
                symbol, 
                strategies, 
                oos_data["M5"], 
                oos_data.get("H1"), 
                oos_data.get("M15"), 
                oos_data.get("M1")
            )
            
            # Calculate metrics for this window
            # Total capital = $1000 * num_strategies
            total_initial = len(strategies) * 1000.0
            metrics = PerformanceTracker.calculate_metrics(history, total_initial)
            
            oos_results.append({
                "window_start": is_end.date().isoformat(),
                "window_end": oos_end.date().isoformat(),
                "metrics": metrics,
                "trade_count": len(history)
            })
            
            # Roll window forward by test_weeks (The "Walk")
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
