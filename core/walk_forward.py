from typing import List, Dict
from core.backtester import BacktestEngine
from core.strategy_engine import StrategyEngine

class WalkForwardValidation:
    def __init__(self, config: dict, strategy: StrategyEngine):
        self.config = config
        self.strategy = strategy

    def run_validation(self, symbol: str, h4: List, m30: List, m15: List, train_months: int = 6, test_months: int = 1) -> List[Dict]:
        """
        Implementation of rolling window walk-forward validation.
        For simplicity, we'll assume the data is chronologically ordered and dense.
        """
        results = []
        
        # Estimate number of candles based on timeframe (M30 = 48 candles per day)
        candles_per_month = 48 * 22 # approx 22 trading days
        train_window = train_months * candles_per_month
        test_window = test_months * candles_per_month
        
        start_idx = 0
        while start_idx + train_window + test_window < len(m30):
            train_m30 = m30[start_idx : start_idx + train_window]
            test_m30 = m30[start_idx + train_window : start_idx + train_window + test_window]
            
            # For H4 and M15, we need to find corresponding slices based on time
            train_start_time = train_m30[0]['time']
            train_end_time = train_m30[-1]['time']
            test_end_time = test_m30[-1]['time']
            
            train_h4 = [c for c in h4 if train_start_time <= c['time'] <= train_end_time]
            train_m15 = [c for c in m15 if train_start_time <= c['time'] <= train_end_time]
            
            test_h4 = [c for c in h4 if train_end_time < c['time'] <= test_end_time]
            test_m15 = [c for c in m15 if train_end_time < c['time'] <= test_end_time]
            
            # Run backtest on test window
            tester = BacktestEngine(self.config, self.strategy)
            test_perf = tester.run(symbol, test_h4, test_m30, test_m15)
            
            results.append({
                "window_start": train_m30[0].get('time'),
                "test_start": test_m30[0].get('time'),
                "performance": test_perf
            })
            
            # Slide window by test_window
            start_idx += test_window
            
        return results
