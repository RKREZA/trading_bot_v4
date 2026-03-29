import random
import copy
from typing import Dict, List
from core.backtester import BacktestEngine
from core.strategy_engine import StrategyEngine

class ValidationSuite:
    def __init__(self, config: dict, strategy: StrategyEngine):
        self.config = config
        self.strategy = strategy

    def run_all_tests(self, symbol: str, h4: List, h1: List, m30: List, m5: List, d1: List) -> Dict:
        results = {}
        
        # Original Backtest
        tester = BacktestEngine(self.config, self.strategy)
        results['base'] = tester.run(symbol, h4, h1, m30, m5, d1, quiet=True)
        
        # 1. Spread Sensitivity Test (2x Spread)
        results['spread_2x'] = self._spread_sensitivity_test(symbol, h4, h1, m30, m5, d1)
        
        # 2. Slippage Stress Test (5x Slippage)
        results['slippage_stress'] = self._slippage_stress_test(symbol, h4, h1, m30, m5, d1)
        
        # 3. Randomized Entry Test
        results['random_entry'] = self._random_entry_test(symbol, h4, h1, m30, m5, d1)
        
        # 4. Data Shuffle Test
        results['data_shuffle'] = self._data_shuffle_test(symbol, h4, h1, m30, m5, d1)
        
        return self._generate_report(results)

    def _spread_sensitivity_test(self, symbol, h4, h1, m30, m5, d1):
        cfg = copy.deepcopy(self.config)
        cfg["symbol_defaults"][symbol]["base_spread"] *= 2.0
        tester = BacktestEngine(cfg, self.strategy)
        return tester.run(symbol, h4, h1, m30, m5, d1, quiet=True)

    def _slippage_stress_test(self, symbol, h4, h1, m30, m5, d1):
        cfg = copy.deepcopy(self.config)
        cfg["symbol_defaults"][symbol]["max_slippage"] *= 5.0
        tester = BacktestEngine(cfg, self.strategy)
        return tester.run(symbol, h4, h1, m30, m5, d1, quiet=True)

    def _random_entry_test(self, symbol, h4, h1, m30, m5, d1):
        class RandomStrategy(StrategyEngine):
            def analyze(self, symbol, h4, h1, m30, m5, current_price, d1_candles=None):
                if random.random() < 0.05: # 5% chance
                    from core.strategy_engine import TradeSignal
                    direction = "BUY" if random.random() > 0.5 else "SELL"
                    # Minimal dummy signal
                    sl = current_price - 10.0 if direction == "BUY" else current_price + 10.0
                    tp = current_price + 20.0 if direction == "BUY" else current_price - 20.0
                    return TradeSignal(symbol, direction, current_price, sl, tp, confidence=50), {}
                return None, {}
        
        tester = BacktestEngine(self.config, RandomStrategy(self.config))
        return tester.run(symbol, h4, h1, m30, m5, d1, quiet=True)

    def _data_shuffle_test(self, symbol, h4, h1, m30, m5, d1):
        m5_shuffled = copy.deepcopy(m5)
        random.shuffle(m5_shuffled) # Shuffle the M5 candles to break sequence
        tester = BacktestEngine(self.config, self.strategy)
        return tester.run(symbol, h4, h1, m30, m5_shuffled, d1, quiet=True)

    def _generate_report(self, results: Dict) -> Dict:
        base_perf = results['base']
        base_profit = base_perf['net_profit']
        
        report = {
            "status": "PASS",
            "warnings": [],
            "metrics": results,
            "structural_checks": self._check_structural_weaknesses(base_perf)
        }
        
        # Validation Logic
        if results['spread_2x']['net_profit'] > base_profit * 0.8 and base_profit > 0:
            report['warnings'].append("High Spread Sensitivity: Strategy is very dependent on low spread")
            
        if results['random_entry']['net_profit'] > base_profit * 0.3 and base_profit > 0:
            report['status'] = "FAIL"
            report['warnings'].append("Random Entry Test: Random signals performed too well (Luck or Bias)")
            
        if report['structural_checks']['failed']:
            report['status'] = "FAIL"
            report['warnings'].extend(report['structural_checks']['reasons'])
            
        return report

    def _check_structural_weaknesses(self, perf: Dict) -> Dict:
        checks = {"failed": False, "reasons": []}
        total_trades = perf.get('total_trades', 0)
        
        if total_trades < 50:
            checks['failed'] = True
            checks['reasons'].append("Sample size too small (<50 trades) for statistical significance")
            
        if perf.get('win_rate', 0) > 85:
            checks['failed'] = True
            checks['reasons'].append("Suspected Curve Fitting: Win rate > 85% is typically unrealistic for this strategy")
            
        if perf.get('max_drawdown', 0) > 30:
            checks['failed'] = True
            checks['reasons'].append("High Max Drawdown (>30%) detected in base backtest")
            
        return checks
