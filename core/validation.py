import random
import copy
from typing import Dict, List
from core.backtester import BacktestEngine
from core.strategy_engine import StrategyEngine

class ValidationSuite:
    def __init__(self, config: dict, strategy: StrategyEngine):
        self.config = config
        self.strategy = strategy

    def run_all_tests(self, symbol: str, h4: List, m30: List, m15: List) -> Dict:
        results = {}
        
        # Original Backtest
        tester = BacktestEngine(self.config, self.strategy)
        results['base'] = tester.run(symbol, h4, m30, m15)
        
        # 1. Spread Sensitivity Test (2x Spread)
        results['spread_2x'] = self._spread_sensitivity_test(symbol, h4, m30, m15)
        
        # 2. Slippage Stress Test (Significantly increased slippage)
        results['slippage_stress'] = self._slippage_stress_test(symbol, h4, m30, m15)
        
        # 3. Randomized Entry Test
        results['random_entry'] = self._random_entry_test(symbol, h4, m30, m15)
        
        # 4. Data Shuffle Test
        results['data_shuffle'] = self._data_shuffle_test(symbol, h4, m30, m15)
        
        return self._generate_report(results)

    def _spread_sensitivity_test(self, symbol, h4, m30, m15):
        cfg = copy.deepcopy(self.config)
        mult = cfg.get("validation", {}).get("spread_sensitivity_multiplier", 2.0)
        cfg["symbol_defaults"][symbol]["base_spread"] *= mult
        tester = BacktestEngine(cfg, self.strategy)
        return tester.run(symbol, h4, m30, m15)

    def _slippage_stress_test(self, symbol, h4, m30, m15):
        cfg = copy.deepcopy(self.config)
        mult = cfg.get("validation", {}).get("slippage_stress_multiplier", 5.0)
        cfg["symbol_defaults"][symbol]["max_slippage"] *= mult
        tester = BacktestEngine(cfg, self.strategy)
        return tester.run(symbol, h4, m30, m15)

    def _random_entry_test(self, symbol, h4, m30, m15):
        # We wrap the strategy to return random signals
        class RandomStrategy(StrategyEngine):
            def analyze(self, symbol, h4, m30, m15, current_price, session=None):
                if random.random() < 0.05: # 5% chance of signal
                    from core.strategy_engine import TradeSignal
                    direction = "BUY" if random.random() > 0.5 else "SELL"
                    sl = current_price - 1.0 if direction == "BUY" else current_price + 1.0
                    tp = current_price + 2.0 if direction == "BUY" else current_price - 2.0
                    return TradeSignal(symbol, direction, current_price, sl, tp), {}
                return None, {}
        
        tester = BacktestEngine(self.config, RandomStrategy(self.config))
        return tester.run(symbol, h4, m30, m15)

    def _data_shuffle_test(self, symbol, h4, m30, m15):
        m30_shuffled = copy.deepcopy(m30)
        random.shuffle(m30_shuffled)
        tester = BacktestEngine(self.config, self.strategy)
        return tester.run(symbol, h4, m30_shuffled, m15)

    def _generate_report(self, results: Dict) -> Dict:
        base_perf = results['base']
        base_profit = base_perf['net_profit']
        
        report = {
            "status": "PASS",
            "warnings": [],
            "metrics": results,
            "structural_checks": self._check_structural_weaknesses(base_perf)
        }
        
        # A. Spread Sensitivity
        if results['spread_2x']['net_profit'] >= base_profit and base_profit > 0:
            report['status'] = "FAIL"
            report['warnings'].append("Strategy profit did NOT decrease with 2x spread (Unrealistic)")
            
        # B. Slippage Stress
        if results['slippage_stress']['net_profit'] >= base_profit and base_profit > 0:
            report['status'] = "FAIL"
            report['warnings'].append("Strategy profit did NOT decrease with high slippage")
            
        # C. Randomized Entry
        if results['random_entry']['net_profit'] > base_profit * 0.5 and base_profit > 0:
            report['status'] = "FAIL"
            report['warnings'].append("Randomized entry test performed too well. System might be flawed.")
            
        if report['structural_checks']['failed']:
            report['status'] = "FAIL"
            report['warnings'].extend(report['structural_checks']['reasons'])
            
        return report

    def _check_structural_weaknesses(self, perf: Dict) -> Dict:
        checks = {"failed": False, "reasons": []}
        
        # 1. Overtrading Detection
        total_trades = perf.get('total_trades', 0)
        if total_trades > 500: # Threshold for a typical 6-12 month backtest
            checks['failed'] = True
            checks['reasons'].append("Overtrading detected: too many trades likely capture noise")
            
        # 2. Risk Concentration
        trades = perf.get('trades', [])
        if trades:
            pnls = [t['pnl'] for t in trades]
            max_loss = min(pnls) if pnls else 0
            if abs(max_loss) > perf['net_profit'] * 0.5 and perf['net_profit'] > 0:
                checks['failed'] = True
                checks['reasons'].append("Risk concentration: a single loss accounts for >50% of profits")
                
        # 3. Equity Curve Quality (Sudden spikes)
        equity = perf.get('equity_curve', [])
        if len(equity) > 10:
            returns = np.diff(equity)
            if np.max(np.abs(returns)) > np.std(returns) * 10:
                checks['failed'] = True
                checks['reasons'].append("Unnatural equity curve: contains extreme spikes")
                
        # 4. Win Rate Check
        if perf.get('win_rate', 0) > 85:
            checks['failed'] = True
            checks['reasons'].append("Extremely high win rate (>85%): likely curve-fitted or lookahead biased")
            
        return checks
