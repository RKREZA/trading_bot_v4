import random
import copy
import logging
from typing import Dict, List
from core.backtester import BacktestEngine
from core.strategy_engine import StrategyEngine

logger = logging.getLogger("trading_bot.validation")

class ValidationSuite:
    """
    Rigorously stress-tests a strategy to ensure it isn't overfitted or fragile.
    Executes a series of adversarial simulations:
    - High Spread/Slippage: Tests durability under poor execution conditions.
    - Random Entry: Vets if the alpha is better than pure luck.
    - Window Stability: Checks consistency across different time periods.
    - Monte Carlo: Shuffles trade order to find worst-case drawdown probabilities.
    """
    def __init__(self, config: dict, strategy: StrategyEngine):
        """
        Initializes the validation suite.
        
        Args:
            config (dict): Bot configuration.
            strategy (StrategyEngine): The strategy instance to vet.
        """
        self.config = config
        self.strategy = strategy

    def run_all_tests(self, symbol: str, h4: List, h1: List, m30: List, m5: List, d1: List) -> Dict:
        """
        Runs the full battery of stress tests and returns an aggregated report.
        
        Args:
            symbol (str): Symbol name.
            h4, h1, m30, m5, d1 (List): Multi-timeframe historical candle data.
            
        Returns:
            Dict: Aggregated validation report with status (PASS/FAIL) and metrics.
        """
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
        results['rolling_stability'] = self._rolling_window_stability_test(symbol, h4, h1, m30, m5, d1)
        
        # 5. Monte Carlo Simulation
        if results['base'].get('trades'):
            results['monte_carlo'] = self.monte_carlo_equity(results['base']['trades'])
        
        return self._generate_report(results)

    def _spread_sensitivity_test(self, symbol, h4, h1, m30, m5, d1):
        stress_config = copy.deepcopy(self.config)
        
        if "backtest" not in stress_config:
            stress_config["backtest"] = {}
        if "spread_pips" not in stress_config["backtest"]:
            stress_config["backtest"]["spread_pips"] = {}

        current_spread = stress_config["backtest"]["spread_pips"].get(symbol, 25)
        stress_config["backtest"]["spread_pips"][symbol] = current_spread * 2.0
        
        tester = BacktestEngine(stress_config, self.strategy)
        return tester.run(symbol, h4, h1, m30, m5, d1, quiet=True)

    def _slippage_stress_test(self, symbol, h4, h1, m30, m5, d1):
        stress_config = copy.deepcopy(self.config)
        
        if "backtest" not in stress_config:
            stress_config["backtest"] = {}
        if "slippage_points" not in stress_config["backtest"]:
            stress_config["backtest"]["slippage_points"] = {}
            
        current_slip = stress_config["backtest"]["slippage_points"].get(symbol, 2.0)
        stress_config["backtest"]["slippage_points"][symbol] = current_slip * 5.0
        
        tester = BacktestEngine(stress_config, self.strategy)
        return tester.run(symbol, h4, h1, m30, m5, d1, quiet=True)

    def _random_entry_test(self, symbol, h4, h1, m30, m5, d1):
        """
        Replaces the strategy logic with a random number generator.
        If random entries perform nearly as well as the strategy, the strategy lacks alpha.
        """
        class RandomStrategy(StrategyEngine):
            def analyze(self, symbol, h4, h1, m30, m5, current_price, d1_candles=None, session=None):
                if random.random() < 0.05: # 5% chance
                    from core.strategy_engine import TradeSignal
                    direction = "BUY" if random.random() > 0.5 else "SELL"
                    # Minimal dummy signal
                    sl = current_price - 10.0 if direction == "BUY" else current_price + 10.0
                    tp = current_price + 20.0 if direction == "BUY" else current_price - 20.0
                    return TradeSignal(direction, current_price, sl, tp, confidence=50), "RANGING", "NEUTRAL"
                return None, "RANGING", "NEUTRAL"
        
        tester = BacktestEngine(self.config, RandomStrategy(self.config))
        return tester.run(symbol, h4, h1, m30, m5, d1, quiet=True)

    def _rolling_window_stability_test(self, symbol, h4, h1, m30, m5, d1):
        """Split data into 3 time windows and test each independently for consistency."""
        n = len(m5)
        third = n // 3
        # Each window gets 200 extra candles of warmup from the previous period
        windows = [
            m5[:third + 200],
            m5[max(0, third - 200):2 * third + 200],
            m5[max(0, 2 * third - 200):]
        ]
        
        window_results = []
        for idx, w in enumerate(windows):
            if len(w) < 400:
                continue
            tester = BacktestEngine(self.config, self.strategy)
            perf = tester.run(symbol, h4, h1, m30, w, d1, quiet=True)
            window_results.append(perf)
        
        if not window_results:
            return {"window_results": [], "consistency_score": 0, "all_profitable": False}
        
        sharpes = [r.get("sharpe_ratio", 0) for r in window_results]
        pfs = [r.get("profit_factor", 0) for r in window_results]
        max_sharpe = max(abs(s) for s in sharpes) if sharpes else 1
        consistency = min(sharpes) / max_sharpe if max_sharpe > 0 else 0
        
        return {
            "window_results": window_results,
            "consistency_score": round(float(consistency), 4),
            "all_profitable": all(r.get("net_profit", 0) > 0 for r in window_results),
            "sharpes": [round(s, 2) for s in sharpes],
            "profit_factors": [round(p, 2) for p in pfs]
        }

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

    def monte_carlo_equity(self, trades: List[Dict], iterations: int = 1000) -> Dict:
        """
        Shuffles the sequence of historical trades 1000 times.
        Calculates the probability of insolvency or extreme drawdown if the 
        distribution of returns remains the same but the order changes.
        
        Args:
            trades (List[Dict]): List of trade objects with 'pnl'.
            iterations (int): Number of shuffle iterations.
            
        Returns:
            Dict: Statistics including P95 (95% confidence) Max Drawdown.
        """
        import numpy as np
        import pandas as pd
        
        pnl_list = [t['pnl'] for t in trades]
        initial_balance = self.config.get("backtest", {}).get("initial_balance", 1000)
        max_drawdowns = []

        for _ in range(iterations):
            shuffled = pnl_list[:]
            random.shuffle(shuffled)
            
            balance = initial_balance
            equity_curve = [balance]
            for pnl in shuffled:
                balance += pnl
                equity_curve.append(balance)
            
            equity_series = pd.Series(equity_curve)
            rolling_max = equity_series.cummax()
            drawdown = (rolling_max - equity_series) / rolling_max * 100
            max_drawdowns.append(drawdown.max())
            
        max_drawdowns.sort()
        # 95th percentile
        p95_dd = max_drawdowns[int(iterations * 0.95)]
        
        if p95_dd > 30:
            logger.warning("Strategy too fragile for live trading! 95%% MC Max Drawdown: %.1f%%", p95_dd)
            
        return {
            "p95_max_drawdown": float(p95_dd),
            "mean_max_drawdown": float(np.mean(max_drawdowns)),
            "worst_case_drawdown": float(max_drawdowns[-1])
        }
