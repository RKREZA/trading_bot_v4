"""
Comprehensive Strategy Backtest & Optimization Suite
Tests all strategy combinations, walk-forward optimization, and Monte Carlo simulations.
"""
import sys
import os
sys.path.append(os.getcwd())

import numpy as np
import pandas as pd
from datetime import datetime, timezone
import json
import logging

from core.common.types import CandleArray
from core.risk.risk_guardian import RiskGuardian
from core.performance_tracker import PerformanceTracker
from backtesting.backtester import PortfolioBacktester
from backtesting.walk_forward import WalkForwardValidator
from backtesting.monte_carlo import MonteCarloSimulator
from strategies import create_strategy

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("comprehensive_backtest")

class ComprehensiveBacktestSuite:
    """
    Institutional-grade comprehensive testing framework.
    Tests all strategies with various market conditions and optimizations.
    """
    
    def __init__(self):
        self.results = {}
        self.config = self._create_base_config()
        
    def _create_base_config(self):
        """Base configuration for backtests."""
        return {
            "symbol": "XAUUSDm",
            "magic_number": 234000,
            "risk_governance": {
                "risk_per_trade_pct": 1.0,
                "max_daily_loss_pct": 2.0,
                "max_drawdown_halt_pct": 10.0,
                "max_parallel_strategies": 4
            },
            "backtest": {
                "initial_balance": 10000.0,
                "initial_balance_per_strategy": 5000.0,
                "deterministic": True,
                "random_seed": 42,
                "utc_offset": 3
            },
            "portfolio_allocations": {
                "TrendFollowing": 0.4,
                "LiquiditySweepBreakout": 0.4,
                "SmartMeanReversion": 0.2
            },
            "symbols_config": {
                "XAUUSDm": {
                    "point": 0.01,
                    "tick_value": 1.0,
                    "lot_step": 0.01,
                    "min_lot": 0.01,
                    "max_lot": 50.0,
                    "commission_per_lot": 7.0
                }
            }
        }
    
    def generate_synthetic_data(self, n_bars=2000, trend="BULLISH", volatility=3.0, base_price=2000.0):
        """Generate synthetic candle data for backtesting."""
        t = (np.arange(n_bars) * 300 + 1700000000).astype(np.int64)
        
        if trend == "BULLISH":
            close = base_price + np.linspace(0, 150, n_bars) + np.random.normal(0, volatility, n_bars)
        elif trend == "BEARISH":
            close = base_price - np.linspace(0, 150, n_bars) + np.random.normal(0, volatility, n_bars)
        elif trend == "RANGING":
            close = base_price + np.sin(np.linspace(0, 10 * np.pi, n_bars)) * 50 + np.random.normal(0, volatility, n_bars)
        elif trend == "VOLATILE":
            close = base_price + np.cumsum(np.random.normal(0, volatility * 2, n_bars))
        else:
            close = base_price + np.random.normal(0, volatility, n_bars)
        
        spread = np.random.randint(10, 25, n_bars)
        
        return CandleArray(
            time=t,
            open=close - np.random.uniform(0.2, 0.8, n_bars),
            high=close + np.random.uniform(0.5, 1.5, n_bars),
            low=close - np.random.uniform(0.5, 1.5, n_bars),
            close=close,
            tick_volume=np.random.randint(100, 500, n_bars),
            spread=spread
        )
    
    def run_single_strategy_backtest(self, strategy_name, market_condition="BULLISH", volatility=3.0):
        """Run backtest for a single strategy."""
        logger.info(f"Testing {strategy_name} on {market_condition} market (vol={volatility})")
        
        config = self._create_base_config()
        
        # Create strategy
        sid = f"{strategy_name.lower()}_v4"
        st_type = strategy_name.upper()
        
        try:
            strategy = create_strategy(sid, st_type, config)
        except Exception as e:
            logger.error(f"Failed to create {strategy_name}: {e}")
            return None
        
        # Generate data
        m5 = self.generate_synthetic_data(2000, market_condition, volatility)
        m15 = self.generate_synthetic_data(700, market_condition, volatility)
        h1 = self.generate_synthetic_data(350, market_condition, volatility)
        m1 = self.generate_synthetic_data(10000, market_condition, volatility)
        
        # Run backtest
        bt = PortfolioBacktester(config)
        
        try:
            history, equity_history = bt.run("XAUUSDm", [strategy], m5, h1, m15, m5, m1)
            
            if not history:
                logger.warning(f"{strategy_name}: No trades executed")
                return {"status": "NO_TRADES", "strategy": strategy_name}
            
            metrics = PerformanceTracker.calculate_metrics(history, 10000, equity_history)
            return {**metrics, "strategy": strategy_name, "market": market_condition, "status": "SUCCESS"}
            
        except Exception as e:
            logger.error(f"Backtest error for {strategy_name}: {e}")
            return {"status": "ERROR", "strategy": strategy_name, "error": str(e)}
    
    def run_portfolio_backtest(self, strategies, market_condition="BULLISH", volatility=3.0):
        """Run portfolio backtest with multiple strategies."""
        logger.info(f"Testing portfolio ({len(strategies)} strategies) on {market_condition}")
        
        config = self._create_base_config()
        
        # Create strategies
        strategy_objs = []
        for name in strategies:
            sid = f"{name.lower()}_v4"
            st_type = name.upper()
            try:
                strat = create_strategy(sid, st_type, config)
                strategy_objs.append(strat)
            except Exception as e:
                logger.error(f"Failed to create {name}: {e}")
        
        if not strategy_objs:
            return None
        
        # Generate data
        m5 = self.generate_synthetic_data(2000, market_condition, volatility)
        m15 = self.generate_synthetic_data(700, market_condition, volatility)
        h1 = self.generate_synthetic_data(350, market_condition, volatility)
        m1 = self.generate_synthetic_data(10000, market_condition, volatility)
        
        # Run backtest
        bt = PortfolioBacktester(config)
        
        try:
            history, equity_history = bt.run("XAUUSDm", strategy_objs, m5, h1, m15, m5, m1)
            
            if not history:
                return {"status": "NO_TRADES", "strategies": strategies}
            
            metrics = PerformanceTracker.calculate_metrics(history, 10000, equity_history)
            
            # Run Monte Carlo
            mc = MonteCarloSimulator(iterations=1000)
            mc_results = mc.run(history, 10000)
            
            return {
                **metrics,
                "strategies": strategies,
                "market": market_condition,
                "status": "SUCCESS",
                "monte_carlo": mc_results
            }
            
        except Exception as e:
            logger.error(f"Portfolio backtest error: {e}")
            return {"status": "ERROR", "error": str(e)}
    
    def run_walk_forward_optimization(self, strategy_name):
        """Run walk-forward optimization for a strategy."""
        logger.info(f"Walk-Forward Optimization for {strategy_name}")
        
        config = self._create_base_config()
        
        sid = f"{strategy_name.lower()}_v4"
        st_type = strategy_name.upper()
        
        try:
            strategy = create_strategy(sid, st_type, config)
        except Exception as e:
            logger.error(f"Failed to create {strategy_name}: {e}")
            return None
        
        # Generate multi-market data
        data = {
            "M5": self.generate_synthetic_data(5000, "BULLISH", 3.0),
            "M15": self.generate_synthetic_data(1700, "BULLISH", 3.0),
            "H1": self.generate_synthetic_data(850, "BULLISH", 3.0),
            "M1": self.generate_synthetic_data(25000, "BULLISH", 3.0)
        }
        
        wfo = WalkForwardValidator(config)
        
        try:
            results = wfo.run_validation("XAUUSDm", [strategy], data)
            return results
        except Exception as e:
            logger.error(f"WFO error for {strategy_name}: {e}")
            return None
    
    def run_institutional_grade_check(self, metrics, mc_results):
        """Verify metrics meet institutional grade standards."""
        checks = {
            "sharpe_above_1": metrics.get("sharpe_ratio", 0) >= 1.0,
            "sortino_above_1.5": metrics.get("sortino_ratio", 0) >= 1.5,
            "max_dd_below_15": float(metrics.get("max_drawdown", "0%").rstrip("%")) < 15,
            "win_rate_above_40": float(metrics.get("win_rate", "0%").rstrip("%")) >= 40,
            "profit_factor_above_1.2": metrics.get("profit_factor", 0) >= 1.2,
            "robustness_score_above_70": mc_results.get("robustness_score", 0) >= 70 if mc_results else False,
            "ruin_probability_zero": float(mc_results.get("probability_of_ruin", "100%").rstrip("%")) == 0 if mc_results else False,
        }
        
        passed = sum(checks.values())
        total = len(checks)
        
        return {
            "checks": checks,
            "passed": passed,
            "total": total,
            "grade": "A" if passed >= 7 else "B" if passed >= 5 else "C" if passed >= 3 else "F",
            "institutional_grade": passed >= 7
        }


def run_all_tests():
    """Run comprehensive backtest suite."""
    print("=" * 80)
    print(" INSTITUTIONAL TRADING BOT - COMPREHENSIVE BACKTEST SUITE ")
    print("=" * 80)
    
    suite = ComprehensiveBacktestSuite()
    
    results = {
        "timestamp": datetime.now().isoformat(),
        "single_strategies": {},
        "portfolios": {},
        "walk_forward": {},
        "institutional_checks": {}
    }
    
    # Test individual strategies
    strategies = ["TrendFollowing", "LiquiditySweepBreakout", "SmartMeanReversion"]
    market_conditions = ["BULLISH", "BEARISH", "RANGING", "VOLATILE"]
    
    print("\n[1] SINGLE STRATEGY BACKTESTS")
    print("-" * 60)
    
    for strategy in strategies:
        results["single_strategies"][strategy] = {}
        for market in market_conditions:
            vol = 5.0 if market == "VOLATILE" else 3.0
            result = suite.run_single_strategy_backtest(strategy, market, vol)
            results["single_strategies"][strategy][market] = result
            
            if result and result.get("status") == "SUCCESS":
                print(f"  {strategy:25} | {market:10} | Sharpe: {result.get('sharpe_ratio', 0):5.2f} | Win%: {result.get('win_rate', '0%'):>6} | DD: {result.get('max_drawdown', '0%'):>6}")
            else:
                print(f"  {strategy:25} | {market:10} | {'NO TRADES' if result.get('status') == 'NO_TRADES' else 'ERROR'}")
    
    # Test portfolio combinations
    print("\n[2] PORTFOLIO COMBINATIONS")
    print("-" * 60)
    
    portfolios = [
        ["TrendFollowing", "LiquiditySweepBreakout", "SmartMeanReversion"],
        ["TrendFollowing", "LiquiditySweepBreakout"],
        ["TrendFollowing", "SmartMeanReversion"],
        ["LiquiditySweepBreakout", "SmartMeanReversion"],
        ["TrendFollowing"],
    ]
    
    for portfolio in portfolios:
        portfolio_name = " + ".join([p.replace("SmartMeanReversion", "MeanRev").replace("LiquiditySweepBreakout", "Breakout").replace("TrendFollowing", "Trend") for p in portfolio])
        
        for market in ["BULLISH", "RANGING"]:
            result = suite.run_portfolio_backtest(portfolio, market)
            results["portfolios"][portfolio_name] = result
            
            if result and result.get("status") == "SUCCESS":
                mc = result.get("monte_carlo", {})
                print(f"  {portfolio_name:35} | {market:10} | Sharpe: {result.get('sharpe_ratio', 0):5.2f} | PF: {result.get('profit_factor', 0):5.2f} | Robustness: {mc.get('robustness_score', 0):5.1f}")
            else:
                print(f"  {portfolio_name:35} | {market:10} | {'NO TRADES' if result.get('status') == 'NO_TRADES' else 'ERROR'}")
    
    # Walk-forward optimization
    print("\n[3] WALK-FORWARD OPTIMIZATION")
    print("-" * 60)
    
    for strategy in ["TrendFollowing", "LiquiditySweepBreakout"]:
        wfo_result = suite.run_walk_forward_optimization(strategy)
        results["walk_forward"][strategy] = wfo_result
        
        if wfo_result:
            avg_ratio = np.mean([r.get("wfo_ratio", 0) for r in wfo_result if r])
            print(f"  {strategy:25} | Avg WFO Ratio: {avg_ratio:.3f} | Windows: {len(wfo_result)}")
        else:
            print(f"  {strategy:25} | WFO FAILED")
    
    # Institutional grade checks
    print("\n[4] INSTITUTIONAL GRADE VERIFICATION")
    print("-" * 60)
    
    # Check best performing strategy/portfolio
    best_key = None
    best_sharpe = -999
    
    for key, result in {**results["single_strategies"], **results["portfolios"]}.items():
        if isinstance(result, dict) and result.get("sharpe_ratio", -999) > best_sharpe:
            best_sharpe = result.get("sharpe_ratio", -999)
            best_key = key
    
    if best_key and results["portfolios"].get(best_key):
        best_result = results["portfolios"][best_key]
        mc_results = best_result.get("monte_carlo", {})
        inst_check = suite.run_institutional_grade_check(best_result, mc_results)
        results["institutional_checks"]["best_portfolio"] = inst_check
        
        print(f"  Best Portfolio: {best_key}")
        print(f"  Grade: {inst_check['grade']} ({inst_check['passed']}/{inst_check['total']} checks passed)")
        for check, passed in inst_check["checks"].items():
            status = "[PASS]" if passed else "[FAIL]"
            print(f"    {status} {check}")
    
    # Save results
    output_file = "backtest_results/comprehensive_results.json"
    os.makedirs("backtest_results", exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n[5] Results saved to: {output_file}")
    
    print("\n" + "=" * 80)
    print(" BACKTEST SUITE COMPLETED ")
    print("=" * 80)
    
    return results


if __name__ == "__main__":
    run_all_tests()
