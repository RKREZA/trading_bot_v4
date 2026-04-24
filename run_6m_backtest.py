import sys
import os
import json

from comprehensive_backtest import ComprehensiveBacktestSuite
from strategies import create_strategy
from backtesting.backtester import PortfolioBacktester
from core.performance_tracker import PerformanceTracker

class SixMonthSuite(ComprehensiveBacktestSuite):
    def run_6m(self):
        strategy_name = "LiquiditySweepBreakout"
        symbol = "XAUUSDm"
        
        base_config = self._create_base_config()
        symbol_config = self.config_loader.get_symbol_config(symbol)
        config = {**base_config, **symbol_config}
        
        sid = f"{strategy_name.lower()}_v7"
        st_type = strategy_name.upper()
        
        strategy = create_strategy(sid, st_type, config)
        
        # 6 months = 180 days approx
        # M5 bars = 180 * 24 * 12 = 51840
        m5_raw = self.load_real_data(symbol=symbol, timeframe="M5", n_bars=51840)
        m15_raw = self.load_real_data(symbol=symbol, timeframe="M15", n_bars=17280)
        h1_raw = self.load_real_data(symbol=symbol, timeframe="H1", n_bars=4320)
        m1_raw = self.load_real_data(symbol=symbol, timeframe="M1", n_bars=259200)
        
        if "risk_governance" not in config: config["risk_governance"] = {}
        config["risk_governance"]["min_tick_density"] = 1
        config["max_spread_points"] = 1500
        
        m5, m15, h1, m1 = self._align_timeframes(m1_raw, m5_raw, m15_raw, h1_raw)
        
        bt = PortfolioBacktester(config)
        print("Starting 6-month backtest for LiquiditySweepBreakout (V7)...")
        history, equity_history = bt.run(symbol, [strategy], m5, h1, m15, m5, m1)
        
        if not history:
            print(json.dumps({"status": "NO_TRADES"}, indent=2))
            return
            
        metrics = PerformanceTracker.calculate_metrics(history, 10000, equity_history)
        
        result = {
            "strategy": strategy_name,
            "period": "Last 6 Months",
            **metrics
        }
        
        print("\n=== 6 MONTH BACKTEST RESULTS ===")
        print(json.dumps(result, indent=2))
        
        print("\n--- Summary ---")
        print(f"Total Trades:  {metrics.get('total_trades', 0)}")
        print(f"Win Rate:      {metrics.get('win_rate', '0%')}")
        print(f"Net Profit:    ${metrics.get('net_profit', 0):.2f}")
        print(f"Profit Factor: {metrics.get('profit_factor', 0):.2f}")
        print(f"Max Drawdown:  {metrics.get('max_drawdown', '0%')}")
        print(f"Sharpe Ratio:  {metrics.get('sharpe_ratio', 0):.2f}")
        print("================================")

if __name__ == "__main__":
    suite = SixMonthSuite()
    suite.run_6m()
