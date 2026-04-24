import sys
import os
import logging
from comprehensive_backtest import ComprehensiveBacktestSuite
from core.config.loader import ConfigLoader
from strategies import create_strategy
from backtesting.backtester import PortfolioBacktester

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("analyze_mss_rate")

def main():
    symbol = "XAUUSDm"
    config_loader = ConfigLoader()
    base_config = config_loader.global_config
    symbol_config = config_loader.get_symbol_config(symbol)
    config = {**base_config, **symbol_config}
    
    if "risk_governance" not in config: config["risk_governance"] = {}
    config["risk_governance"]["min_tick_density"] = 1
    config["max_spread_points"] = 1500
    config["backtest"] = config.get("backtest", {})
    # Enable debugging to see all rejection reasons if needed, but the backtester collects them anyway.
    # config["backtest"]["debug_signals"] = True
    
    suite = ComprehensiveBacktestSuite()
    
    # 25,000 M5 bars = approx 3 months of data
    m5 = suite.load_real_data(symbol=symbol, timeframe="M5", n_bars=25000)
    m15 = suite.load_real_data(symbol=symbol, timeframe="M15", n_bars=8500)
    h1 = suite.load_real_data(symbol=symbol, timeframe="H1", n_bars=2200)
    m1 = suite.load_real_data(symbol=symbol, timeframe="M1", n_bars=150000)
    
    m5, m15, h1, m1 = suite._align_timeframes(m1, m5, m15, h1)
    
    strategy_id = "liquiditysweepbreakout_v7"
    strat = create_strategy(strategy_id, "LIQUIDITYSWEEPBREAKOUT", config)
    
    bt = PortfolioBacktester(config)
    print("Running diagnostic backtest to collect MSS funnel stats...")
    bt.run(symbol, [strat], m5, h1, m15, m5, m1)
    
    # Look at rejection stats
    stats = bt.rejection_stats.get(strat.strategy_id, {})
    
    mss_timeout = stats.get("MSS Timeout", 0)
    waiting_mss = stats.get("Waiting for MSS", 0)
    # The number of trades executed can be inferred from the history
    trades_executed = len(bt.history)
    
    print("\n--- MSS CONFIRMATION RATE ANALYSIS ---")
    print(f"Total Trades Executed: {trades_executed}")
    print(f"MSS Timeouts: {mss_timeout}")
    
    # Sweeps that passed rejection validation are those that either ended in a trade or an MSS Timeout
    total_valid_sweeps = trades_executed + mss_timeout
    
    if total_valid_sweeps > 0:
        confirmation_rate = (trades_executed / total_valid_sweeps) * 100
        timeout_rate = (mss_timeout / total_valid_sweeps) * 100
        print(f"Total Sweeps Reaching MSS Stage: {total_valid_sweeps}")
        print(f"MSS Confirmation Rate: {confirmation_rate:.2f}%")
        print(f"MSS Timeout Rate: {timeout_rate:.2f}%")
    else:
        print("No valid sweeps reached the MSS stage.")

if __name__ == "__main__":
    main()
