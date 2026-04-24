import sys
import os
import logging
from comprehensive_backtest import ComprehensiveBacktestSuite
from core.config.loader import ConfigLoader
from strategies import create_strategy
from backtesting.backtester import PortfolioBacktester
from core.performance_tracker import PerformanceTracker

logging.basicConfig(level=logging.WARNING)

def test_instrument_timeframe(symbol: str, timeframe: str):
    print(f"\n--- Testing Pivot: {symbol} on {timeframe} ---")
    
    relaxed_params = {
        "volume_zscore_threshold": 0.8,
        "rejection_percentile": 0.25,
        "sweep_depth_percentile": 0.40,
        "max_atr_ratio": 3.0,
        "vwap_filter_enabled": False,
        "ema_trend_filter": False,
        "max_trades_per_day": 5,
        "mss_timeout_bars": 10,
        "lookback": 20
    }
    
    config_loader = ConfigLoader()
    base_config = config_loader.global_config
    
    test_config = {**base_config}
    test_config["symbols_config"] = {symbol: relaxed_params}
    
    # Relax governance to allow anything
    if "risk_governance" not in test_config:
        test_config["risk_governance"] = {}
    test_config["risk_governance"]["min_tick_density"] = 1
    test_config["max_spread_points"] = 1500
    test_config["backtest"] = test_config.get("backtest", {})
    test_config["backtest"]["disable_checkpoint"] = True

    suite = ComprehensiveBacktestSuite()
    
    # Load 3 months of data
    try:
        # We need M1 data to do realistic backtesting, or at least base timeframe.
        print("Loading data...")
        if timeframe == "M5":
            m5 = suite.load_real_data(symbol=symbol, timeframe="M5", n_bars=25000)
            m15 = suite.load_real_data(symbol=symbol, timeframe="M15", n_bars=8500)
            h1 = suite.load_real_data(symbol=symbol, timeframe="H1", n_bars=2200)
            m1 = suite.load_real_data(symbol=symbol, timeframe="M1", n_bars=150000)
            m5, m15, h1, m1 = suite._align_timeframes(m1, m5, m15, h1)
            bt_data = {"m5": m5, "h1": h1, "m15": m15, "entry_tf": m5, "m1": m1}
        else: # M15
            # If entry TF is M15, the backtester expects m5, h1, m15, entry, m1
            # In LiquiditySweepBreakout, it hardcodes using m5 array from PortfolioBacktester. 
            # So if we want to run it on M15, we might need to spoof the m5 array with m15 data!
            # Let's just load M15 and pass it as M5.
            m15 = suite.load_real_data(symbol=symbol, timeframe="M15", n_bars=25000)
            h1 = suite.load_real_data(symbol=symbol, timeframe="H1", n_bars=8500)
            m1 = suite.load_real_data(symbol=symbol, timeframe="M1", n_bars=150000)
            # Alignment won't perfectly match M5 since we spoof it, but for a simple trade count check:
            m15_spoof, m15_real, h1_aligned, m1_aligned = suite._align_timeframes(m1, m15, m15, h1)
            bt_data = {"m5": m15_spoof, "h1": h1_aligned, "m15": m15_real, "entry_tf": m15_spoof, "m1": m1_aligned}
            
        strategy_id = "liquiditysweepbreakout_v7"
        strat = create_strategy(strategy_id, "LIQUIDITYSWEEPBREAKOUT", test_config)
        
        bt = PortfolioBacktester(test_config)
        history, _ = bt.run(symbol, [strat], bt_data["m5"], bt_data["h1"], bt_data["m15"], bt_data["entry_tf"], bt_data["m1"])
        
        trades = len(history) if history else 0
        print(f"-> {symbol} {timeframe} yielded {trades} trades.")
        
    except Exception as e:
        print(f"Error testing {symbol} {timeframe}: {e}")

if __name__ == "__main__":
    test_instrument_timeframe("EURUSDm", "M5")
    test_instrument_timeframe("BTCUSDm", "M5")
