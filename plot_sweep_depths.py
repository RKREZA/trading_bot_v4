import sys
import os
import matplotlib.pyplot as plt
import numpy as np

from comprehensive_backtest import ComprehensiveBacktestSuite
from core.config.loader import ConfigLoader
from strategies import create_strategy
from backtesting.backtester import PortfolioBacktester

def main():
    symbol = "XAUUSDm"
    config_loader = ConfigLoader()
    base_config = config_loader.global_config
    symbol_config = config_loader.get_symbol_config(symbol)
    config = {**base_config, **symbol_config}
    
    if "risk_governance" not in config: config["risk_governance"] = {}
    config["risk_governance"]["min_tick_density"] = 1
    config["max_spread_points"] = 1500
    
    suite = ComprehensiveBacktestSuite()
    
    m5 = suite.load_real_data(symbol=symbol, timeframe="M5", n_bars=25000)
    m15 = suite.load_real_data(symbol=symbol, timeframe="M15", n_bars=8500)
    h1 = suite.load_real_data(symbol=symbol, timeframe="H1", n_bars=2200)
    m1 = suite.load_real_data(symbol=symbol, timeframe="M1", n_bars=150000)
    
    m5, m15, h1, m1 = suite._align_timeframes(m1, m5, m15, h1)
    
    strategy_id = "liquiditysweepbreakout_v7"
    strat = create_strategy(strategy_id, "LIQUIDITYSWEEPBREAKOUT", config)
    
    bt = PortfolioBacktester(config)
    print("Running diagnostic backtest to collect sweep depths...")
    bt.run(symbol, [strat], m5, h1, m15, m5, m1)
    
    depths = list(strat._recent_sweep_depths)
    if not depths:
        print("No sweep depths collected.")
        return
        
    print(f"Collected {len(depths)} sweep depths. Plotting...")
    
    plt.figure(figsize=(10, 6))
    plt.hist(depths, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
    
    median_val = np.median(depths)
    percentile_60 = np.percentile(depths, 60)
    
    plt.axvline(median_val, color='red', linestyle='dashed', linewidth=1.5, label=f'Median: {median_val:.2f}')
    plt.axvline(percentile_60, color='purple', linestyle='dashed', linewidth=1.5, label=f'60th Percentile: {percentile_60:.2f}')
    
    plt.title('Distribution of Sweep Depths (ATR Multiple)')
    plt.xlabel('Sweep Depth (x ATR)')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(axis='y', alpha=0.75)
    
    artifacts_dir = os.path.expanduser('~/.gemini/antigravity/brain/911b5ee1-8e5b-4b0d-8d00-351c948fb81f/artifacts')
    os.makedirs(artifacts_dir, exist_ok=True)
    out_path = os.path.join(artifacts_dir, 'sweep_depths.png')
    
    plt.savefig(out_path)
    print(f"Plot saved to {out_path}")

if __name__ == "__main__":
    main()
