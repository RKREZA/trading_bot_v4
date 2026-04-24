import logging
from datetime import datetime
import json
import os

from core.config.loader import ConfigLoader
from backtesting.walk_forward import WalkForwardValidator
from strategies import create_strategy

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("run_wfo")

def main():
    symbol = "XAUUSDm"
    
    config_loader = ConfigLoader()
    base_config = config_loader.global_config
    
    # We load 8 months of data: 4 months initial training + 4 months test (to allow 1 step)
    # Actually, WFO handles dates dynamically, so let's load a full year
    # to allow multiple walk-forward steps. 
    # 1 year = 250 trading days = approx 75000 M5 bars.
    print(f"Loading data for {symbol}...")
    from comprehensive_backtest import ComprehensiveBacktestSuite
    suite = ComprehensiveBacktestSuite()
    
    data = {
        "M5": suite.load_real_data(symbol=symbol, timeframe="M5", n_bars=75000),
        "M15": suite.load_real_data(symbol=symbol, timeframe="M15", n_bars=25000),
        "H1": suite.load_real_data(symbol=symbol, timeframe="H1", n_bars=6500),
        "M1": suite.load_real_data(symbol=symbol, timeframe="M1", n_bars=350000)
    }
    
    symbol_config = config_loader.get_symbol_config(symbol)
    config = {**base_config, **symbol_config}
    
    # Relax liquidity gates for historic data
    if "risk_governance" not in config: config["risk_governance"] = {}
    config["risk_governance"]["min_tick_density"] = 1
    config["max_spread_points"] = 1500
    
    strategy_id = "liquiditysweepbreakout_v7"
    strat = create_strategy(strategy_id, "LIQUIDITYSWEEPBREAKOUT", config)
    
    print("\nStarting WFO (4mo In-Sample, 2mo Out-of-Sample, 1mo step)")
    wfo = WalkForwardValidator(config)
    results = wfo.run_validation(
        symbol=symbol,
        strategies=[strat],
        data=data,
        window_weeks=16,   # 4 months
        test_weeks=8,      # 2 months
        step_weeks=4,      # 1 month step
        min_wfo_ratio=0.50,
        run_mc=False       # Speed up
    )
    
    wfo.summarize_wfo_results()
    
    os.makedirs("backtest_results", exist_ok=True)
    with open("backtest_results/wfo_liquidity_sweep.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    print("\nDone. Results saved to backtest_results/wfo_liquidity_sweep.json")

if __name__ == "__main__":
    main()
