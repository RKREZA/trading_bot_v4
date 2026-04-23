import sys
import os
import logging
sys.path.append(os.getcwd())

from backtesting.backtester import PortfolioBacktester
from strategies.trend_following import TrendFollowingStrategy
from core.common.types import CandleArray
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from strategies import create_strategy

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("diagnostic_backtest")

# Inject regime logging into backtester for this diagnostic
from backtesting.backtester import PortfolioBacktester
import backtesting.backtester as bt_mod

original_run = PortfolioBacktester.run

def diagnostic_run(self, symbol, strategies, m1, m5, m15, h1, d1):
    print(f"DEBUG: Starting run for {symbol}")
    return original_run(self, symbol, strategies, m1, m5, m15, h1, d1)

# PortfolioBacktester.run = diagnostic_run # We could monkeypatch if needed

def run_diagnostic():
    # 1. Load sample data (last 1000 bars from parquet)
    symbol = "XAUUSDm"
    path = f"data_cache/{symbol}/M5.parquet"
    df = pd.read_parquet(path).tail(2000) # Give enough room for indicators
    
    m5 = CandleArray(
        time=df['time'].values,
        open=df['open'].values,
        high=df['high'].values,
        low=df['low'].values,
        close=df['close'].values,
        tick_volume=df['tick_volume'].values,
        spread=df['spread'].values
    )
    
    # 2. Setup Config with DEBUG_SIGNALS
    config = {
        "symbol": symbol,
        "magic_number": 123,
        "backtest": {
            "initial_balance": 10000.0,
            "deterministic": True,
            "utc_offset": 3,
            "disable_checkpoint": True,
            "debug_signals": True
        },
        "symbols_config": {
            symbol: {
                "point": 0.01,
                "tick_value": 1.0,
                "lot_step": 0.01,
                "min_lot": 0.01,
                "max_lot": 50.0,
                "commission_per_lot": 7.0
            }
        },
        "risk_governance": {
            "risk_per_trade_pct": 1.0,
            "min_confidence": 0.5
        }
    }
    
    # 3. Initialize Backtester
    bt = PortfolioBacktester(config)
    
    # Select the strategy to diagnose
    strat = create_strategy("TrendFollowing", None, config)
    # strat = create_strategy("LiquiditySweepBreakout", None, config)
    # strat = create_strategy("SmartMeanReversion", None, config)
    
    # 5. Run it
    print("STARTING DIAGNOSTIC BACKTEST...")
    # We pass placeholders for htf_data etc as TF strategy only uses M15/H1 for context
    # Usually it derives those or handles None
    # For this diagnostic, we'll give it the same M5 for everything
    bt.run(symbol, [strat], m5, m5, m5, m5, m5)

if __name__ == "__main__":
    run_diagnostic()
