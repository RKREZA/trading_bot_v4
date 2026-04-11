import pandas as pd
import numpy as np
from core.config.loader import ConfigLoader
from core.common.types import CandleArray
from backtesting.backtester import PortfolioBacktester
from strategies import create_strategy
import logging

# Setup minimal logging
logging.basicConfig(level=logging.INFO)

def diagnose_backtest():
    symbol = "XAUUSDm"
    loader = ConfigLoader()
    config = loader.get_symbol_config(symbol)
    
    # Relax gates
    if "risk_governance" not in config: config["risk_governance"] = {}
    config["risk_governance"]["min_tick_density"] = 1
    config["max_spread_points"] = 1500
    
    def load_tf(tf, n):
        df = pd.read_parquet(f"data_cache/{symbol}/{tf}.parquet").tail(n)
        return CandleArray(
            time=df['time'].values,
            open=df['open'].values,
            high=df['high'].values,
            low=df['low'].values,
            close=df['close'].values,
            tick_volume=df['tick_volume'].values,
            spread=df['spread'].values
        )

    # Padding-Aware Loading
    m1 = load_tf("M1", 500000)
    m5 = load_tf("M5", 35000)
    m15 = load_tf("M15", 12000)
    h1 = load_tf("H1", 3000)
    
    # NEW ALIGNMENT: Pad History
    max_t = m1.time[-1]
    min_t_exec = m1.time[0]
    
    # Only clip the end
    m5_c = m5[m5.time <= max_t]
    m15_c = m15[m15.time <= max_t]
    h1_c = h1[h1.time <= max_t]
    
    # Find the M5 index that corresponds to the start of M1
    start_idx = np.searchsorted(m5_c.time, min_t_exec)
    
    print(f"Total M5 bars: {len(m5_c)}")
    print(f"Execution starts at M5 index: {start_idx}")
    
    sid = "trendfollowing_v4"
    st = create_strategy(sid, "TRENDFOLLOWING", config)
    
    rejections = {}
    signals_found = 0
    
    # Run loop from start of M1 for 1000 M5 bars
    for i in range(start_idx, min(start_idx + 1000, len(m5_c))):
        from core.base_strategy import MarketData
        from datetime import datetime
        
        # Strategies see the history!
        md = MarketData(
            symbol=symbol,
            htf_candles=h1_c[:np.searchsorted(h1_c.time, m5_c.time[i], side='right')],
            m15_candles=m15_c[:np.searchsorted(m15_c.time, m5_c.time[i], side='right')],
            m5_candles=m5_c[:i+1],
            d1_candles=None,
            current_price=m5_c.close[i],
            bid=m5_c.close[i],
            ask=m5_c.close[i] + 0.1,
            spread=0.1,
            point=0.01,
            session="LONDON",
            timestamp=datetime.fromtimestamp(m5_c.time[i])
        )
        
        signal = st.generate_signal(md)
        if signal:
            signals_found += 1
        else:
            reason = getattr(st, 'last_rejection_reason', 'Unknown')
            rejections[reason] = rejections.get(reason, 0) + 1

    print(f"\nResults over {1000} bars:")
    print(f"  SIGNALS FOUND: {signals_found}")
    print("\nRejection Stats:")
    for r, count in rejections.items():
        print(f"  {r}: {count}")

if __name__ == "__main__":
    diagnose_backtest()
