import pandas as pd
import numpy as np
from core.config.loader import ConfigLoader
from core.common.types import CandleArray
from backtesting.backtester import PortfolioBacktester
from strategies import create_strategy
from core.base_strategy import MarketData
from datetime import datetime

def debug_rejections():
    symbol = "XAUUSDm"
    loader = ConfigLoader()
    config = loader.get_symbol_config(symbol)
    
    # Load data
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

    m1 = load_tf("M1", 10000)
    m5 = load_tf("M5", 3000)
    m15 = load_tf("M15", 1000)
    h1 = load_tf("H1", 300)
    
    sid = "trendfollowing_v4"
    st = create_strategy(sid, "TRENDFOLLOWING", config)
    
    print(f"Strategy: {sid}")
    print(f"Min Tick Density: {config.get('risk_governance', {}).get('min_tick_density', 45)}")
    
    # Check last 10 bars
    for i in range(-10, 0):
        md = MarketData(
            symbol=symbol,
            htf_candles=h1,
            m15_candles=m15[len(m15)+i-100:len(m15)+i], # Slice for indicators
            m5_candles=m5[len(m5)+i-100:len(m5)+i],
            d1_candles=None,
            current_price=m5.close[i],
            bid=m5.close[i],
            ask=m5.close[i] + 0.1,
            spread=0.1,
            point=0.01,
            session="LONDON",
            timestamp=datetime.fromtimestamp(m5.time[i])
        )
        
        passed = st.is_spread_safe(md)
        vol = m15.v[i]
        dens = vol / 15.0
        print(f"Bar {i} | Time: {md.timestamp} | Vol: {vol} | Density: {dens:.1f} | Safe: {passed} | Reason: {st.last_rejection_reason if not passed else ''}")

if __name__ == "__main__":
    debug_rejections()
