import os
import sys
import MetaTrader5 as mt5
from datetime import datetime, timezone
import numpy as np
from dotenv import load_dotenv

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
load_dotenv()

from core.connection import MT5Connection
from core.common.types import CandleArray
from core.session_detector import SessionDetector

def main():
    conn = MT5Connection()
    if not conn.connect():
        print("MT5 connection failed")
        return

    symbol = "XAUUSDm"
    rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M1, 0, 1500)
    if rates is None or len(rates) < 1440:
        print("Failed to get enough M1 rates")
        conn.disconnect()
        return

    m1 = CandleArray(
        time=rates['time'],
        open=rates['open'],
        high=rates['high'],
        low=rates['low'],
        close=rates['close'],
        tick_volume=rates['tick_volume'],
        spread=rates['spread']
    )
    tick = mt5.symbol_info_tick(symbol)
    
    server_time = conn.get_broker_time(symbol)
    session = SessionDetector.get_session(server_time, conn.server_utc_offset, symbol=symbol)
    
    print(f"Server Time: {server_time}")
    print(f"Session: {session}")
    print(f"Spread: {(tick.ask - tick.bid):.3f}")
    
    # Simulate NPatternGrid logic
    avg_body = np.mean(np.abs(m1.o[-1440:-1] - m1.c[-1440:-1]))
    print(f"24h Average Body Size: {avg_body:.5f}")
    
    # Check last 15 candles for impulse
    threshold_map = {
        "TOKYO": 0.85,
        "LONDON/NY": 0.92,
        "LONDON": 0.90,
        "NEW_YORK": 0.87
    }
    body_ratio_thresh = next((v for k, v in threshold_map.items() if k in session), 0.87)
    print(f"Required Body Ratio for Impulse ({session}): {body_ratio_thresh:.2f}")

    found_impulse = False
    for i in range(len(m1) - 1, len(m1) - 15, -1):
        o, h, l, c = m1.o[i], m1.h[i], m1.l[i], m1.c[i]
        candle_range = h - l
        if candle_range == 0: continue
        body_size = abs(c - o)
        body_ratio = body_size / candle_range
        
        is_large = body_size > avg_body
        is_big_candle = body_ratio > body_ratio_thresh and is_large
        
        if is_big_candle:
            found_impulse = True
            is_bullish = c > o
            direction = "BULL" if is_bullish else "BEAR"
            print(f"\n[!] Impulse Detected at idx {i} (Time: {m1.time[i]}): {direction}")
            print(f"    Body: {body_size:.2f} (> {avg_body:.2f}), Ratio: {body_ratio:.2f} (> {body_ratio_thresh:.2f})")
            
            # Check retrace
            v_high = np.max(m1.h[i:len(m1)])
            v_low = np.min(m1.l[i:len(m1)])
            v_range = v_high - v_low
            
            print(f"    Swing High: {v_high:.2f}, Swing Low: {v_low:.2f}, Range: {v_range:.2f}")
            
            # Historical retrace average logic is complex, let's assume 0.618 for diagnostic
            r_pct = 0.618
            if is_bullish:
                retrace_level = v_high - (v_range * r_pct)
                current_low = m1.l[-1]
                print(f"    Required Bullish Retrace Level (0.618): {retrace_level:.2f}")
                print(f"    Current Low: {current_low:.2f}")
                if current_low <= retrace_level:
                    print("    >>> RETRACE MET! Signal should trigger if spread/session allows.")
                else:
                    print("    >>> Retrace NOT deep enough yet.")
            else:
                retrace_level = v_low + (v_range * r_pct)
                current_high = m1.h[-1]
                print(f"    Required Bearish Retrace Level (0.618): {retrace_level:.2f}")
                print(f"    Current High: {current_high:.2f}")
                if current_high >= retrace_level:
                    print("    >>> RETRACE MET! Signal should trigger if spread/session allows.")
                else:
                    print("    >>> Retrace NOT high enough yet.")
        else:
            if body_ratio > 0.7 or body_size > avg_body:
                print(f"Candle {i} (Time: {m1.time[i]}) - body_size: {body_size:.2f}, ratio: {body_ratio:.2f} -> Rejected")

    if not found_impulse:
        print("\nNo impulse candles detected in the last 15 minutes that meet the strict institutional criteria.")

    conn.disconnect()

if __name__ == "__main__":
    main()
