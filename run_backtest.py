import os
import sys
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("backtest")

def generate_synthetic_data(symbol="BTCUSDm", bars=10000):
    np.random.seed(42)
    logger.info(f"Generating {bars} bars for {symbol}...")
    
    timestamps = []
    opens, highs, lows, closes = [], [], [], []
    volumes, spreads = [], []
    
    price = 45000.0
    volatility = 800.0
    
    for i in range(bars):
        ts = 1704067200 + i * 300
        
        if i % 2000 == 0:
            volatility = np.random.uniform(500, 1500)
        
        trend = np.random.choice([-1, 0, 1], p=[0.3, 0.4, 0.3])
        noise = np.random.randn() * volatility
        price_change = trend * volatility * 0.2 + noise
        price += price_change
        
        timestamps.append(ts)
        opens.append(price)
        highs.append(price + abs(np.random.randn()) * volatility * 0.5)
        lows.append(price - abs(np.random.randn()) * volatility * 0.5)
        closes.append(price + price_change * 0.5)
        volumes.append(int(np.random.uniform(500, 5000)))
        spreads.append(int(np.random.uniform(20, 80)))
    
    return pd.DataFrame({
        'time': timestamps, 'open': opens, 'high': highs, 'low': lows,
        'close': closes, 'tick_volume': volumes, 'spread': spreads
    })

def run_strategy(candles, strat_name, strat_config):
    from core.common.types import CandleArray, TradeSignal
    from core.indicator_engine import IndicatorEngine
    
    features = IndicatorEngine.precalculate_all("BTCUSDm", "M5", candles)
    for k, v in features.items():
        candles._indicators[k] = v
    
    if strat_name == "TrendFollowing":
        ema50 = candles.get_indicator("ema_50")
        ema200 = candles.get_indicator("ema_200")
        adx = candles.get_indicator("adx_14")
        
        if len(ema200) < 5 or len(adx) < 1:
            return None
        
        if ema50[-1] > ema200[-1] and adx[-1] > strat_config.get("adx_threshold", 20):
            return ("BUY", 50)
        elif ema50[-1] < ema200[-1] and adx[-1] > strat_config.get("adx_threshold", 20):
            return ("SELL", 50)
    
    elif strat_name == "LiquiditySweepBreakout":
        if len(candles.close) < 30:
            return None
        
        high = np.max(candles.close[-30:])
        low = np.min(candles.close[-30:])
        
        if candles.close[-1] > high * 0.99:
            return ("SELL", 40)
        elif candles.close[-1] < low * 1.01:
            return ("BUY", 40)
    
    elif strat_name == "SmartMeanReversion":
        rsi = candles.get_indicator("rsi_14")
        if len(rsi) < 1:
            return None
        
        if rsi[-1] < 35:
            return ("BUY", 30)
        elif rsi[-1] > 65:
            return ("SELL", 30)
    
    return None

def run_backtest(symbol="BTCUSDm"):
    logger.info("=" * 60)
    logger.info(f"BACKTEST - {symbol}")
    logger.info("=" * 60)
    
    df = generate_synthetic_data(bars=10000)
    
    from core.common.types import CandleArray
    candles = CandleArray(
        time=np.array(df['time'].values, dtype=np.int64),
        open=np.array(df['open'].values, dtype=np.float64),
        high=np.array(df['high'].values, dtype=np.float64),
        low=np.array(df['low'].values, dtype=np.float64),
        close=np.array(df['close'].values, dtype=np.float64),
        tick_volume=np.array(df['tick_volume'].values, dtype=np.int64),
        spread=np.array(df['spread'].values, dtype=np.float64)
    )
    
    from core.config.loader import ConfigLoader
    config = ConfigLoader()
    symbol_cfg = config.get_symbol_config(symbol)
    
    strategies = {}
    
    # Strategies are at root level after config merge
    for key in symbol_cfg.keys():
        if key in ["TrendFollowing", "LiquiditySweepBreakout", "SmartMeanReversion", "RangeBounce", "PureBreakoutOneMinute"]:
            strategies[key] = symbol_cfg[key]
    
    logger.info(f"Loaded strategies: {list(strategies.keys())}")
    
    results = {
        "symbol": symbol,
        "start_time": df['time'].iloc[0],
        "end_time": df['time'].iloc[-1],
        "total_bars": len(df),
        "strategies": {}
    }
    
    for strat_name, strat_cfg in strategies.items():
        enabled = strat_cfg.get("enabled", False)
        logger.info(f"  {strat_name} enabled: {enabled}")
        
        if not enabled:
            continue
        
        logger.info(f"Running {strat_name}...")
        
        trades = []
        equity = 10000.0
        
        for i in range(200, len(candles.time) - 100, 50):
            c = CandleArray(
                time=candles.time[:i+1],
                open=candles.open[:i+1],
                high=candles.high[:i+1],
                low=candles.low[:i+1],
                close=candles.close[:i+1],
                tick_volume=candles.tick_volume[:i+1],
                spread=candles.spread[:i+1]
            )
            
            signal = run_strategy(c, strat_name, strat_cfg)
            
            if signal:
                direction, sl_pips = signal
                entry = c.close[-1]
                exit_idx = min(i + 50, len(candles.close) - 1)
                exit_price = candles.close[exit_idx]
                
                if direction == "BUY":
                    pnl = (exit_price - entry) / entry * 100 * 10000
                else:
                    pnl = (entry - exit_price) / entry * 100 * 10000
                
                equity += pnl
                trades.append({"entry": entry, "exit": exit_price, "pnl": pnl, "direction": direction})
        
        wins = sum(1 for t in trades if t["pnl"] > 0)
        win_rate = wins / len(trades) * 100 if trades else 0
        
        results["strategies"][strat_name] = {
            "total_trades": len(trades),
            "wins": wins,
            "losses": len(trades) - wins,
            "win_rate": round(win_rate, 1),
            "final_equity": round(equity, 2),
            "profit_pct": round((equity - 10000) / 100, 2)
        }
        
        logger.info(f"  {strat_name}: {len(trades)} trades, {win_rate:.1f}% win, ${equity:.2f}")
    
    return results

if __name__ == "__main__":
    results = run_backtest("BTCUSDm")
    
    print("\n" + "=" * 60)
    print("BACKTEST RESULTS")
    print("=" * 60)
    
    for strat_name, stats in results["strategies"].items():
        print(f"\n{strat_name}:")
        print(f"  Trades: {stats['total_trades']} (W:{stats['wins']} L:{stats['losses']})")
        print(f"  Win Rate: {stats['win_rate']}%")
        print(f"  Profit: {stats['profit_pct']}%")
        print(f"  Final: ${stats['final_equity']}")