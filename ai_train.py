import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
import logging
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ai_train")

def generate_synthetic_data(symbol="XAUUSDm", bars=20000):
    np.random.seed(42)
    logger.info(f"Generating {bars} bars of synthetic data for {symbol}...")
    
    timestamps = []
    opens = []
    highs = []
    lows = []
    closes = []
    volumes = []
    spreads = []
    
    base_price = 2000.0
    price = base_price
    volatility = 15.0
    trend = 0
    
    for i in range(bars):
        ts = 1704067200 + i * 300
        timestamps.append(ts)
        
        if i % 3000 == 0:
            trend = np.random.choice([-1, 0, 1], p=[0.3, 0.4, 0.3])
            volatility = np.random.uniform(10, 25)
        
        noise = np.random.randn() * volatility
        price_change = trend * volatility * 0.3 + noise
        price = price + price_change
        
        high = price + abs(np.random.randn()) * volatility * 0.5
        low = price - abs(np.random.randn()) * volatility * 0.5
        
        opens.append(price)
        highs.append(high)
        lows.append(low)
        closes.append(price + price_change * 0.5)
        volumes.append(int(np.random.uniform(100, 1000)))
        spreads.append(np.random.uniform(20, 50))
    
    return pd.DataFrame({
        'time': timestamps,
        'open': opens,
        'high': highs,
        'low': lows,
        'close': closes,
        'tick_volume': volumes,
        'spread': spreads
    })

def create_candle_array(df):
    from core.common.types import CandleArray
    return CandleArray(
        time=np.array(df['time'].values, dtype=np.int64),
        open=np.array(df['open'].values, dtype=np.float64),
        high=np.array(df['high'].values, dtype=np.float64),
        low=np.array(df['low'].values, dtype=np.float64),
        close=np.array(df['close'].values, dtype=np.float64),
        tick_volume=np.array(df['tick_volume'].values, dtype=np.int64),
        spread=np.array(df['spread'].values, dtype=np.float64)
    )

def run_strategy(candles, strategy_name):
    from core.common.types import TradeSignal
    from core.indicator_engine import IndicatorEngine
    
    features = IndicatorEngine.precalculate_all("XAUUSDm", "M5", candles)
    
    for k, v in features.items():
        candles._indicators[k] = v
    
    direction = "NONE"
    sl_pips = 300
    
    if strategy_name == "TrendFollowing":
        ema_fast = candles.get_indicator("ema_50")
        ema_mid = candles.get_indicator("ema_100")
        ema_slow = candles.get_indicator("ema_200")
        adx = candles.get_indicator("adx_14")
        
        if len(ema_slow) < 5 or len(adx) < 1:
            return None, 300
        
        is_long = ema_fast[-1] > ema_mid[-1] > ema_slow[-1]
        is_short = ema_fast[-1] < ema_mid[-1] < ema_slow[-1]
        strong_trend = adx[-1] > 25
        
        if is_long and strong_trend:
            direction = "BUY"
            sl_pips = 300
        elif is_short and strong_trend:
            direction = "SELL"
            sl_pips = 300
    
    elif strategy_name == "LiquiditySweepBreakout":
        if len(candles.close) < 20:
            return None, 400
        
        recent = candles.close[-20:]
        high = np.max(recent)
        low = np.min(recent)
        current = candles.close[-1]
        
        body_ratio = abs(candles.close[-1] - candles.open[-1]) / (candles.high[-1] - candles.low[-1] + 0.001)
        
        if current > high * 0.99 and body_ratio > 0.7:
            direction = "SELL"
            sl_pips = 400
        elif current < low * 1.01 and body_ratio > 0.7:
            direction = "BUY"
            sl_pips = 400
    
    elif strategy_name == "SmartMeanReversion":
        rsi = candles.get_indicator("rsi_14")
        bb_upper = candles.get_indicator("bb_upper")
        bb_lower = candles.get_indicator("bb_lower")
        
        if len(rsi) < 1 or len(bb_upper) < 1:
            return None, 200
        
        current = candles.close[-1]
        
        if rsi[-1] < 30 and current < bb_lower[-1]:
            direction = "BUY"
            sl_pips = 200
        elif rsi[-1] > 70 and current > bb_upper[-1]:
            direction = "SELL"
            sl_pips = 200
    
    return direction if direction != "NONE" else None, sl_pips

def main():
    logger.info("=" * 60)
    logger.info("AI MODEL TRAINING PIPELINE")
    logger.info("=" * 60)
    
    df = generate_synthetic_data(bars=20000)
    candles = create_candle_array(df)
    
    config = json.load(open("config/config.json"))
    
    all_trades = []
    strategies = ["TrendFollowing", "LiquiditySweepBreakout", "SmartMeanReversion"]
    
    for strategy_name in strategies:
        logger.info(f"Running backtest for {strategy_name}...")
        count = 0
        
        for i in range(200, len(candles.time) - 100, 50):
            c = create_candle_array(df.iloc[:i+1])
            direction, sl_pips = run_strategy(c, strategy_name)
            
            if direction:
                entry = c.close[-1]
                exit = df['close'].iloc[min(i+50, len(df)-1)]
                pnl_pct = (exit - entry) / entry * 100
                outcome = 1 if pnl_pct > 0 else 0
                
                from core.ai.features import FeatureEngineer
                features = FeatureEngineer.extract_features(c, direction, sl_pips)
                
                if features:
                    features['outcome'] = outcome
                    features['pnl_pct'] = pnl_pct
                    features['strategy'] = strategy_name
                    all_trades.append(features)
                    count += 1
        
        logger.info(f"  -> {count} trades collected")
    
    logger.info(f"Total trades: {len(all_trades)}")
    
    if len(all_trades) < 50:
        logger.warning("Insufficient trades, generating synthetic data...")
        for _ in range(500):
            all_trades.append({
                'body_ratio': np.random.uniform(0.1, 0.9),
                'upper_wick_ratio': np.random.uniform(0, 0.4),
                'lower_wick_ratio': np.random.uniform(0, 0.4),
                'atr_ratio': np.random.uniform(0.5, 1.5),
                'spread_ratio': np.random.uniform(0.8, 1.2),
                'adx': np.random.uniform(10, 40),
                'hour_of_day': np.random.randint(0, 24),
                'day_of_week': np.random.randint(0, 5),
                'signal_dir': np.random.choice([-1, 1]),
                'sl_pips': np.random.uniform(100, 500),
                'outcome': np.random.choice([0, 1], p=[0.4, 0.6]),
                'pnl_pct': np.random.uniform(-2, 3),
                'strategy': 'SYNTHETIC'
            })
    
    df_train = pd.DataFrame(all_trades)
    
    feature_cols = ['body_ratio', 'upper_wick_ratio', 'lower_wick_ratio', 'atr_ratio', 
                  'spread_ratio', 'adx', 'hour_of_day', 'day_of_week', 'signal_dir', 'sl_pips']
    
    X = df_train[feature_cols].fillna(0)
    y = df_train['outcome'].fillna(0).astype(int)
    
    logger.info(f"Training: {len(X)} samples, {y.sum()} wins ({y.mean()*100:.1f}%)")
    
    from core.ai.model import AIModelWrapper
    model = AIModelWrapper()
    model.train(X, y)
    
    import joblib
    joblib.dump(X, os.path.join(model.model_dir, "v4_rf_baseline.pkl"))
    
    logger.info("=" * 60)
    logger.info(f"TRAINING COMPLETE - Model saved to {model.model_path}")
    logger.info("=" * 60)
    
    from core.ai.predictor import AIPredictor
    predictor = AIPredictor(config)
    logger.info(f"AI Enabled: {predictor.enabled}")
    logger.info(f"AI Ready: {predictor.engine.is_ready}")
    
    test_features = {
        'body_ratio': 0.7,
        'upper_wick_ratio': 0.1,
        'lower_wick_ratio': 0.2,
        'atr_ratio': 1.2,
        'spread_ratio': 1.0,
        'adx': 30,
        'hour_of_day': 14,
        'day_of_week': 2,
        'signal_dir': 1,
        'sl_pips': 300
    }
    
    prob = predictor.engine.predict_probability(test_features)
    logger.info(f"Test prediction (BUY signal): {prob:.2%} win probability")

if __name__ == "__main__":
    main()