"""
Strategy Signal Generation Verification Suite
Tests that all strategies can generate valid signals with proper market data.
"""
import sys
import os
sys.path.append(os.getcwd())

import numpy as np
from datetime import datetime, timezone
import logging

from core.common.types import CandleArray, TradeSignal
from core.base_strategy import MarketData
from core.regime_detector import RegimeDetector
from strategies import create_strategy

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("signal_test")

class StrategySignalTester:
    """Tests strategy signal generation capabilities."""
    
    def __init__(self):
        self.results = []
        
    def generate_realistic_data(self, n=500):
        """Generate more realistic candle data that triggers strategy conditions."""
        t = (np.arange(n) * 300 + 1700000000).astype(np.int64)
        
        # Create trending data with proper structure
        base = 2000.0
        trend = np.linspace(0, 100, n)
        noise = np.cumsum(np.random.normal(0, 0.5, n))
        close = base + trend + noise
        
        # Ensure enough volatility
        close = base + np.sin(np.linspace(0, 15, n)) * 30 + np.cumsum(np.random.normal(0.1, 2, n))
        
        high = close + np.abs(np.random.uniform(0.5, 2, n))
        low = close - np.abs(np.random.uniform(0.5, 2, n))
        open_price = low + np.random.uniform(0.2, 0.8, n) * (high - low)
        
        spread = np.random.randint(10, 30, n)
        tick_volume = np.random.randint(100, 800, n)
        
        return CandleArray(
            time=t, open=open_price, high=high, low=low, 
            close=close, tick_volume=tick_volume, spread=spread
        )
    
    def test_strategy_signal_generation(self, strategy_name, config):
        """Test that a strategy can generate signals."""
        logger.info(f"Testing {strategy_name}...")
        
        sid = f"{strategy_name.lower()}_v4"
        st_type = strategy_name.upper()
        
        try:
            strategy = create_strategy(sid, st_type, config)
        except Exception as e:
            return {"strategy": strategy_name, "status": "CREATE_ERROR", "error": str(e)}
        
        # Generate realistic market data
        m5 = self.generate_realistic_data(500)
        m15 = self.generate_realistic_data(200)
        h1 = self.generate_realistic_data(100)
        
        # Create market data with proper structure
        current_price = float(m5.close[-1])
        
        md = MarketData(
            symbol="XAUUSDm",
            htf_candles=h1,
            m15_candles=m15,
            m5_candles=m5,
            d1_candles=None,
            current_price=current_price,
            bid=current_price,
            ask=current_price + 0.5,
            spread=0.5,
            session="LONDON",
            timestamp=datetime.now(timezone.utc)
        )
        
        # Test signal generation
        try:
            signal = strategy.generate_signal(md)
            
            if signal is None:
                return {"strategy": strategy_name, "status": "NO_SIGNAL", "signal_type": "None"}
            
            return {
                "strategy": strategy_name,
                "status": "SUCCESS",
                "signal_direction": signal.direction,
                "signal_confidence": signal.confidence,
                "has_stop_loss": hasattr(signal, 'stop_loss') and signal.stop_loss > 0,
                "has_take_profit": hasattr(signal, 'take_profit') and signal.take_profit > 0,
            }
            
        except Exception as e:
            return {"strategy": strategy_name, "status": "SIGNAL_ERROR", "error": str(e)}
    
    def test_stop_loss_take_profit(self, strategy_name, config):
        """Test that strategy generates proper SL/TP levels."""
        sid = f"{strategy_name.lower()}_v4"
        st_type = strategy_name.upper()
        
        try:
            strategy = create_strategy(sid, st_type, config)
        except:
            return None
        
        m5 = self.generate_realistic_data(500)
        current_price = float(m5.close[-1])
        
        md = MarketData(
            symbol="XAUUSDm",
            htf_candles=m5,
            m15_candles=m5,
            m5_candles=m5,
            d1_candles=None,
            current_price=current_price,
            bid=current_price,
            ask=current_price + 0.5,
            spread=0.5,
            session="LONDON",
            timestamp=datetime.now(timezone.utc)
        )
        
        # Generate a BUY signal first
        signal = TradeSignal(
            direction="BUY",
            price=current_price,
            confidence=0.8,
            timestamp=datetime.now(timezone.utc)
        )
        
        try:
            sl = strategy.get_stop_loss(signal, md)
            tp = strategy.get_take_profit(signal, md)
            
            return {
                "strategy": strategy_name,
                "has_sl": sl > 0,
                "has_tp": tp > 0,
                "sl_distance": abs(current_price - sl) if sl > 0 else 0,
                "tp_distance": abs(tp - current_price) if tp > 0 else 0,
                "valid_sl": (sl > 0 and sl < current_price) if signal.direction == "BUY" else (sl > current_price),
                "valid_tp": (tp > current_price) if signal.direction == "BUY" else (tp < current_price),
            }
        except Exception as e:
            return {"strategy": strategy_name, "status": "SLTP_ERROR", "error": str(e)}


def run_signal_tests():
    """Run all strategy signal tests."""
    print("=" * 80)
    print(" STRATEGY SIGNAL GENERATION VERIFICATION ")
    print("=" * 80)
    
    tester = StrategySignalTester()
    
    config = {
        "TrendFollowing": {
            "enabled": True,
            "allowed_sessions": ["LONDON", "TOKYO", "NEW_YORK"],
            "adx_threshold": 20,
            "sl_atr": 2.0,
            "rr_target": 2.0
        },
        "LiquiditySweepBreakout": {
            "enabled": True,
            "allowed_sessions": ["LONDON", "TOKYO"],
            "body_thresh": 0.65,
            "h1_strength_thresh": 0.50,
            "sl_atr": 2.0,
            "rr_target": 4.0
        },
        "SmartMeanReversion": {
            "enabled": True,
            "allowed_sessions": ["NEW_YORK", "LONDON"],
            "bb_period": 20,
            "bb_std": 2.2,
            "rsi_period": 14,
            "rsi_overbought": 70,
            "rsi_oversold": 30,
            "sl_atr": 1.5,
            "rr_target": 2.0
        },
        "LiquiditySession": {
            "enabled": True,
            "allowed_sessions": ["LONDON", "NEW_YORK"],
            "range_maturity_limit": 4.0,
            "vol_trigger_mult": 0.5,
            "sl_atr": 1.5,
            "rr_target": 8.0
        }
    }
    
    strategies = ["TrendFollowing", "LiquiditySweepBreakout", "SmartMeanReversion", "LiquiditySession"]
    
    print("\n[1] SIGNAL GENERATION TESTS")
    print("-" * 60)
    
    for strategy in strategies:
        result = tester.test_strategy_signal_generation(strategy, config)
        if result:
            status = result.get("status", "UNKNOWN")
            if status == "SUCCESS":
                print(f"  {strategy:25} | PASS | Direction: {result.get('signal_direction', 'N/A'):5} | Confidence: {result.get('signal_confidence', 0):.2f}")
            else:
                print(f"  {strategy:25} | {status}")
    
    print("\n[2] STOP LOSS / TAKE PROFIT TESTS")
    print("-" * 60)
    
    for strategy in strategies:
        result = tester.test_stop_loss_take_profit(strategy, config)
        if result:
            if result.get("status") == "SLTP_ERROR":
                print(f"  {strategy:25} | ERROR: {result.get('error', 'Unknown')}")
            else:
                sl_ok = "OK" if result.get("valid_sl") else "FAIL"
                tp_ok = "OK" if result.get("valid_tp") else "FAIL"
                print(f"  {strategy:25} | SL: {sl_ok} ({result.get('sl_distance', 0):.2f}) | TP: {tp_ok} ({result.get('tp_distance', 0):.2f})")
    
    print("\n[3] REGIME DETECTION TEST")
    print("-" * 60)
    
    regime_detector = RegimeDetector()
    m5 = tester.generate_realistic_data(500)
    
    regime = regime_detector.detect(m5)
    print(f"  Detected Regime: {regime.market_type.value}")
    print(f"  ADX: {regime.adx:.2f}")
    print(f"  Volatility: {regime.volatility.value}")
    
    print("\n" + "=" * 80)
    print(" SIGNAL VERIFICATION COMPLETE ")
    print("=" * 80)


if __name__ == "__main__":
    run_signal_tests()
