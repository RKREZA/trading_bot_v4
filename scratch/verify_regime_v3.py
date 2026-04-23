import unittest
import numpy as np
from datetime import datetime
from dataclasses import replace
import os
import sys
import time

# Add project root to path
sys.path.append(os.getcwd())

from core.regime_detector import RegimeDetector, RegimeState, MarketRegime
from core.regime_store import SQLiteWALRegimeStore, MemoryRegimeStore
from core.base_strategy import MarketData

class MockCandles:
    def __init__(self, close_val=2000.0, vol_val=100.0, length=500):
        self.close = np.full(length, close_val)
        self.tick_volume = np.full(length, vol_val)
        self.high = self.close + 1.0
        self.low = self.close - 1.0
        self.open = self.close
        self.time = np.arange(length)
        
    def __len__(self):
        return len(self.close)
    
    @property
    def c(self): return self.close
    @property
    def v(self): return self.tick_volume
    @property
    def h(self): return self.high
    @property
    def l(self): return self.low
    @property
    def o(self): return self.open

    def get_indicator(self, name):
        if "atr" in name: return np.full(len(self.close), 1.0)
        if "adx" in name: return np.full(len(self.close), 25.0)
        if "ema_50" in name: return np.full(len(self.close), 2000.0)
        if "ema_200" in name: return np.full(len(self.close), 1950.0)
        if "bb_20_2_upper" in name: return np.full(len(self.close), 2010.0)
        if "bb_20_2_lower" in name: return np.full(len(self.close), 1990.0)
        return np.zeros(len(self.close))

    def ema(self, p): return self.get_indicator(f"ema_{p}")
    def adx(self, p): return self.get_indicator(f"adx")
    def atr(self, p): return self.get_indicator(f"atr")
    def bollinger_bands(self, p, m):
        upper = self.get_indicator(f"bb_{p}_{int(m)}_upper")
        lower = self.get_indicator(f"bb_{p}_{int(m)}_lower")
        mid = self.close
        return upper, lower, mid

class TestRegimeEngineV3(unittest.TestCase):
    def setUp(self):
        self.detector = RegimeDetector()
        self.m5 = MockCandles()
        self.h1 = MockCandles()
        self.market_data = MarketData(
            symbol="XAUUSDm",
            htf_candles=self.h1,
            m15_candles=self.m5,
            m5_candles=self.m5,
            d1_candles=None,
            current_price=2000.0,
            bid=2000.0,
            ask=2001.0,
            spread=1.0,
            point=0.01,
            session="LONDON",
            timestamp=datetime.now()
        )
        self.state = RegimeState()
        self.exec_id = "TEST:SESSION:12345"

    def test_determinism(self):
        """Rule 1: Same input + state = Same output"""
        res1, state1, trace1 = self.detector.detect(self.market_data, self.state, self.exec_id, "STRAT_A")
        res2, state2, trace2 = self.detector.detect(self.market_data, self.state, self.exec_id, "STRAT_A")
        
        self.assertEqual(res1.market_type, res2.market_type)
        self.assertEqual(res1.confidence, res2.confidence)
        self.assertEqual(state1, state2)
        self.assertEqual(trace1, trace2)

    def test_state_isolation(self):
        """Verify that strategy_id isolates state in stores"""
        store = MemoryRegimeStore()
        state_a = RegimeState(breakout_count=5)
        state_b = RegimeState(breakout_count=0)
        
        store.save("STRAT_A", state_a)
        store.save("STRAT_B", state_b)
        
        self.assertEqual(store.load("STRAT_A").breakout_count, 5)
        self.assertEqual(store.load("STRAT_B").breakout_count, 0)

    def test_priority_logic(self):
        """Verify priority: Liquidity > Rejection > Transition..."""
        # Simulate a SWEEP:
        # HTF range in last 24 bars is [1999, 2001]
        self.h1.high = np.full(500, 2001.0)
        self.h1.low = np.full(500, 1999.0)
        
        # Current bar high is 2002 (breaks HTF high), close is 2000 (back inside)
        self.m5.high[-1] = 2002.0
        self.m5.close[-1] = 2000.0
        
        res, _, trace = self.detector.detect(self.market_data, self.state, self.exec_id, "TEST")
        self.assertEqual(res.market_type, MarketRegime.LIQUIDITY_EVENT)
        self.assertIn("LIQUIDITY_PRIORITY", trace.decision_path)

    def test_sqlite_persistence(self):
        """Verify SQLite WAL store works"""
        db_path = "scratch/test_regime_v3_tmp.db"
        if os.path.exists(db_path): 
            try: os.remove(db_path)
            except: pass
        
        store = SQLiteWALRegimeStore(db_path)
        state = RegimeState(breakout_count=3, last_direction="BUY")
        store.save("TEST_STRAT", state)
        
        loaded = store.load("TEST_STRAT")
        self.assertEqual(loaded.breakout_count, 3)
        self.assertEqual(loaded.last_direction, "BUY")
        
        # Cleanup
        try: store.conn.close()
        except: pass
        if os.path.exists(db_path): 
            try: os.remove(db_path)
            except: pass

if __name__ == "__main__":
    unittest.main()
