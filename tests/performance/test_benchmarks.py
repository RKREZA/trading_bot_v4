import sys
import os
sys.path.append(os.getcwd())

import time
import unittest
import numpy as np
from datetime import datetime, timezone

class TestPerformanceBenchmarks(unittest.TestCase):
    """Performance benchmarks to ensure the bot meets latency requirements."""

    @classmethod
    def setUpClass(cls):
        cls.results = {}

    def _benchmark(self, name, func, iterations=100):
        """Helper to benchmark a function."""
        times = []
        for _ in range(iterations):
            start = time.perf_counter()
            func()
            elapsed = time.perf_counter() - start
            times.append(elapsed * 1000)  # Convert to ms
        
        avg = np.mean(times)
        p50 = np.percentile(times, 50)
        p95 = np.percentile(times, 95)
        p99 = np.percentile(times, 99)
        
        self.results[name] = {
            "avg_ms": avg,
            "p50_ms": p50,
            "p95_ms": p95,
            "p99_ms": p99
        }
        
        return avg, p95

    def test_candle_array_creation(self):
        """Benchmark CandleArray creation from list of dicts."""
        from core.common.types import CandleArray
        
        candles_list = [
            {
                "time": 1700000000 + i * 60,
                "open": 2000.0 + i * 0.1,
                "high": 2001.0 + i * 0.1,
                "low": 1999.0 + i * 0.1,
                "close": 2000.5 + i * 0.1,
                "tick_volume": 100 + i,
                "spread": 10 + i % 5
            }
            for i in range(100)
        ]
        
        avg, p95 = self._benchmark("CandleArray.from_dicts(100)", 
            lambda: CandleArray.from_dicts(candles_list))
        
        self.assertLess(p95, 10, f"CandleArray creation too slow: {p95:.2f}ms")
        print(f"  CandleArray(100): avg={avg:.3f}ms, p95={p95:.3f}ms")

    def test_indicator_calculation(self):
        """Benchmark EMA calculation."""
        from core.common.types import CandleArray
        
        candles_list = [
            {
                "time": 1700000000 + i * 60,
                "open": 2000.0 + np.sin(i/10),
                "high": 2001.0 + np.sin(i/10),
                "low": 1999.0 + np.sin(i/10),
                "close": 2000.5 + np.sin(i/10),
                "tick_volume": 100,
                "spread": 10
            }
            for i in range(1000)
        ]
        
        candles = CandleArray.from_dicts(candles_list)
        
        avg, p95 = self._benchmark("EMA(50) on 1000 candles",
            lambda: candles.ema(50))
        
        self.assertLess(p95, 50, f"EMA calculation too slow: {p95:.2f}ms")
        print(f"  EMA(50) on 1000: avg={avg:.3f}ms, p95={p95:.3f}ms")

    def test_risk_guardian_lot_calculation(self):
        """Benchmark lot size calculation."""
        from core.risk.risk_guardian import RiskGuardian
        
        config = {
            "risk_governance": {
                "risk_per_trade_pct": 1.0,
                "max_daily_loss_pct": 2.0,
                "max_drawdown_halt_pct": 10.0
            },
            "backtest": {"initial_balance": 10000}
        }
        
        guardian = RiskGuardian(config)
        symbol_info = {
            "point": 0.01,
            "tick_value": 1.0,
            "min_lot": 0.01,
            "max_lot": 20.0,
            "lot_step": 0.01
        }
        
        avg, p95 = self._benchmark("Lot size calculation",
            lambda: guardian.calculate_lot_size(10000, 2.5, symbol_info))
        
        self.assertLess(p95, 1, f"Lot calculation too slow: {p95:.2f}ms")
        print(f"  Lot calculation: avg={avg:.3f}ms, p95={p95:.3f}ms")

    def test_signal_validation(self):
        """Benchmark signal validation."""
        from core.risk.risk_guardian import RiskGuardian
        from core.common.types import TradeSignal, CandleArray
        from core.base_strategy import MarketData
        
        config = {
            "risk_governance": {
                "risk_per_trade_pct": 1.0,
                "max_daily_loss_pct": 2.0,
                "max_drawdown_halt_pct": 10.0
            },
            "backtest": {"initial_balance": 10000}
        }
        
        guardian = RiskGuardian(config)
        
        candles_list = [
            {
                "time": 1700000000 + i * 60,
                "open": 2000.0,
                "high": 2001.0,
                "low": 1999.0,
                "close": 2000.0,
                "tick_volume": 100,
                "spread": 10
            }
            for i in range(100)
        ]
        
        candles = CandleArray.from_dicts(candles_list)
        
        signal = TradeSignal(
            direction="BUY",
            price=2000.0,
            stop_loss=1995.0,
            take_profit=2010.0,
            confidence=0.8
        )
        
        market_data = MarketData(
            symbol="XAUUSD",
            htf_candles=candles,
            m15_candles=candles,
            m5_candles=candles,
            d1_candles=candles,
            current_price=2000.0,
            bid=2000.0,
            ask=2001.0,
            spread=1.0,
            session="LONDON",
            timestamp=datetime.now(timezone.utc)
        )
        
        symbol_info = {"point": 0.01, "tick_value": 1.0, "min_lot": 0.01, "max_lot": 20.0}
        
        avg, p95 = self._benchmark("Signal validation",
            lambda: guardian.validate_signal(signal, 10000, market_data, symbol_info))
        
        self.assertLess(p95, 5, f"Signal validation too slow: {p95:.2f}ms")
        print(f"  Signal validation: avg={avg:.3f}ms, p95={p95:.3f}ms")

    def test_news_filter_lookup(self):
        """Benchmark news filter blocking check."""
        from core.news_filter import InstitutionalNewsFilter
        
        config = {
            "news_filter": {
                "enabled": True,
                "buffer_before_min": 30,
                "buffer_after_min": 15
            }
        }
        
        nf = InstitutionalNewsFilter(config)
        nf.events = [
            {"title": "FOMC", "country": "USD", "impact": "High", "timestamp": time.time() + 600},
            {"title": "GDP", "country": "GBP", "impact": "High", "timestamp": time.time() + 1200},
        ]
        
        avg, p95 = self._benchmark("News filter lookup",
            lambda: nf.is_blocked("EURUSD", time.time()))
        
        self.assertLess(p95, 1, f"News lookup too slow: {p95:.2f}ms")
        print(f"  News lookup: avg={avg:.3f}ms, p95={p95:.3f}ms")

    def test_session_detection(self):
        """Benchmark session detection."""
        from core.session_detector import SessionDetector
        
        configs = [
            {"backtest": {"utc_offset": 0}},
            {"backtest": {"utc_offset": 3}},
            {"backtest": {"utc_offset": -5}},
        ]
        
        test_times = [
            datetime(2024, 1, 15, 8, 0),   # London
            datetime(2024, 1, 15, 14, 0),  # New York
            datetime(2024, 1, 15, 0, 0),    # Tokyo
        ]
        
        avg, p95 = self._benchmark("Session detection",
            lambda: [SessionDetector.get_session(t, c.get("backtest", {}).get("utc_offset", 0)) 
                     for c in configs for t in test_times])
        
        self.assertLess(p95, 2, f"Session detection too slow: {p95:.2f}ms")
        print(f"  Session detection: avg={avg:.3f}ms, p95={p95:.3f}ms")

    @classmethod
    def tearDownClass(cls):
        print("\n" + "=" * 60)
        print("PERFORMANCE BENCHMARK RESULTS")
        print("=" * 60)
        print(f"{'Test':<35} {'Avg (ms)':<12} {'P95 (ms)':<12}")
        print("-" * 60)
        for name, metrics in cls.results.items():
            print(f"{name:<35} {metrics['avg_ms']:<12.4f} {metrics['p95_ms']:<12.4f}")
        print("=" * 60)


class TestLatencyRequirements(unittest.TestCase):
    """Verify the bot meets critical latency requirements."""

    def test_trading_cycle_under_1_second(self):
        """Critical: Full trading cycle must complete in under 1 second."""
        from core.risk.risk_guardian import RiskGuardian
        from core.news_filter import InstitutionalNewsFilter
        from core.common.types import CandleArray, TradeSignal
        from core.base_strategy import MarketData
        
        start = time.perf_counter()
        
        # Simulate a full trading cycle
        config = {"risk_governance": {"risk_per_trade_pct": 1.0, "max_daily_loss_pct": 2.0, "max_drawdown_halt_pct": 10.0}}
        
        guardian = RiskGuardian(config)
        
        candles_list = [
            {
                "time": 1700000000 + i * 60,
                "open": 2000.0,
                "high": 2001.0,
                "low": 1999.0,
                "close": 2000.0,
                "tick_volume": 100,
                "spread": 10
            }
            for i in range(100)
        ]
        candles = CandleArray.from_dicts(candles_list)
        
        md = MarketData("XAUUSD", candles, candles, candles, candles, 2000.0, 2000.0, 2001.0, 1.0, "LONDON", datetime.now(timezone.utc))
        
        signal = TradeSignal("BUY", 2000.0, 0.8, stop_loss=1995.0, take_profit=2010.0)
        sym_info = {"point": 0.01, "tick_value": 1.0, "min_lot": 0.01, "max_lot": 20.0}
        
        guardian.validate_signal(signal, 10000, md, sym_info)
        guardian.calculate_lot_size(10000, 2.5, sym_info)
        guardian.check_governance(10000, 10000)
        
        elapsed = time.perf_counter() - start
        
        self.assertLess(elapsed, 1.0, f"Trading cycle too slow: {elapsed*1000:.2f}ms")
        print(f"\nTrading cycle latency: {elapsed*1000:.2f}ms (requirement: <1000ms)")

    def test_memory_usage_acceptable(self):
        """Verify memory usage stays within acceptable bounds."""
        import sys
        
        from core.common.types import CandleArray
        
        candles_list = [
            {
                "time": 1700000000 + i * 60,
                "open": 2000.0,
                "high": 2001.0,
                "low": 1999.0,
                "close": 2000.0,
                "tick_volume": 100,
                "spread": 10
            }
            for i in range(1000)
        ]
        
        candles = CandleArray.from_dicts(candles_list)
        
        size_bytes = sys.getsizeof(candles.close) + sys.getsizeof(candles.high) + sys.getsizeof(candles.low) + sys.getsizeof(candles.open)
        
        self.assertLess(size_bytes, 1_000_000, f"Memory usage too high: {size_bytes/1024:.2f}KB")
        print(f"\nCandleArray(1000) memory: {size_bytes/1024:.2f}KB")


if __name__ == "__main__":
    unittest.main(verbosity=2)
