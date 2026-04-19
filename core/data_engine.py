import logging
import time
import threading
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timezone, timedelta
from collections import deque
from enum import Enum

from core.connection import MT5Connection
from core.data.manager import DataManager
from core.indicator_engine import IndicatorEngine
from core.common.types import CandleArray

logger = logging.getLogger("trading_bot.data_engine")

class DataQuality(Enum):
    EXCELLENT = "EXCELLENT"
    GOOD = "GOOD"
    STALE = "STALE"
    INVALID = "INVALID"

class MarketState:
    """Snapshot of processed market data for a symbol across multiple timeframes."""
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.timestamp = 0.0
        self.m5: Optional[CandleArray] = None
        self.m15: Optional[CandleArray] = None
        self.h1: Optional[CandleArray] = None
        self.d1: Optional[CandleArray] = None
        self.last_updated = 0.0
        self.quality: DataQuality = DataQuality.INVALID
        self.gaps: List[Tuple[str, float, float]] = []
        self.bar_count = 0

class DataEngine:
    """
    V6-INSIGNIA Asynchronous Data & Indicator Engine (Institutional Pillar 3).
    Runs on a dedicated background thread to fetch candles and calculate 
    complex technical indicators (ADX, ATR, RSI, EMAs) non-blockingly.
    Features multi-symbol batching, data quality validation, and health monitoring.
    """
    
    def __init__(self, connection: MT5Connection, config: dict):
        self.connection = connection
        self.config = config
        self.data_manager = DataManager(config)
        self.states: Dict[str, MarketState] = {}
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        
        self.update_interval = config.get("performance", {}).get("data_engine_interval", 2.0)
        self.symbols = list(config.get("symbols_config", {}).keys())
        if "XAUUSDm" not in self.symbols:
            self.symbols.append("XAUUSDm")

        self._history: deque = deque(maxlen=100)
        self._error_count = 0
        self._last_health_check = 0.0

    def start(self):
        """Launches the background processing thread."""
        logger.info(f"DataEngine Service starting for symbols: {self.symbols}")
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def stop(self):
        """Gracefully halts the background thread."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=2.0)

    def get_state(self, symbol: str) -> Optional[MarketState]:
        """Provides a non-blocking snapshot of the latest computed market data."""
        with self._lock:
            return self.states.get(symbol)

    def get_health(self) -> Dict[str, Any]:
        """Returns health metrics for monitoring."""
        return {
            "symbols_tracked": len(self.symbols),
            "active_states": len(self.states),
            "error_count": self._error_count,
            "last_cycle": self._history[-1] if self._history else None
        }

    def _validate_data_quality(self, candles: CandleArray, tf: str) -> Tuple[DataQuality, List[Tuple[str, float, float]]]:
        """Validates data integrity and detects gaps."""
        gaps = []
        
        if candles is None or len(candles) < 10:
            return DataQuality.INVALID, []

        for i in range(1, len(candles)):
            expected = candles.time[i-1] + (300 if tf == "M5" else 900 if tf == "M15" else 3600 if tf == "H1" else 86400)
            actual = candles.time[i]
            if actual - expected > 3600:
                gaps.append((tf, expected, actual))

        if len(candles) < 50:
            return DataQuality.INVALID, gaps
        elif len(candles) < 100:
            return DataQuality.GOOD, gaps
        elif gaps:
            return DataQuality.STALE, gaps
        return DataQuality.EXCELLENT, gaps

    def _run_loop(self):
        """Continuous background processing loop."""
        cycle_count = 0
        while not self._stop_event.is_set():
            try:
                cycle_start = time.time()
                for symbol in self.symbols:
                    self._update_symbol_state(symbol)
                
                cycle_count += 1
                cycle_time = time.time() - cycle_start
                self._history.append(cycle_time)

                if cycle_count % 30 == 0:
                    health = self.get_health()
                    logger.info(f"DataEngine health: {health}")

                time.sleep(self.update_interval)
            except Exception as e:
                logger.error(f"DataEngine Error: {e}", exc_info=True)
                self._error_count += 1
                time.sleep(5)

    def _update_symbol_state(self, symbol: str):
        """Fetches and processes all timeframes for a symbol."""
        try:
            now = datetime.now(timezone.utc)
            m5_candles = self.data_manager.prepare_data(symbol, "M5", start_date=now - timedelta(days=5))
            m15_candles = self.data_manager.prepare_data(symbol, "M15", start_date=now - timedelta(days=10))
            h1_candles = self.data_manager.prepare_data(symbol, "H1", start_date=now - timedelta(days=20))
            d1_candles = self.data_manager.prepare_data(symbol, "D1", start_date=now - timedelta(days=150))
            
            if len(m5_candles) == 0:
                return

            m5_quality, m5_gaps = self._validate_data_quality(m5_candles, "M5")
            m15_quality, m15_gaps = self._validate_data_quality(m15_candles, "M15")
            h1_quality, h1_gaps = self._validate_data_quality(h1_candles, "H1")
            
            all_gaps = m5_gaps + m15_gaps + h1_gaps
            quality = min(m5_quality, m15_quality, h1_quality, key=lambda x: x.value)

            m5_candles._indicators = IndicatorEngine.precalculate_all(symbol, "M5", m5_candles)
            m15_candles._indicators = IndicatorEngine.precalculate_all(symbol, "M15", m15_candles)
            h1_candles._indicators = IndicatorEngine.precalculate_all(symbol, "H1", h1_candles)
            d1_candles._indicators = IndicatorEngine.precalculate_all(symbol, "D1", d1_candles)
            
            new_state = MarketState(symbol)
            new_state.m5 = m5_candles
            new_state.m15 = m15_candles
            new_state.h1 = h1_candles
            new_state.d1 = d1_candles
            new_state.timestamp = m5_candles.time[-1]
            new_state.last_updated = time.time()
            new_state.quality = quality
            new_state.gaps = all_gaps
            new_state.bar_count = len(m5_candles)
            
            with self._lock:
                self.states[symbol] = new_state
                
        except Exception as e:
            logger.debug(f"Process update failed for {symbol}: {e}")
