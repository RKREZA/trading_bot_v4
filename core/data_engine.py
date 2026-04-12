import logging
import time
import threading
from typing import Dict, Any, Optional
from datetime import datetime, timezone

from core.connection import MT5Connection
from core.data.manager import DataManager
from core.indicator_engine import IndicatorEngine
from core.common.types import CandleArray

logger = logging.getLogger("trading_bot.data_engine")

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

class DataEngine:
    """
    V5-INSIGNIA Asynchronous Data & Indicator Engine (Institutional Pillar 3).
    Runs on a dedicated background thread to fetch candles and calculate 
    complex technical indicators (ADX, ATR, RSI, EMAs) non-blockingly.
    """
    
    def __init__(self, connection: MT5Connection, config: dict):
        self.connection = connection
        self.config = config
        self.data_manager = DataManager(config)
        self.states: Dict[str, MarketState] = {}
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        
        # Throttling to prevent MT5 lock contention
        self.update_interval = config.get("performance", {}).get("data_engine_interval", 1.0)
        self.symbols = list(config.get("symbols_config", {}).keys())
        if "XAUUSDm" not in self.symbols:
            self.symbols.append("XAUUSDm")

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

    def _run_loop(self):
        """Continuous background processing loop."""
        while not self._stop_event.is_set():
            try:
                for symbol in self.symbols:
                    self._update_symbol_state(symbol)
                
                # Sleep between symbol cycles to allow main execution thread access to MT5
                time.sleep(self.update_interval)
            except Exception as e:
                logger.error(f"DataEngine Error: {e}", exc_info=True)
                time.sleep(5)

    def _update_symbol_state(self, symbol: str):
        """Fetches and processes all timeframes for a symbol."""
        from datetime import timedelta, datetime, timezone
        try:
            # 1. Fetch Multi-Timeframe Candles (Non-blocking because of lock sharing)
            # Use a slightly larger lookback than the strategy requirement to ensure indicators are valid
            # We multiply required bars by a generous safety factor (days) to account for weekends/holidays.
            # M5 (300 bars = ~1.04 trading days) -> 5 days
            # M15 (300 bars = ~3.12 trading days) -> 10 days
            # H1 (300 bars = ~12.5 trading days) -> 20 days
            # D1 (100 bars = 100 trading days) -> 150 days
            now = datetime.now(timezone.utc)
            m5_candles = self.data_manager.prepare_data(symbol, "M5", start_date=now - timedelta(days=5))
            m15_candles = self.data_manager.prepare_data(symbol, "M15", start_date=now - timedelta(days=10))
            h1_candles = self.data_manager.prepare_data(symbol, "H1", start_date=now - timedelta(days=20))
            d1_candles = self.data_manager.prepare_data(symbol, "D1", start_date=now - timedelta(days=150))
            
            if len(m5_candles) == 0:
                return

            # 2. Parallel Indicator Calculation (CPU Heavy)
            # We do this calculation OUTSIDE the connection lock where possible 
            # (IndicatorEngine.precalculate_all works on local CandleArray/DataFrame objects)
            m5_candles._indicators = IndicatorEngine.precalculate_all(symbol, "M5", m5_candles)
            m15_candles._indicators = IndicatorEngine.precalculate_all(symbol, "M15", m15_candles)
            h1_candles._indicators = IndicatorEngine.precalculate_all(symbol, "H1", h1_candles)
            d1_candles._indicators = IndicatorEngine.precalculate_all(symbol, "D1", d1_candles)
            
            # 3. Create & Store State Snapshot
            new_state = MarketState(symbol)
            new_state.m5 = m5_candles
            new_state.m15 = m15_candles
            new_state.h1 = h1_candles
            new_state.d1 = d1_candles
            new_state.timestamp = m5_candles.time[-1]
            new_state.last_updated = time.time()
            
            with self._lock:
                self.states[symbol] = new_state
                
        except Exception as e:
            logger.debug(f"Process update failed for {symbol}: {e}")
