import pytest
import time
from unittest.mock import patch, MagicMock
from core.data_handler import DataFetcher
from core import CandleArray

class TestDataFetcher:
    """Verifies the reliability of the Data Acquisition Layer."""

    @pytest.fixture
    def fetcher(self, mock_config):
        return DataFetcher()

    @patch("core.data_handler.mt5")
    def test_fetch_from_mt5_success(self, mock_mt5, fetcher):
        """Verify successful candle fetching from MT5 backend."""
        mock_mt5.symbol_select.return_value = True
        mock_mt5.copy_rates_from_pos.return_value = [
            {"time": 1000, "open": 2000.0, "high": 2010.0, "low": 1990.0, "close": 2005.0, "tick_volume": 100, "spread": 5},
            {"time": 1300, "open": 2005.0, "high": 2015.0, "low": 1995.0, "close": 2010.0, "tick_volume": 120, "spread": 6}
        ]
        
        with patch.dict("core.data_handler.TIMEFRAME_MAP", {"M5": 1}, clear=False):
            candles = fetcher.fetch_candles("XAUUSDm", "M5", count=2)
            
            assert isinstance(candles, CandleArray)
            assert len(candles) == 2
            assert float(candles.close[-1]) == 2010.0

    @patch("core.data_handler.mt5")
    def test_cache_hit(self, mock_mt5, fetcher):
        """Ensure repeated requests for same data return cached objects."""
        # 1. Setup cache manually
        fake_candles = MagicMock(spec=CandleArray)
        fetcher._cache["XAUUSDm_M5"] = {
            "timestamp": time.time(),
            "data": [{"time": 1000}], # Minimal mock data
            "array": fake_candles
        }
        
        # 2. Fetch data (Should NOT call MT5)
        with patch.dict("core.data_handler.TIMEFRAME_MAP", {"M5": 1}, clear=False):
            candles = fetcher.fetch_candles("XAUUSDm", "M5", count=1)
            
            assert candles is fake_candles
            assert mock_mt5.copy_rates_from_pos.call_count == 0

    @patch("core.data_handler.mt5")
    def test_incremental_merging(self, mock_mt5, fetcher):
        """Verify that new MT5 data is correctly appended to existing cache."""
        # 1. Setup cache with old data
        fetcher._cache["XAUUSDm_M5"] = {
            "timestamp": time.time() - 1000,
            "data": [
                {"time": 700, "open": 1995.0, "high": 2005.0, "low": 1985.0, "close": 2000.0, "tick_volume": 80, "spread": 4},
                {"time": 1000, "open": 2000.0, "high": 2010.0, "low": 1990.0, "close": 2005.0, "tick_volume": 100, "spread": 5}
            ],
            "array": CandleArray.from_dicts([
                {"time": 700, "open": 1995.0, "high": 2005.0, "low": 1985.0, "close": 2000.0, "tick_volume": 80, "spread": 4},
                {"time": 1000, "open": 2000.0, "high": 2010.0, "low": 1990.0, "close": 2005.0, "tick_volume": 100, "spread": 5}
            ])
        }
        
        # 2. MT5 returns newer data
        mock_mt5.symbol_select.return_value = True
        mock_mt5.copy_rates_from_pos.return_value = [
            {"time": 1300, "open": 2005.0, "high": 2015.0, "low": 1995.0, "close": 2010.0, "tick_volume": 120, "spread": 6}
        ]
        
        with patch.dict("core.data_handler.TIMEFRAME_MAP", {"M5": 1}, clear=False):
            candles = fetcher.fetch_candles("XAUUSDm", "M5", count=2)
            
            assert len(candles) == 2
            assert int(candles.time[0]) == 1000
            assert int(candles.time[1]) == 1300
