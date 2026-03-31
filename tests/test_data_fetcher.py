import pytest
import time
from unittest.mock import patch, MagicMock
from core.data_fetcher import DataFetcher
from core.types import CandleArray

@pytest.fixture
def mock_mt5():
    with patch("core.data_fetcher.mt5") as mock:
        yield mock

@pytest.fixture
def data_fetcher():
    return DataFetcher()

def test_cache_miss_fetches_from_mt5(data_fetcher, mock_mt5):
    mock_mt5.symbol_select.return_value = True
    # Return 2 records
    mock_mt5.copy_rates_from_pos.return_value = (
        (1672531200, 1800.0, 1805.0, 1795.0, 1802.0, 100, 0, 0),
        (1672534800, 1802.0, 1810.0, 1800.0, 1808.0, 150, 0, 0)
    )
    
    # Needs to match the DataFrame structure
    import pandas as pd
    df_mock = pd.DataFrame([
        {'time': 1672531200, 'open': 1800.0, 'high': 1805.0, 'low': 1795.0, 'close': 1802.0, 'tick_volume': 100},
        {'time': 1672534800, 'open': 1802.0, 'high': 1810.0, 'low': 1800.0, 'close': 1808.0, 'tick_volume': 150}
    ])
    
    with patch("pandas.DataFrame", return_value=df_mock):
        candles = data_fetcher.fetch_candles("XAUUSDm", "H1", count=2)
        
        assert isinstance(candles, CandleArray)
        assert len(candles) == 2
        assert candles.close[0] == 1802.0
        assert mock_mt5.copy_rates_from_pos.call_count == 1

def test_cache_hit_returns_cached_array(data_fetcher, mock_mt5):
    # Pre-populate cache
    fake_array = CandleArray.from_dicts([{'time': 1000, 'open': 1, 'high': 2, 'low': 0.5, 'close': 1.5, 'tick_volume': 10}])
    data_fetcher._cache["XAUUSDm_H1"] = {
        "timestamp": time.time(),
        "data": [{'time': 1000, 'open': 1, 'high': 2, 'low': 0.5, 'close': 1.5, 'tick_volume': 10}],
        "array": fake_array
    }
    
    candles = data_fetcher.fetch_candles("XAUUSDm", "H1", count=1)
    
    assert candles is fake_array
    assert mock_mt5.copy_rates_from_pos.call_count == 0

def test_incremental_fetch_merges_data(data_fetcher, mock_mt5):
    # Base cache (2 candles to satisfy count=2)
    data_fetcher._cache["XAUUSDm_H1"] = {
        "timestamp": time.time() - 3600, # Expired TTL
        "data": [
            {'time': 500, 'open': 1.0, 'high': 2.0, 'low': 0.5, 'close': 1.5, 'tick_volume': 10},
            {'time': 1000, 'open': 1.0, 'high': 2.0, 'low': 0.5, 'close': 1.5, 'tick_volume': 10}
        ],
        "array": None
    }

    mock_mt5.symbol_select.return_value = True
    mock_mt5.copy_rates_from_pos.return_value = [
        (2000, 1.5, 3.0, 1.0, 2.5, 20, 0, 0)
    ]

    import pandas as pd
    df_mock = pd.DataFrame([
        {'time': 2000, 'open': 1.5, 'high': 3.0, 'low': 1.0, 'close': 2.5, 'tick_volume': 20}
    ])

    with patch("pandas.DataFrame", return_value=df_mock):
        # Even if we request 2, if cache has 1, the code only fetches 10 for incremental
        candles = data_fetcher.fetch_candles("XAUUSDm", "H1", count=2)
        
        # Original 1000 + New 2000 = 2 candles
        assert len(candles) == 2
        assert candles.close[-1] == 2.5
        assert candles.time[0] == 1000
