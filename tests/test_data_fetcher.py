import time
from unittest.mock import patch

import pytest

from core.data_handler import DataFetcher
from core.types import CandleArray


@pytest.fixture
def mock_mt5():
    with patch("core.data_handler.mt5") as mock:
        yield mock


@pytest.fixture
def data_fetcher():
    return DataFetcher()


def test_cache_miss_fetches_from_mt5(data_fetcher, mock_mt5):
    mock_mt5.TIMEFRAME_H1 = 1
    mock_mt5.symbol_select.return_value = True
    rates = [
        {"time": 1000, "open": 1.0, "high": 2.0, "low": 0.5, "close": 1.5, "tick_volume": 10},
        {"time": 2000, "open": 1.5, "high": 2.5, "low": 1.0, "close": 2.0, "tick_volume": 20},
    ]
    mock_mt5.copy_rates_from_pos.return_value = rates

    with patch.dict("core.data_handler.TIMEFRAME_MAP", {"H1": 1}, clear=False):
        candles = data_fetcher.fetch_candles("XAUUSDm", "H1", count=2)

    assert isinstance(candles, CandleArray)
    assert len(candles) == 2
    assert float(candles.close[-1]) == 2.0


def test_cache_hit_returns_cached_array(data_fetcher, mock_mt5):
    fake = CandleArray.from_dicts([
        {"time": 1000, "open": 1, "high": 2, "low": 0.5, "close": 1.5, "tick_volume": 10}
    ])
    data_fetcher._cache["XAUUSDm_H1"] = {
        "timestamp": time.time(),
        "data": [{"time": 1000, "open": 1, "high": 2, "low": 0.5, "close": 1.5, "tick_volume": 10}],
        "array": fake,
    }

    with patch.dict("core.data_handler.TIMEFRAME_MAP", {"H1": 1}, clear=False):
        candles = data_fetcher.fetch_candles("XAUUSDm", "H1", count=1)

    assert candles is fake
    assert mock_mt5.copy_rates_from_pos.call_count == 0


def test_incremental_fetch_merges_data(data_fetcher, mock_mt5):
    data_fetcher._cache["XAUUSDm_H1"] = {
        "timestamp": time.time() - 1000,
        "data": [
            {"time": 1000, "open": 1.0, "high": 2.0, "low": 0.5, "close": 1.5, "tick_volume": 10},
            {"time": 2000, "open": 1.5, "high": 2.5, "low": 1.0, "close": 2.0, "tick_volume": 20},
        ],
        "array": CandleArray.from_dicts([
            {"time": 1000, "open": 1.0, "high": 2.0, "low": 0.5, "close": 1.5, "tick_volume": 10},
            {"time": 2000, "open": 1.5, "high": 2.5, "low": 1.0, "close": 2.0, "tick_volume": 20},
        ]),
    }

    mock_mt5.symbol_select.return_value = True
    mock_mt5.copy_rates_from_pos.return_value = [
        {"time": 3000, "open": 2.0, "high": 3.0, "low": 1.5, "close": 2.5, "tick_volume": 30}
    ]

    with patch.dict("core.data_handler.TIMEFRAME_MAP", {"H1": 1}, clear=False):
        candles = data_fetcher.fetch_candles("XAUUSDm", "H1", count=2)

    assert len(candles) == 2
    assert int(candles.time[0]) == 2000
    assert float(candles.close[-1]) == 2.5
