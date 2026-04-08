import pytest
import time
from datetime import datetime, timezone
from core.news_filter import InstitutionalNewsFilter

@pytest.fixture
def mock_config():
    return {
        "news_filter": {
            "enabled": True,
            "impact_levels": ["High"],
            "buffer_before_min": 30,
            "buffer_after_min": 15,
            "auto_close_before_min": 5,
            "cache_file": "config/test_news_cache.json"
        }
    }

def test_news_blocking_logic(mock_config):
    news_filter = InstitutionalNewsFilter(mock_config)
    
    # Mock some events
    base_time = 1712600000 # Some fixed timestamp
    news_filter.events = [
        {
            "title": "US CPI",
            "country": "USD",
            "impact": "High",
            "timestamp": base_time
        },
        {
            "title": "JPY Interest Rate",
            "country": "JPY",
            "impact": "High",
            "timestamp": base_time + 3600 # 1 hour later
        }
    ]
    
    # Test symbols
    symbol_usd = "XAUUSDm"
    symbol_jpy = "GBPJPYm"
    symbol_eur = "EURGBP" # Not affected by USD or JPY in this mock
    
    # 1. T-15 minutes (Blocked)
    assert news_filter.is_blocked(symbol_usd, base_time - 15 * 60) == "US CPI"
    
    # 2. T+10 minutes (Blocked)
    assert news_filter.is_blocked(symbol_usd, base_time + 10 * 60) == "US CPI"
    
    # 3. T-45 minutes (Not Blocked)
    assert news_filter.is_blocked(symbol_usd, base_time - 45 * 60) is None
    
    # 4. T+30 minutes (Not Blocked, buffer_after is 15)
    assert news_filter.is_blocked(symbol_usd, base_time + 30 * 60) is None
    
    # 5. JPY news check
    assert news_filter.is_blocked(symbol_jpy, base_time + 3600) == "JPY Interest Rate"
    assert news_filter.is_blocked(symbol_usd, base_time + 3600) is None

def test_auto_close_logic(mock_config):
    news_filter = InstitutionalNewsFilter(mock_config)
    
    base_time = 1712600000
    news_filter.events = [
        {"title": "USD News", "country": "USD", "timestamp": base_time}
    ]
    
    # 4 minutes before news (Should trigger auto-close)
    targets = news_filter.get_auto_close_targets(base_time - 4 * 60)
    assert "USD" in targets
    
    # 10 minutes before news (Should NOT trigger auto-close yet)
    targets = news_filter.get_auto_close_targets(base_time - 10 * 60)
    assert "USD" not in targets
