import pytest
import numpy as np
from datetime import datetime, timezone
from core import CandleArray, MarketData

@pytest.fixture
def mock_config():
    """Standard Institutional Bot Configuration."""
    return {
        "symbol": "XAUUSDm",
        "risk_governance": {
            "risk_per_trade_pct": 1.0,
            "max_daily_loss_pct": 3.0,
            "max_drawdown_halt_pct": 10.0,
            "max_parallel_strategies": 4
        },
        "backtest": {
            "initial_balance": 10000.0,
            "initial_balance_per_strategy": 5000.0,
            "deterministic": True
        },
        "symbols_config": {
            "XAUUSDm": {
                "point": 0.01,
                "tick_value": 1.0,
                "lot_step": 0.01,
                "min_lot": 0.01,
                "max_lot": 50.0,
                "spread_pips": 15,
                "commission_per_lot": 7.0
            }
        }
    }

@pytest.fixture
def candle_factory():
    """Factory to create synthetic CandleArrays for testing."""
    def _create(n=100, trend="FLAT", volatility=1.0, base_price=2000.0):
        t = (np.arange(n) * 300 + 1700000000).astype(np.int64)
        
        if trend == "BULLISH":
            close = base_price + np.linspace(0, 10, n)
        elif trend == "BEARISH":
            close = base_price - np.linspace(0, 10, n)
        else:
            close = base_price + np.random.normal(0, volatility, n)
            
        return CandleArray(
            time=t,
            open=close - 0.5 * volatility,
            high=close + 1.5 * volatility,
            low=close - 1.5 * volatility,
            close=close,
            tick_volume=np.random.randint(100, 500, n),
            spread=np.full(n, 15)
        )
    return _create

@pytest.fixture
def market_data_factory(candle_factory):
    """Factory to create MarketData snapshots for strategy testing."""
    def _create(trend="BULLISH", session="LONDON"):
        m5 = candle_factory(n=260, trend=trend)
        h1 = candle_factory(n=80, trend=trend)
        m15 = candle_factory(n=120, trend=trend)
        
        current_price = float(m5.close[-1])
        
        return MarketData(
            symbol="XAUUSDm",
            htf_candles=h1,
            m15_candles=m15,
            m5_candles=m5,
            d1_candles=None,
            current_price=current_price,
            bid=current_price,
            ask=current_price + 1.0,
            spread=1.0,
            session=session,
            timestamp=datetime.now(timezone.utc),
            preprocessed={
                "m_bias": trend,
                "m_high": float(m5.high[-2]),
                "m_low": float(m5.low[-2]),
                "sweep_bull": False,
                "sweep_bear": False,
                "in_htf_demand": trend == "BULLISH",
                "in_htf_supply": trend == "BEARISH"
            }
        )
    return _create

@pytest.fixture
def symbol_info():
    """Standard symbol properties for risk and sizing tests."""
    return {
        "point": 0.01,
        "tick_value": 1.0,
        "min_lot": 0.01,
        "max_lot": 50.0,
        "lot_step": 0.01,
        "spread_pips": 15,
        "commission_per_lot": 7.0
    }
