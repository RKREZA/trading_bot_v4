import numpy as np
import pytest

from core.base_strategy import MarketData
from core.types import CandleArray, TradeSignal
from strategies import create_strategy


def _arr(n, step_sec, base=100.0, slope=0.02):
    t = (np.arange(n) * step_sec + 1700000000).astype(np.int64)
    c = base + np.arange(n) * slope
    return CandleArray(
        time=t,
        open=c - 0.02,
        high=c + 0.10,
        low=c - 0.10,
        close=c,
        tick_volume=np.full(n, 120),
    )


def _market_data():
    m5 = _arr(260, 300)
    h1 = _arr(80, 3600)
    m15 = _arr(120, 900)
    return MarketData(
        symbol="XAUUSDm",
        htf_candles=h1,
        m15_candles=m15,
        m5_candles=m5,
        d1_candles=None,
        current_price=float(m5.close[-1]),
        session="LONDON",
        timestamp=None,
        preprocessed={
            "m_bias": "BULLISH",
            "m_high": float(m5.high[-2]),
            "m_low": float(m5.low[-2]),
            "sweep_bull": False,
            "sweep_bear": False,
            "in_htf_demand": True,
            "in_htf_supply": False,
        },
    )


@pytest.mark.parametrize("sid", ["TREND_FOLLOWING", "MEAN_REVERSION", "BREAKOUT", "LIQUIDITY_SESSION"])
def test_strategy_generate_signal_no_crash(sid):
    cfg = {"params": {}, "enabled": True, "strategy_defaults": {"min_confidence": 0.55}}
    s = create_strategy(sid, cfg)
    md = _market_data()

    sig = s.generate_signal(md)
    assert sig is None or isinstance(sig, TradeSignal)
    if isinstance(sig, TradeSignal):
        assert sig.direction in {"BUY", "SELL", "NONE"}


def test_unknown_strategy_raises():
    with pytest.raises(ValueError):
        create_strategy("DOES_NOT_EXIST", {})
