"""Tests for unified Pydantic types and CandleArray."""

import numpy as np
import pytest
from core.common.types import (
    CandleArray,
    TradeSignal,
    ExecutionIntent,
    MarketSnapshot,
    ExecutionOutcome,
    OrderState,
)


class TestCandleArray:
    def test_from_arrays(self, candle_factory):
        candles = candle_factory(n=50, trend="BULLISH")
        assert len(candles) == 50
        assert candles.time[0] < candles.time[-1]

    def test_set_limit_anti_lookahead(self, candle_factory):
        candles = candle_factory(n=100)
        candles.set_limit(50)
        assert len(candles.close[:candles._limit]) == 50

    def test_slicing(self, candle_factory):
        candles = candle_factory(n=100)
        sliced = candles[20:40]
        assert len(sliced) == 20

    def test_empty_array(self):
        candles = CandleArray.from_dicts([])
        assert len(candles) == 0


class TestTradeSignal:
    def test_creation(self):
        sig = TradeSignal(
            direction="BUY",
            price=2000.0,
            confidence=0.85,
            stop_loss=1990.0,
            take_profit=2020.0,
        )
        assert sig.direction == "BUY"
        assert sig.confidence == 0.85

    def test_none_direction(self):
        sig = TradeSignal(direction="NONE", price=0.0, confidence=0.0)
        assert sig.direction == "NONE"


class TestExecutionIntent:
    def test_intent_hash_deterministic(self):
        intent1 = ExecutionIntent(
            symbol="XAUUSDm",
            direction="BUY",
            volume=0.1,
            stop_loss=1990.0,
            take_profit=2020.0,
            strategy_id="smc_v1",
            setup_timestamp=1700000000,
        )
        intent2 = ExecutionIntent(
            symbol="XAUUSDm",
            direction="BUY",
            volume=0.1,
            stop_loss=1990.0,
            take_profit=2020.0,
            strategy_id="smc_v1",
            setup_timestamp=1700000000,
        )
        assert intent1.intent_hash == intent2.intent_hash

    def test_different_intents_different_hash(self):
        intent1 = ExecutionIntent(
            symbol="XAUUSDm", direction="BUY", volume=0.1,
            stop_loss=1990.0, take_profit=2020.0, strategy_id="smc_v1",
            setup_timestamp=1700000000,
        )
        intent2 = ExecutionIntent(
            symbol="EURUSD", direction="SELL", volume=0.2,
            stop_loss=1.0900, take_profit=1.0800, strategy_id="trend_v1",
            setup_timestamp=1700000001,
        )
        assert intent1.intent_hash != intent2.intent_hash


class TestOrderState:
    def test_enum_values(self):
        assert OrderState.PENDING.value == "PENDING"
        assert OrderState.FILLED.value == "FILLED"
        assert OrderState.REJECTED.value == "REJECTED"
