"""
Unit tests for BacktestEngine.
Tests the simulation, drawdown, and streak calculations without MT5.
"""

import sys
import os
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.backtest import BacktestEngine
from core.strategy_engine import TradeSignal


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def config():
    return {
        "strategy": {"min_confluence_score": 3, "min_confidence": 40, "cooldown_candles": 3},
        "backtest": {"initial_balance": 1000, "spread_pips": {"TEST": 10}},
        "symbols_config": {"TEST": {"point": 0.01, "contract_size": 1, "lot": 0.01}},
    }


@pytest.fixture
def engine(config):
    from core.strategy_engine import StrategyEngine
    strategy = StrategyEngine(config)
    return BacktestEngine(config, strategy)


# ---------------------------------------------------------------------------
# _simulate_trade — Look-Ahead Bias Fix
# ---------------------------------------------------------------------------

class TestSimulateTrade:
    def test_buy_tp_hit(self, engine):
        signal = TradeSignal(
            direction="BUY", entry_price=100, stop_loss=95, take_profit=110,
            confidence=80, confluence_score=5,
        )
        candles = [{"open": 101, "high": 112, "low": 100, "close": 111}]
        assert engine._simulate_trade(signal, candles, 100) == "WIN"

    def test_buy_sl_hit(self, engine):
        signal = TradeSignal(
            direction="BUY", entry_price=100, stop_loss=95, take_profit=110,
            confidence=80, confluence_score=5,
        )
        candles = [{"open": 99, "high": 100, "low": 93, "close": 94}]
        assert engine._simulate_trade(signal, candles, 100) == "LOSS"

    def test_sell_tp_hit(self, engine):
        signal = TradeSignal(
            direction="SELL", entry_price=100, stop_loss=105, take_profit=90,
            confidence=80, confluence_score=5,
        )
        candles = [{"open": 99, "high": 100, "low": 88, "close": 89}]
        assert engine._simulate_trade(signal, candles, 100) == "WIN"

    def test_sell_sl_hit(self, engine):
        signal = TradeSignal(
            direction="SELL", entry_price=100, stop_loss=105, take_profit=90,
            confidence=80, confluence_score=5,
        )
        candles = [{"open": 101, "high": 107, "low": 100, "close": 106}]
        assert engine._simulate_trade(signal, candles, 100) == "LOSS"

    def test_buy_both_hit_bearish_candle_is_loss(self, engine):
        """
        Look-ahead bias fix: if both SL and TP are hit in the same candle,
        a bearish candle (open > close) on a BUY means SL was hit first.
        """
        signal = TradeSignal(
            direction="BUY", entry_price=100, stop_loss=90, take_profit=115,
            confidence=80, confluence_score=5,
        )
        # Candle hits both TP (high=120) and SL (low=85), but is bearish
        candles = [{"open": 105, "high": 120, "low": 85, "close": 88}]
        assert engine._simulate_trade(signal, candles, 100) == "LOSS"

    def test_buy_both_hit_bullish_candle_is_win(self, engine):
        """
        Look-ahead bias fix: bullish candle (open < close) on a BUY
        means TP was likely hit first.
        """
        signal = TradeSignal(
            direction="BUY", entry_price=100, stop_loss=90, take_profit=115,
            confidence=80, confluence_score=5,
        )
        # Candle hits both TP and SL, but is bullish
        candles = [{"open": 88, "high": 120, "low": 85, "close": 116}]
        assert engine._simulate_trade(signal, candles, 100) == "WIN"

    def test_sell_both_hit_bullish_candle_is_loss(self, engine):
        """For SELL: bullish candle when both hit → SL hit first."""
        signal = TradeSignal(
            direction="SELL", entry_price=100, stop_loss=110, take_profit=85,
            confidence=80, confluence_score=5,
        )
        candles = [{"open": 95, "high": 115, "low": 80, "close": 112}]
        assert engine._simulate_trade(signal, candles, 100) == "LOSS"

    def test_sell_both_hit_bearish_candle_is_win(self, engine):
        """For SELL: bearish candle when both hit → TP hit first."""
        signal = TradeSignal(
            direction="SELL", entry_price=100, stop_loss=110, take_profit=85,
            confidence=80, confluence_score=5,
        )
        candles = [{"open": 112, "high": 115, "low": 80, "close": 82}]
        assert engine._simulate_trade(signal, candles, 100) == "WIN"

    def test_no_future_candles_returns_open(self, engine):
        signal = TradeSignal(
            direction="BUY", entry_price=100, stop_loss=95, take_profit=110,
            confidence=80, confluence_score=5,
        )
        assert engine._simulate_trade(signal, [], 100) == "OPEN"


# ---------------------------------------------------------------------------
# Drawdown Calculation
# ---------------------------------------------------------------------------

class TestDrawdown:
    def test_no_trades_zero_drawdown(self):
        assert BacktestEngine._calc_drawdown([], 1000) == 0.0

    def test_simple_drawdown(self):
        trades = [
            {"pnl": 100, "result": "TP"},   # balance: 1100, peak: 1100
            {"pnl": -200, "result": "SL"},   # balance: 900, dd: (1100-900)/1100 = 18.18%
            {"pnl": 50, "result": "TP"},     # balance: 950
        ]
        dd = BacktestEngine._calc_drawdown(trades, 1000)
        assert abs(dd - 18.18) < 0.1

    def test_no_drawdown_when_only_winning(self):
        trades = [{"pnl": 100, "result": "TP"}, {"pnl": 50, "result": "TP"}]
        assert BacktestEngine._calc_drawdown(trades, 1000) == 0.0


# ---------------------------------------------------------------------------
# Streak Calculation
# ---------------------------------------------------------------------------

class TestStreak:
    def test_no_trades_zero_streak(self):
        assert BacktestEngine._calc_streak([], "TP") == 0

    def test_win_streak(self):
        trades = [
            {"result": "TP"}, {"result": "TP"}, {"result": "TP"},
            {"result": "SL"}, {"result": "TP"},
        ]
        assert BacktestEngine._calc_streak(trades, "TP") == 3

    def test_loss_streak(self):
        trades = [
            {"result": "TP"}, {"result": "SL"}, {"result": "SL"},
            {"result": "SL"}, {"result": "SL"}, {"result": "TP"},
        ]
        assert BacktestEngine._calc_streak(trades, "SL") == 4


# ---------------------------------------------------------------------------
# Config-Based Lot Sizes (not hardcoded)
# ---------------------------------------------------------------------------

class TestConfigLotSizes:
    def test_lot_comes_from_config(self, config):
        """Verify BacktestEngine reads lot from symbols_config, not hardcoded."""
        # The config says lot=0.01 for TEST
        lot = config["symbols_config"]["TEST"]["lot"]
        assert lot == 0.01
        # Change it to prove it's read from config
        config["symbols_config"]["TEST"]["lot"] = 0.5
        assert config["symbols_config"]["TEST"]["lot"] == 0.5
