"""Backtesting integration tests for V4 PortfolioBacktester."""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtesting.backtester import PortfolioBacktester
from core.base_strategy import BaseStrategy
from core.types import CandleArray, TradeSignal
from core.risk_engine import RiskEngine
from strategies import create_strategy


class DeterministicPulseStrategy(BaseStrategy):
    def generate_signal(self, market_data):
        if len(market_data.m5_candles) % 25 == 0:
            return TradeSignal(
                direction="BUY",
                price=float(market_data.current_price),
                confidence=0.95,
                timestamp=market_data.timestamp,
            )
        return TradeSignal(direction="NONE")

    def get_stop_loss(self, signal, market_data):
        return float(market_data.current_price) - 0.20

    def get_take_profit(self, signal, market_data):
        return float(market_data.current_price) + 0.20


def _build_arrays(n_m5: int = 260):
    base_ts = 1700000000

    # M5 trend
    t5 = (np.arange(n_m5) * 300 + base_ts).astype(np.int64)
    close5 = 100.0 + np.linspace(0, 2.0, n_m5)
    open5 = close5 - 0.02
    high5 = close5 + 0.10
    low5 = close5 - 0.10
    vol5 = np.full(n_m5, 150)
    m5 = CandleArray(time=t5, open=open5, high=high5, low=low5, close=close5, tick_volume=vol5)

    # M1 for intrabar execution (TP reachable, SL not touched)
    n_m1 = n_m5 * 5
    t1 = (np.arange(n_m1) * 60 + base_ts).astype(np.int64)
    close1 = 100.0 + np.linspace(0, 2.0, n_m1)
    open1 = close1 - 0.01
    high1 = close1 + 0.35
    low1 = close1 - 0.05
    vol1 = np.full(n_m1, 120)
    m1 = CandleArray(time=t1, open=open1, high=high1, low=low1, close=close1, tick_volume=vol1)

    # HTF arrays
    n_m15 = (n_m5 // 3) + 20
    t15 = (np.arange(n_m15) * 900 + base_ts).astype(np.int64)
    close15 = 100.0 + np.linspace(0, 2.0, n_m15)
    m15 = CandleArray(
        time=t15,
        open=close15 - 0.02,
        high=close15 + 0.12,
        low=close15 - 0.12,
        close=close15,
        tick_volume=np.full(n_m15, 100),
    )

    n_h1 = (n_m5 // 12) + 30
    t1h = (np.arange(n_h1) * 3600 + base_ts).astype(np.int64)
    closeh = 100.0 + np.linspace(0, 2.0, n_h1)
    h1 = CandleArray(
        time=t1h,
        open=closeh - 0.02,
        high=closeh + 0.15,
        low=closeh - 0.15,
        close=closeh,
        tick_volume=np.full(n_h1, 100),
    )

    return m5, h1, m15, m1


def _config():
    return {
        "backtest": {
            "initial_balance": 1000,
            "deterministic": True,
            "random_seed": 7,
            "costs": {
                "entry_commission_sides": 1.0,
                "exit_commission_sides": 1.0,
            },
        },
        "risk_governance": {
            "risk_per_trade_pct": 0.5,
            "max_daily_loss_pct": 3.0,
            "max_drawdown_halt_pct": 10.0,
        },
        "symbols_config": {
            "XAUUSDm": {
                "point": 0.01,
                "tick_value": 1.0,
                "commission_per_lot": 0.5,
                "spread_pips": 2.0,
            }
        },
        "execution": {
            "entry_slippage_pips": 0.2,
            "tp_exit_slippage_pips": 0.1,
            "sl_exit_slippage_pips": 0.25,
            "forced_exit_slippage_pips": 0.2,
            "max_spread_pips": 10.0,
        },
    }


def test_backtester_runs_and_produces_history():
    cfg = _config()
    m5, h1, m15, m1 = _build_arrays()
    bt = PortfolioBacktester(cfg)
    strategies = [DeterministicPulseStrategy("pulse_v1", cfg)]

    history = bt.run("XAUUSDm", strategies, m5, h1, m15, m1)

    assert len(history) > 0
    assert all("strategy_id" in t for t in history)
    assert any(t["result"] in {"TP", "FORCED_CLOSE"} for t in history)


def test_backtester_resets_state_between_runs():
    cfg = _config()
    m5, h1, m15, m1 = _build_arrays()
    bt = PortfolioBacktester(cfg)
    strategies = [DeterministicPulseStrategy("pulse_v1", cfg)]

    history_1 = bt.run("XAUUSDm", strategies, m5, h1, m15, m1)
    history_2 = bt.run("XAUUSDm", strategies, m5, h1, m15, m1)

    assert len(history_1) == len(history_2)
    assert history_1[0]["timestamp"] == history_2[0]["timestamp"]


def test_backtester_deterministic_seed_is_reproducible():
    cfg = _config()
    m5, h1, m15, m1 = _build_arrays()

    bt_a = PortfolioBacktester(cfg)
    bt_b = PortfolioBacktester(cfg)
    strategies_a = [DeterministicPulseStrategy("pulse_v1", cfg)]
    strategies_b = [DeterministicPulseStrategy("pulse_v1", cfg)]

    history_a = bt_a.run("XAUUSDm", strategies_a, m5, h1, m15, m1)
    history_b = bt_b.run("XAUUSDm", strategies_b, m5, h1, m15, m1)

    assert len(history_a) == len(history_b)
    assert [round(t["pnl"], 8) for t in history_a] == [round(t["pnl"], 8) for t in history_b]
    assert [round(t.get("entry_slippage_pips", 0.0), 8) for t in history_a] == [
        round(t.get("entry_slippage_pips", 0.0), 8) for t in history_b
    ]


def test_risk_engine_reads_risk_governance_block():
    cfg = _config()
    risk = RiskEngine(cfg)

    assert risk.risk_per_trade_pct == 0.5
    assert risk.max_daily_loss_pct == 3.0
    assert risk.max_total_drawdown_pct == 10.0


def test_trend_strategy_receives_enough_h1_history():
    cfg = _config()
    m5, h1, m15, m1 = _build_arrays()
    bt = PortfolioBacktester(cfg)
    trend = create_strategy("TREND_FOLLOWING", cfg)

    history = bt.run("XAUUSDm", [trend], m5, h1, m15, m1)

    assert isinstance(history, list)
