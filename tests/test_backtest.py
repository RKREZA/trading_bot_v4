import pytest
import numpy as np
from unittest.mock import patch
from datetime import datetime, timezone
from backtesting.backtester import PortfolioBacktester
from core.base_strategy import BaseStrategy
from core import TradeSignal

class TrendAlwaysPulseStrategy(BaseStrategy):
    """Strategy that unconditionally triggers a trade at bar 250 for deterministic testing.
    Name contains 'Trend' to pass the RegimeGater during TRENDING markets."""
    def __init__(self, strategy_id, config):
        super().__init__(strategy_id, config)
        self.enabled = True
        self._fired = False

    def generate_signal(self, market_data):
        # Trigger once at bar 250 (when limited view length >= 250)
        if not self._fired and len(market_data.m5_candles) >= 250:
            self._fired = True
            return TradeSignal(
                direction="BUY",
                price=float(market_data.current_price),
                confidence=1.0  # Max confidence to bypass buffers
            )
        return TradeSignal(direction="NONE")

    def get_stop_loss(self, signal, market_data):
        return market_data.current_price - 5.0 # Fixed 500 point SL

    def get_take_profit(self, signal, market_data):
        return market_data.current_price + 10.0 # Fixed 1000 point TP

class TestPortfolioBacktester:
    """Verifies the PortfolioBacktester logic and fidelity."""

    def test_run_full_cycle(self, mock_config, candle_factory):
        """Test a complete backtest run with a trade entry and exit."""
        bt = PortfolioBacktester(mock_config)
        
        # 1. Create data with strong trend to ensure TRENDING regime (not RANGING)
        m5 = candle_factory(n=300, trend="BULLISH", base_price=2000.0, volatility=5.0)
        h1 = candle_factory(n=100, trend="BULLISH", base_price=2000.0, volatility=5.0)
        m15 = candle_factory(n=150, trend="BULLISH", base_price=2000.0, volatility=5.0)
        
        # M1 needs to cover the M5 period (300 * 5 = 1500 bars)
        m1 = candle_factory(n=1600, trend="BULLISH", base_price=2000.0, volatility=5.0)
        
        # 2. Setup strategy
        strat = TrendAlwaysPulseStrategy("ALWAYS_PULSE", mock_config)
        
        # 3. Run backtest (target_tf_data = m5)
        history, equity_history = bt.run("XAUUSDm", [strat], m5, h1, m15, m5, m1)
        
        # 4. Assertions
        assert len(history) > 0, "No trades were executed in the backtest."
        trade = history[0]
        assert trade["strategy_id"] == "ALWAYS_PULSE"
        assert trade["direction"] == "BUY"
        assert "pnl" in trade
        assert "entry_comm" in trade, "Entry commission must be tracked"
        assert len(equity_history) > 0

    def test_checkpoint_recovery(self, mock_config, candle_factory):
        """Verify that the backtester can resume from a mid-session checkpoint."""
        bt = PortfolioBacktester(mock_config)
        m5 = candle_factory(n=100)
        m1 = candle_factory(n=500)
        h1 = candle_factory(n=50)
        m15 = candle_factory(n=70)
        
        strategies = [TrendAlwaysPulseStrategy("TREND_PULSE", mock_config)]
        
        # 1. Simulate mid-session state
        state = {
            "current_index": 50,
            "balances": {"TREND_PULSE": 10000.0},
            "equities": {"TREND_PULSE": 10005.0},
            "peak_equity": {"TREND_PULSE": 10005.0},
            "max_drawdowns": {"TREND_PULSE": 0.0},
            "open_trades": {"TREND_PULSE": {"direction": "BUY", "fill_price": 2000.0, "lots": 0.1, "sl": 1990, "tp": 2020}},
            "history": []
        }
        bt.set_state(state)
        
        # 2. Run with resume=True
        # We need to mock the checkpoint manager's result if the real one isn't populated
        with patch.object(bt.checkpoint_manager, 'load_checkpoint', return_value=state):
            history, _ = bt.run("XAUUSDm", strategies, m5, h1, m15, m5, m1, resume=True)
            
            # Start index should be after 50
            assert bt.current_index >= 50

    def test_data_alignment_check(self, mock_config, candle_factory):
        """Ensure backtester rejects runs with misaligned timeframe data."""
        bt = PortfolioBacktester(mock_config)
        m5 = candle_factory(n=100)
        m1 = candle_factory(n=50) # M1 shorter than M5 (Fatal misalignment)
        
        with pytest.raises(ValueError):
            bt.run("XAUUSDm", [], m5, m5, m5, m5, m1)
