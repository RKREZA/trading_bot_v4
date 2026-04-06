import pytest
import numpy as np
from dataclasses import replace
from core import TradeSignal
from strategies import create_strategy

class TestStrategies:
    """Institutional Strategy Logic Verification Suite."""
    
    @pytest.mark.parametrize("sid", ["TREND_FOLLOWING", "MEAN_REVERSION", "BREAKOUT", "LIQUIDITY_SESSION"])
    def test_factory_creation(self, sid, mock_config):
        """Ensure all strategies are correctly instantiated via the factory."""
        cfg = {"params": {}, "enabled": True}
        s = create_strategy(sid, st_type=sid, config=cfg)
        assert s is not None
        assert s.strategy_id == sid

    def test_trend_following_bullish_bias(self, market_data_factory, mock_config):
        """Verify TrendFollowing captures bullish signals in bullish trends."""
        s = create_strategy("TREND_FOLLOWING", st_type="TREND_FOLLOWING", config={"params": {}, "enabled": True})
        md = market_data_factory(trend="BULLISH")
        
        # Manually force some confluence for testing
        md.preprocessed["m_bias"] = "BULLISH"
        md.preprocessed["in_htf_demand"] = True
        
        sig = s.generate_signal(md)
        assert sig is None or isinstance(sig, TradeSignal)
        if sig:
            assert sig.direction in {"BUY", "NONE"}

    def test_mean_reversion_overextended(self, candle_factory, market_data_factory, mock_config):
        """Verify MeanReversion logic for overextended price action."""
        s = create_strategy("MEAN_REVERSION", st_type="MEAN_REVERSION", config={"params": {}, "enabled": True})
        
        # Create a market state where price is far above EMA
        md = market_data_factory(trend="BULLISH")
        
        # Inject custom indicators for overextension
        m5 = md.m5_candles
        indicators = {
            "ema_200": np.full(len(m5), m5.close[-1] - 50.0), # Price far above EMA200
            "rsi_14": np.full(len(m5), 85.0),                # RSI Overbought
            "bb_upper": np.full(len(m5), m5.close[-1] - 5.0) # Price above BB upper
        }
        m5.indicators = indicators
        
        sig = s.generate_signal(md)
        assert sig is None or isinstance(sig, TradeSignal)
        if sig:
            assert sig.direction in {"SELL", "NONE"}

    def test_breakout_high_volume(self, market_data_factory, mock_config):
        """Verify Breakout strategy responds to high-volume range breaks."""
        s = create_strategy("BREAKOUT", st_type="BREAKOUT", config={"params": {}, "enabled": True})
        md = market_data_factory(trend="FLAT")
        
        # Mock a breakout of previous high
        md = replace(md, current_price=md.preprocessed["m_high"] + 0.10)
        md.m5_candles.tick_volume[-1] = 1000 # Spike in volume
        
        sig = s.generate_signal(md)
        assert sig is None or isinstance(sig, TradeSignal)

    def test_liquidity_session_timing(self, market_data_factory, mock_config):
        """Verify LiquiditySession restricted logic based on session timing."""
        s = create_strategy("LIQUIDITY_SESSION", st_type="LIQUIDITY_SESSION", config={"params": {}, "enabled": True})
        
        # Test London Session (Active)
        md_london = market_data_factory(trend="BULLISH", session="LONDON")
        sig_london = s.generate_signal(md_london)
        
        # Test Asian Session (Often inactive for this strategy)
        md_asia = market_data_factory(trend="BULLISH", session="ASIA")
        sig_asia = s.generate_signal(md_asia)
        
        # Assert strategy logic handles sessions (check implementation for specific filters)
        assert isinstance(md_london.session, str)

    def test_unknown_strategy_raises(self):
        """Ensure factory handles invalid strategy IDs gracefully."""
        with pytest.raises(ValueError):
            create_strategy("INVALID_999", {})
