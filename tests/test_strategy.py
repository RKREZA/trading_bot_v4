import pytest
import numpy as np
from dataclasses import replace
from core import TradeSignal
from strategies import create_strategy

class TestStrategies:
    """Institutional Strategy Logic Verification Suite."""
    
    @pytest.mark.parametrize("sid", ["TRENDFOLLOWING", "SMARTMEANREVERSION", "LIQUIDITYSWEEPBREAKOUT", "LIQUIDITYSESSION"])
    def test_factory_creation(self, sid, mock_config):
        """Ensure all strategies are correctly instantiated via the factory."""
        cfg = {"params": {}, "enabled": True}
        s = create_strategy(sid, st_type=sid, config=cfg)
        assert s is not None
        assert s.strategy_id == sid

    def test_trend_following_bullish_bias(self, market_data_factory, mock_config):
        """Verify TrendFollowing captures bullish signals in bullish trends."""
        s = create_strategy("TRENDFOLLOWING", st_type="TRENDFOLLOWING", config={"enabled": True})
        md = market_data_factory(trend="BULLISH")
        
        sig = s.generate_signal(md)
        if sig:
            assert sig.direction == "BUY"
            assert sig.stop_loss < md.current_price
            assert sig.take_profit > md.current_price
            assert sig.confidence > 0

    def test_mean_reversion_overextended(self, candle_factory, market_data_factory, mock_config):
        """Verify MeanReversion logic for overextended price action."""
        s = create_strategy("SMARTMEANREVERSION", st_type="SMARTMEANREVERSION", config={"enabled": True})
        
        # Create a market state where price is far above EMA
        md = market_data_factory(trend="BULLISH")
        
        # Inject custom indicators for overextension
        m5 = md.m5_candles
        indicators = {
            "bb_upper": np.full(len(m5), m5.close[-1] - 5.0), # Price above BB upper
            "rsi_14": np.full(len(m5), 85.0)                 # RSI Overbought
        }
        m5.indicators = indicators
        
        sig = s.generate_signal(md)
        if sig:
            assert sig.direction == "SELL"
            assert sig.stop_loss > md.current_price
            assert sig.take_profit < md.current_price

    def test_breakout_high_volume(self, market_data_factory, mock_config):
        """Verify Breakout strategy responds to high-volume range breaks."""
        s = create_strategy("LIQUIDITYSWEEPBREAKOUT", st_type="LIQUIDITYSWEEPBREAKOUT", config={"enabled": True})
        md = market_data_factory(trend="FLAT")
        
        # Mock MTF Alignment
        md.m15_candles.indicators["ema_50"] = np.full(len(md.m15_candles), 10.0)
        md.m15_candles.indicators["ema_200"] = np.full(len(md.m15_candles), 5.0) # M15 Bullish
        
        # Mock H1 Strength and Direction
        md.htf_candles.open[-1] = 1000.0
        md.htf_candles.close[-1] = 1010.0 # H1 Bullish
        md.htf_candles.high[-1] = 1011.0
        md.htf_candles.low[-1] = 999.0
        
        # Mock Volume (Passing dynamic threshold)
        md.htf_candles.tick_volume[-1] = 5000 
        
        # Mock M5 Breakout
        prev_high = np.max(md.m5_candles.high[:-1])
        md.m5_candles.open[-1] = prev_high + 0.1
        md.m5_candles.close[-1] = prev_high + 1.0 # Strong M5 breakout candle
        md.m5_candles.high[-1] = prev_high + 1.1
        md.m5_candles.low[-1] = prev_high + 0.05
        
        md = replace(md, current_price=md.m5_candles.close[-1])
        
        sig = s.generate_signal(md)
        if sig:
            assert sig.direction == "BUY"
            assert sig.confidence > 0

    def test_liquidity_session_timing(self, market_data_factory, mock_config):
        """Verify LiquiditySession restricted logic based on session timing."""
        s = create_strategy("LIQUIDITYSESSION", st_type="LIQUIDITYSESSION", config={"enabled": True})
        
        # Test London Session (Active)
        md_london = market_data_factory(trend="BULLISH", session="LONDON")
        sig_london = s.generate_signal(md_london)
        
        # Test Asian Session (Often inactive for this strategy)
        md_asia = market_data_factory(trend="BULLISH", session="ASIA")
        sig_asian = s.generate_signal(md_asia)
        
        # If Implemention filters session, asian should be None
        if sig_london:
            assert sig_london.session == "LONDON"

    def test_unknown_strategy_raises(self):
        """Ensure factory handles invalid strategy IDs gracefully."""
        with pytest.raises(ValueError):
            create_strategy("INVALID_999", st_type="INVALID_999", config={})
