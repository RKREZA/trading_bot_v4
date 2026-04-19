from core.common.types import MarketRegime, VolatilityStatus
from core.base_strategy import MarketData # Required for type hinting internally
import numpy as np
import logging
import math

class RegimeInfo:
    def __init__(self, market_type: MarketRegime, volatility: VolatilityStatus, confidence: float, adx_val: float, atr_val: float):
        self.market_type = market_type
        self.volatility = volatility
        self.confidence = confidence
        self.adx = adx_val
        self.atr = atr_val

    def __repr__(self):
        return f"<Regime:{self.market_type.value} Vol:{self.volatility.value} Conf:{self.confidence:.2f} ADX:{self.adx:.1f} ATR:{self.atr:.5f}>"

class RegimeDetector:
    """
    Institutional Regime Detection Layer.
    Implements Deterministic Multi-Factor Classification:
    - Liquidity Event Detection
    - Expansion / Breakout RegExp
    - Acceptance vs Rejection Logic
    - Transition / Uncertainty Management
    - Multi-Timeframe Alignment
    """

    def __init__(self, adx_period: int = 14, atr_period: int = 14):
        self.adx_period = adx_period
        self.atr_period = atr_period
        self.logger = logging.getLogger("trading_bot.regime_detector")
        # State tracking for acceptance logic
        self._consecutive_closes_outside_bb = 0
        self._breakout_direction = 0  # 1 for upper, -1 for lower
        self._last_price = 0.0

    def detect(self, market_data: 'MarketData') -> RegimeInfo:
        m5_candles = market_data.m5_candles
        htf_candles = market_data.htf_candles
        
        adx_series = m5_candles.get_indicator(f"adx_{self.adx_period}")
        atr_series = m5_candles.get_indicator(f"atr_{self.atr_period}")
        
        if len(adx_series) < 20 or len(atr_series) < 100 or len(htf_candles) < 24:
            return RegimeInfo(MarketRegime.TRANSITION, VolatilityStatus.NORMAL, 0.0, 0.0, 0.0)

        current_close = m5_candles.c[-1]
        current_open = m5_candles.o[-1]
        current_high = m5_candles.h[-1]
        current_low = m5_candles.l[-1]
        
        # 1. Base Metrics
        adx = adx_series[-1]
        atr = atr_series[-1]
        
        avg_atr = np.mean(atr_series[-100:])
        vol_ratio = atr / avg_atr if avg_atr > 0 else 1.0

        # Multi-Timeframe Bias (HTF Trend, Slope & Volatility)
        ema50 = htf_candles.ema(50)
        htf_ema_fast = ema50[-1] if (ema50 is not None and len(ema50) > 0) else current_close
        
        ema200 = htf_candles.ema(200)
        htf_ema_slow = ema200[-1] if (ema200 is not None and len(ema200) > 0) else current_close
        
        # Binary bias
        htf_trend = 1 if htf_ema_fast > htf_ema_slow else (-1 if htf_ema_fast < htf_ema_slow else 0)
        
        # Momentum/Slope validation
        htf_slope = 0.0
        if ema50 is not None and len(ema50) > 5:
            htf_slope = (ema50[-1] - ema50[-5]) / max(atr, 0.0001)
        
        htf_atr_series = htf_candles.get_indicator(f"atr_{self.atr_period}")
        htf_vol_ratio = 1.0
        if len(htf_atr_series) >= 24:
            htf_avg_atr = np.mean(htf_atr_series[-24:])
            htf_vol_ratio = htf_atr_series[-1] / htf_avg_atr if htf_avg_atr > 0 else 1.0

        # Volatility Status Baseline
        vol_status = VolatilityStatus.NORMAL
        if vol_ratio > 1.8:
            vol_status = VolatilityStatus.HIGH
        elif vol_ratio < 0.6:
            vol_status = VolatilityStatus.LOW

        # 2. Advanced Feature Detection

        # Feature A: Displacement & Compression (Bollinger Bands)
        upper, lower, mid = m5_candles.bollinger_bands(20, 2.0)
        bb_bandwidth = (upper[-1] - lower[-1]) / mid[-1]
        bb_bandwidth_history = (upper[-20:] - lower[-20:]) / mid[-20:]
        avg_bandwidth = np.mean(bb_bandwidth_history)
        
        is_breakout = (current_close > upper[-1] or current_close < lower[-1])
        
        # Displacement Strength: candle body vs ATR
        body = abs(current_close - current_open)
        displacement_strength = body / atr if atr > 0 else 0
        is_strong_displacement = displacement_strength > 1.25

        # Momentum persistence (volume confirmation + follow-through + structural close)
        is_persistence = False
        if len(m5_candles.v) >= 10:
            avg_vol_recent = np.mean(m5_candles.v[-10:-1])
            is_increasing_vol = m5_candles.v[-1] > avg_vol_recent
            
            # Close position within the candle range (Top 25% or Bottom 25%)
            candle_range = current_high - current_low
            close_position = (current_close - current_low) / max(candle_range, 0.0001)
            is_strong_close = (is_breakout and current_close > upper[-1] and close_position > 0.75) or \
                              (is_breakout and current_close < lower[-1] and close_position < 0.25)
                              
            # Is displacement sustained over previous candle?
            prev_body = abs(m5_candles.c[-2] - m5_candles.o[-2])
            is_sustained_momentum = body > (prev_body * 0.8)

            is_persistence = is_increasing_vol and is_strong_displacement and is_strong_close and is_sustained_momentum
        
        # Acceptance vs Return-to-Mean (Rejection) logic
        is_rejection = False
        if is_breakout:
            current_dir = 1 if current_close > upper[-1] else -1
            if current_dir == self._breakout_direction:
                self._consecutive_closes_outside_bb += 1
            else:
                self._consecutive_closes_outside_bb = 1
                self._breakout_direction = current_dir
        else:
            # Rejection analysis: If we were previously outside, but now closed back inside strongly
            if self._consecutive_closes_outside_bb > 0:
                # Calculate return to mean speed (opposing candle body > previous breakout body)
                if len(m5_candles.c) >= 2:
                    prev_body = abs(m5_candles.c[-2] - m5_candles.o[-2])
                    if body > (prev_body * 1.5): # Harsh aggressive snap-back
                        is_rejection = True
            
            # Accelerate time decay if rejection found
            if is_rejection:
                self._consecutive_closes_outside_bb = 0
                self._breakout_direction = 0
            else:
                self._consecutive_closes_outside_bb = max(0, self._consecutive_closes_outside_bb - 1)
                if self._consecutive_closes_outside_bb == 0:
                    self._breakout_direction = 0
            
        # Feature B: Institutional Adaptive Liquidity Sweep (HTF PDH/PDL via rolling window)
        # Using ATR-adaptive lookback window for key levels
        base_lookback = 24
        vol_adj = max(0.5, min(2.0, 1.0 / vol_ratio))
        adaptive_lookback = int(base_lookback * vol_adj)
        
        # Hard guard for insufficient history
        safe_lookback = min(adaptive_lookback, len(htf_candles) - 1)
        if safe_lookback < 5:
            rolling_htf_high = current_high
            rolling_htf_low = current_low
        else:
            rolling_htf_high = np.max(htf_candles.h[-safe_lookback:-1])
            rolling_htf_low = np.min(htf_candles.l[-safe_lookback:-1])
        
        # Wick penetration speed/depth (must penetrate past HTF key level and return)
        is_liquidity_sweep_high = (current_high > rolling_htf_high) and (current_close < rolling_htf_high) and ((current_high - rolling_htf_high) > (atr * 0.5))
        is_liquidity_sweep_low = (current_low < rolling_htf_low) and (current_close > rolling_htf_low) and ((rolling_htf_low - current_low) > (atr * 0.5))
        
        # Volume Confirmation: Liquidity sweeps require climatic volume participation
        vol_confirmed = False
        if len(m5_candles.v) >= 20:
            avg_vol = np.mean(m5_candles.v[-20:-1])
            if m5_candles.v[-1] > (avg_vol * 1.5):
                vol_confirmed = True
                
        is_liquidity_event = (is_liquidity_sweep_high or is_liquidity_sweep_low) and vol_confirmed

        # Feature C: Regime Conflict Detection
        m5_trend = 1 if current_close > m5_candles.ema(50)[-1] else (-1 if current_close < m5_candles.ema(50)[-1] else 0)
        # Include HTF slope divergence (e.g., HTF is bullish but momentum is crashing)
        is_momentum_divergence = (htf_trend == 1 and htf_slope < -0.2) or (htf_trend == -1 and htf_slope > 0.2)
        
        is_bias_conflict = (htf_trend != 0 and m5_trend != 0 and htf_trend != m5_trend)
        is_vol_conflict = (htf_vol_ratio < 0.8 and vol_ratio > 1.5) # LTF expanding inside HTF compression noise
        is_conflict = is_bias_conflict or is_vol_conflict or is_momentum_divergence

        # 3. Institutional Classification State Machine (Deterministic Conflict Resolver)
        # Priority 0: Snap Rejections (Immediate rotation to Mean Reversion)
        if is_rejection:
            market_type = MarketRegime.RANGE
            
        # Priority 1: Systemic Conflict or Divergence (Force Neutrality)
        elif is_conflict:
            market_type = MarketRegime.TRANSITION
            
        # Priority 2: Institutional Structure (Liquidity Sweep > Momentum)
        elif is_liquidity_event:
            market_type = MarketRegime.LIQUIDITY_EVENT
            
        # Priority 3: Expansion (Hysteresis-protected momentum)
        elif is_breakout:
            # Multi-factor validation for True Trend
            is_valid_expansion = bb_bandwidth > (avg_bandwidth * 1.5) and vol_ratio > 1.5 and is_strong_displacement
            
            if is_valid_expansion:
                # Acceptance confirmed via persistence and structural close
                if self._consecutive_closes_outside_bb >= 2 and is_persistence:
                    market_type = MarketRegime.TREND 
                else:
                    market_type = MarketRegime.EXPANSION
            else:
                # Weak breakout without bandwidth expansion -> likely noise or early range exhaustion
                market_type = MarketRegime.TRANSITION

        # Priority 4: Background Regimes (ADX / Sentiment)
        else:
            if adx >= 25: 
                market_type = MarketRegime.TREND
            elif adx < 20:
                market_type = MarketRegime.RANGE
            else:
                market_type = MarketRegime.TRANSITION

        # 4. Confidence Fusion Engine
        # Base calculations normalized (0 to 1)
        norm_vol = min(1.0, max(0.0, (vol_ratio - 0.5) / 1.5))
        norm_adx = min(1.0, adx / 50.0)
        norm_disp = min(1.0, displacement_strength / 2.0)
        
        # Fusion Matrix (0.35 Vol + 0.25 ADX + 0.25 Displacement) + Bonuses for structural alignment
        fused_confidence = (0.35 * norm_vol) + (0.25 * norm_adx) + (0.25 * norm_disp)
        
        if is_persistence:
            fused_confidence += 0.15  # Institutional momentum bonus
            
        if is_liquidity_event:
            fused_confidence += 0.15  # Structure alignment bonus

        # Penalties for minor divergence or contradictions 
        if not is_persistence and market_type in (MarketRegime.EXPANSION, MarketRegime.TREND):
            fused_confidence -= 0.20
            
        if is_conflict:
            fused_confidence -= 0.35  # Heavy penalty
            
        if is_rejection:
            fused_confidence -= 0.25  # Sudden rejection destabilizes confidence
            
        # Ensure bounds
        final_confidence = min(0.99, max(0.01, fused_confidence))

        return RegimeInfo(market_type, vol_status, final_confidence, adx, atr)
