from core.common.types import MarketRegime, VolatilityStatus
from core.base_strategy import MarketData
from dataclasses import dataclass, field, asdict
import numpy as np
import logging
import math
import json
import time

@dataclass(frozen=True)
class RegimeState:
    """Immutable state snapshot for Institutional Regime Engine v3."""
    breakout_count: int = 0
    last_direction: int = 0
    last_session: str = "UNKNOWN"
    last_bar_index: int = 0
    last_reset_ts: float = 0.0

@dataclass
class RegimeTrace:
    """Observability ledger for auditing regime classification decisions."""
    regime: str
    divergence_score: float
    sigmoid_inputs: dict
    feature_snapshot: dict
    decision_path: list[str]
    execution_id: str

@dataclass
class RegimeInfo:
    """Institutional classification result with embedded audit trace."""
    market_type: MarketRegime
    volatility: VolatilityStatus
    confidence: float
    adx: float
    atr: float
    session: str = "UNKNOWN"
    trace: RegimeTrace = field(default=None)

def reduce_state(state: RegimeState, regime: MarketRegime, direction: int, execution_id: str, session: str, bar_index: int) -> RegimeState:
    """
    Pure state reducer. Deterministically transforms old state into a new snapshot.
    Enforces rules for breakout counting and state resets.
    """
    new_breakout_count = state.breakout_count
    new_direction = direction
    
    # Rule 1: Session Reset
    if session != state.last_session:
        return RegimeState(breakout_count=0, last_direction=0, last_session=session, last_bar_index=bar_index, last_reset_ts=time.time())

    # Rule 2: Regime-Specific Transitions
    if regime == MarketRegime.LIQUIDITY_EVENT:
        new_breakout_count = 0  # Structural event resets momentum continuity
    elif regime == MarketRegime.TREND:
        # Only increment if direction matches
        if direction == state.last_direction:
            new_breakout_count = min(3, state.breakout_count + 1)
        else:
            new_breakout_count = 1
    elif regime in (MarketRegime.RANGE, MarketRegime.TRANSITION):
        new_breakout_count = max(0, state.breakout_count - 1)
        if new_breakout_count == 0:
            new_direction = 0

    return RegimeState(
        breakout_count=new_breakout_count,
        last_direction=new_direction,
        last_session=session,
        last_bar_index=bar_index,
        last_reset_ts=state.last_reset_ts
    )

class RegimeDetector:
    """
    Institutional Regime Inference Engine v3 (LOCKED).
    Pure functional deterministic classifier.
    """

    def __init__(self, adx_period: int = 14, atr_period: int = 14):
        self.adx_period = adx_period
        self.atr_period = atr_period
        self.logger = logging.getLogger("trading_bot.regime_detector")

    def _sigmoid(self, z: float) -> float:
        """Stabilized sigmoid function for confidence fusion."""
        try:
            if z > 10: return 0.99
            if z < -10: return 0.01
            return 1.0 / (1.0 + math.exp(-z))
        except OverflowError:
            return 0.99 if z > 0 else 0.01

    def detect(self, market_data: MarketData, state: RegimeState, execution_id: str, strategy_id: str, is_live: bool = False) -> (RegimeInfo, RegimeState, RegimeTrace):
        """
        Pure deterministic inference cycle following the Institutional v3 8-step pipeline.
        
        Args:
            market_data: Snapshot of HTF and LTF candle series.
            state: Previous validated RegimeState.
            execution_id: Canonical anchor for the current cycle.
            strategy_id: Unique strategy identifier for state isolation.
            is_live: Flag for runtime-specific observability contracts.
        """
        m5_candles = market_data.m5_candles
        htf_candles = market_data.htf_candles
        decision_path = []

        # --- STEP 1: VALIDATION GATE ---
        adx_series = m5_candles.get_indicator(f"adx_{self.adx_period}")
        atr_series = m5_candles.get_indicator(f"atr_{self.atr_period}")
        
        # Guard for insufficient data
        if adx_series is None or len(adx_series) < 20 or \
           atr_series is None or len(atr_series) < 100 or \
           htf_candles is None or len(htf_candles) < 50:
            decision_path.append("FAIL_DATA_INADEQUATE")
            self.logger.warning(
                f"[{execution_id}] REGIME DATA INADEQUATE: "
                f"adx_len={len(adx_series) if adx_series is not None else 0}, "
                f"atr_len={len(atr_series) if atr_series is not None else 0}, "
                f"htf_len={len(htf_candles) if htf_candles is not None else 0} "
                f"(need adx>=20, atr>=100, htf>=50)"
            )
            trace = RegimeTrace(MarketRegime.TRANSITION.value, 0.0, {}, {}, decision_path, execution_id)
            return RegimeInfo(MarketRegime.TRANSITION, VolatilityStatus.NORMAL, 0.0, 0.0, 0.0, session=market_data.session, trace=trace), state, trace

        # --- STEP 2: VOLATILITY SHOCK FILTER ---
        atr = atr_series[-1]
        avg_atr = np.mean(atr_series[-100:])
        vol_ratio = atr / avg_atr if avg_atr > 0 else 1.0
        
        if vol_ratio > 3.0 or vol_ratio < 0.2:
            decision_path.append("SHOCK_RESET")
            # Explicitly return reset state
            new_state = RegimeState(last_session=market_data.session, last_reset_ts=time.time())
            trace = RegimeTrace(MarketRegime.TRANSITION.value, 1.0, {"vol": vol_ratio}, {}, decision_path, execution_id)
            return RegimeInfo(MarketRegime.TRANSITION, VolatilityStatus.HIGH if vol_ratio > 3.0 else VolatilityStatus.LOW, 0.01, adx_series[-1], atr, session=market_data.session, trace=trace), new_state, trace

        # --- STEP 3: FEATURE CACHE (SINGLE PASS) ---
        current_close = m5_candles.c[-1]
        current_open = m5_candles.o[-1]
        adx = adx_series[-1]
        
        # HTF Trend Core
        ema50 = htf_candles.ema(50)
        ema200 = htf_candles.ema(200)
        htf_ema_fast = ema50[-1]
        htf_ema_slow = ema200[-1]
        htf_trend = 1 if htf_ema_fast > htf_ema_slow else (-1 if htf_ema_fast < htf_ema_slow else 0)
        
        # Momentum/Slope
        htf_slope = (ema50[-1] - ema50[-5]) / max(atr, 0.0001) if len(ema50) > 5 else 0.0
        
        # LTF Trend (M5)
        m5_ema50 = m5_candles.ema(50)
        m5_trend = 1 if current_close > m5_ema50[-1] else -1

        # BB / Displacement
        upper, lower, mid = m5_candles.bollinger_bands(20, 2.0)
        bb_bandwidth = (upper[-1] - lower[-1]) / mid[-1]
        avg_bandwidth = np.mean((upper[-20:] - lower[-20:]) / mid[-20:])
        
        body = abs(current_close - current_open)
        displacement_strength = body / atr if atr > 0 else 0
        
        # --- STEP 4: DIVERGENCE ENGINE ---
        trend_mismatch = 1.0 if (htf_trend != 0 and m5_trend != htf_trend) else 0.0
        vol_mismatch = 1.0 if (vol_ratio > 1.5 and avg_bandwidth < bb_bandwidth * 0.7) else 0.0 # Heuristic: Vol spike but no BB compression
        momentum_divergence = 1.0 if (htf_trend == 1 and htf_slope < -0.2) or (htf_trend == -1 and htf_slope > 0.2) else 0.0
        
        divergence_score = (0.4 * trend_mismatch) + (0.3 * vol_mismatch) + (0.3 * momentum_divergence)

        # --- STEP 5: REGIME CLASSIFICATION (STRICT PRIORITY) ---
        market_regime = MarketRegime.RANGE
        
        # A. Liquidity Check
        rolling_htf_high = np.max(htf_candles.h[-24:-1]) if len(htf_candles) > 24 else market_data.current_price
        rolling_htf_low = np.min(htf_candles.l[-24:-1]) if len(htf_candles) > 24 else market_data.current_price
        
        is_sweep = (market_data.m5_candles.h[-1] > rolling_htf_high and current_close < rolling_htf_high) or \
                   (market_data.m5_candles.l[-1] < rolling_htf_low and current_close > rolling_htf_low)
        
        # B. Rejection Check
        is_breakout = (current_close > upper[-1] or current_close < lower[-1])
        is_rejection = False
        if not is_breakout and state.breakout_count > 0:
            if body > (abs(m5_candles.c[-2] - m5_candles.o[-2]) * 1.5): # Strong snap-back
                is_rejection = True

        # FINAL PRIORITY CHAIN
        if is_sweep:
            market_regime = MarketRegime.LIQUIDITY_EVENT
            decision_path.append("LIQUIDITY_PRIORITY")
        elif is_rejection:
            market_regime = MarketRegime.RANGE
            decision_path.append("REJECTION_PRIORITY")
        elif divergence_score > 0.70:
            market_regime = MarketRegime.TRANSITION
            decision_path.append("CONFLICT_OVERRIDE")
        elif is_breakout and bb_bandwidth > (avg_bandwidth * 1.2):
            if state.breakout_count >= 1:
                market_regime = MarketRegime.TREND
                decision_path.append("TREND_CONTINUATION")
            else:
                market_regime = MarketRegime.EXPANSION
                decision_path.append("MOMENTUM_EXPANSION")
        else:
            if adx > 25:
                market_regime = MarketRegime.TREND
                decision_path.append("ADX_TREND")
            elif adx < 20:
                market_regime = MarketRegime.RANGE
                decision_path.append("ADX_RANGE")
            else:
                market_regime = MarketRegime.TRANSITION
                decision_path.append("DEFAULT_TRANSITION")

        # --- STEP 6: SIGMOID CONFIDENCE ---
        # Feature Normalization [0, 1]
        n_vol = min(1.0, vol_ratio / 2.0)
        n_adx = min(1.0, adx / 50.0)
        n_disp = min(1.0, displacement_strength / 2.0)
        n_cont = min(1.0, state.breakout_count / 3.0)
        
        # Z-Space Weighting
        z = (3.0 * n_vol) + (2.0 * n_adx) + (2.5 * n_disp) + (2.0 * n_cont) - (4.0 * divergence_score)
        confidence = self._sigmoid(z)
        
        # --- STEP 7: STATE TRANSITION (REDUCER) ---
        direction = 1 if current_close > mid[-1] else -1
        new_state = reduce_state(
            state, 
            market_regime, 
            direction, 
            execution_id, 
            market_data.session, 
            int(market_data.timestamp.timestamp())
        )

        # --- STEP 8: TRACE OUTPUT ---
        vol_status = VolatilityStatus.NORMAL
        if vol_ratio > 1.8: vol_status = VolatilityStatus.HIGH
        elif vol_ratio < 0.6: vol_status = VolatilityStatus.LOW

        trace = RegimeTrace(
            regime=market_regime.value,
            divergence_score=float(divergence_score),
            sigmoid_inputs={"vol": n_vol, "adx": n_adx, "disp": n_disp, "cont": n_cont, "z": z},
            feature_snapshot={"adr": atr, "vol_ratio": vol_ratio, "htf_trend": htf_trend},
            decision_path=decision_path,
            execution_id=execution_id
        )

        return RegimeInfo(market_regime, vol_status, confidence, adx, atr, session=market_data.session, trace=trace), new_state, trace
