import numpy as np
import logging
from typing import Optional, Dict, Any
from core.base_strategy import BaseStrategy, MarketData
from core.common.types import TradeSignal
from core.session_detector import SessionDetector

logger = logging.getLogger("trading_bot.strategy.smart_mean_reversion")

class SmartMeanReversionStrategy(BaseStrategy):
    """
    V5 Institutional Smart Mean Reversion (Numpy-Hardened).
    Fades algorithmic chop using manual mathematical arrays for BB and RSI.
    """

    def __init__(self, strategy_id: str, config: dict):
        super().__init__(strategy_id, config)
        
        # [ Institutional Config Resolution ]: Access the resolved strategy block
        strat_config = self.get_strat_config()
        
        self.bb_period = strat_config.get("bb_period", 20)
        self.bb_std = strat_config.get("bb_std", 2.0)
        self.rsi_period = strat_config.get("rsi_period", 21)
        self.rsi_overbought = strat_config.get("rsi_overbought", 75)
        self.rsi_oversold = strat_config.get("rsi_oversold", 25)
        
        self.sl_atr = strat_config.get("sl_atr", 1.5)
        self.tp_atr = strat_config.get("tp_atr", 4.5)
        self.min_confidence = float(strat_config.get("min_confidence", self.min_confidence))
        
        # ── Institutional Gating ──
        self.allowed_sessions = strat_config.get("allowed_sessions", ["TOKYO", "LONDON", "NEW_YORK", "LONDON/NY"])
        self.va_lookback = strat_config.get("va_lookback", 200)
        self.adx_max_threshold = strat_config.get("adx_max_threshold", 35.0)
        self.vol_climax_ratio = strat_config.get("vol_climax_ratio", 1.2)
        self.buffer_atr_mult = 0.2  # Gold Wick-Hunt Protection
        
        # ── State Tracking ──
        self._last_signal_bar = 0
        self.min_bars_between_signals = strat_config.get("min_bars_between_signals", 15)
        self.max_ema_slope = strat_config.get("max_ema_slope", 10.0) # Pips per bar on H1 (Safety Gate)


    def _calculate_value_area(self, prices: np.ndarray, volumes: np.ndarray, bins: int = 50) -> Dict[str, float]:
        """
        Institutional Volume Profile (VBP) Calculator.
        Determines VAH, VAL, and POC within the specified window.
        """
        if len(prices) == 0: return {"poc": 0, "vah": 0, "val": 0}
        
        p_min, p_max = np.min(prices), np.max(prices)
        if p_min == p_max: return {"poc": p_min, "vah": p_min, "val": p_min}
        
        step = (p_max - p_min) / bins
        profile = np.zeros(bins)
        
        for i in range(len(prices)):
            idx = int((prices[i] - p_min) / step)
            if idx >= bins: idx = bins - 1
            profile[idx] += volumes[i]
            
        total_vol = np.sum(volumes)
        target_vol = total_vol * 0.70 # Institutional 70% Value Area
        
        poc_idx = np.argmax(profile)
        poc = p_min + (poc_idx * step) + (step / 2)
        
        # Expand from POC to cover 70% volume
        current_vol = profile[poc_idx]
        l_idx, r_idx = poc_idx, poc_idx
        
        while current_vol < target_vol:
            # Check expansion left/right
            l_val = profile[l_idx-1] if l_idx > 0 else 0
            r_val = profile[r_idx+1] if r_idx < bins-1 else 0
            
            if l_val == 0 and r_val == 0: break
            
            if l_val >= r_val:
                current_vol += l_val
                l_idx -= 1
            else:
                current_vol += r_val
                r_idx += 1
                
        return {
            "poc": poc,
            "vah": p_min + (r_idx * step) + step,
            "val": p_min + (l_idx * step)
        }

    def _detect_volume_climax(self, m5: Any) -> bool:
        """Determines if the current candle represents a volume exhaustion climax."""
        vol_sma = m5.get_indicator("vol_sma_20")
        current_vol_sma = vol_sma[-1] if len(vol_sma) > 0 and not np.isnan(vol_sma[-1]) else 0
        
        # Fallback: Manual calculation if pre-calculation is missing
        if current_vol_sma == 0:
            if len(m5.v) >= 20:
                current_vol_sma = np.mean(m5.v[-20:])
            else:
                return False # Not enough data
                
        return m5.v[-1] > (current_vol_sma * self.vol_climax_ratio)

    def _is_trend_too_strong(self, market_data: MarketData) -> bool:
        """
        Institutional 'Runaway Train' Filter.
        Calculates the slope of H1 EMA(200). If the trend is too parabolic, 
        mean reversion is statistically suicidal.
        """
        h1 = market_data.htf_candles
        ema_200 = h1.get_indicator("ema_200")
        
        if len(ema_200) < 6: return False
        
        # Calculate slope over last 5 bars (H1)
        slope = (ema_200[-1] - ema_200[-6]) / 5.0
        
        # Convert to pips (Gold: 0.01 = 1 pip usually, but let's be safe with point)
        point = 0.01 # Standard for XAUUSDm
        pips_per_bar = abs(slope) / point
        
        return pips_per_bar > self.max_ema_slope


    def generate_signal(self, market_data: MarketData) -> Optional[TradeSignal]:
        if not self.is_spread_safe(market_data): return None
        
        if not SessionDetector.is_session_active(market_data.timestamp, allowed_sessions=self.allowed_sessions):
            self.last_rejection_reason = "Out of Kill Zone"
            return None
            
        m5 = market_data.m5_candles
        if len(m5) < self.va_lookback: return None
        
        # 1. Cooldown & Time Gating
        if len(m5) - self._last_signal_bar < self.min_bars_between_signals:
            return None
        
        # 2. Institutional Gating (ADX & EMA Slope)
        htf = market_data.htf_candles
        adx_14 = market_data.m15_candles.get_indicator("adx_14")
        
        if len(adx_14) > 0 and adx_14[-1] > self.adx_max_threshold:
            self.last_rejection_reason = "Trend Strength High (ADX > Threshold)"
            return None
            
        if self._is_trend_too_strong(market_data):
            self.last_rejection_reason = "Parabolic Trend Detected (EMA Slope High)"
            return None
            
        # 3. Value Area Calculation (VBP)
        recent_c = m5.c[-self.va_lookback:]
        recent_v = m5.v[-self.va_lookback:]
        va = self._calculate_value_area(recent_c, recent_v)
        
        atr = m5.atr(14)[-1]
        buffer = self.buffer_atr_mult * atr
        
        current_price = market_data.current_price
        
        # 4. Signal Detection (Fading Extensions)
        direction = None
        if current_price > (va["vah"] + buffer):
            # Price is overextended beyond VAH + Buffer
            if self._detect_volume_climax(m5): # Exhaustion confirmed
                direction = "SELL"
        elif current_price < (va["val"] - buffer):
            # Price is overextended below VAL - Buffer
            if self._detect_volume_climax(m5):
                direction = "BUY"
                
        if direction:
            # Check for Rejection Candle (Closing back toward Value)
            last = m5[-1]
            target_price = va["poc"]
            
            if direction == "SELL" and last.close < last.high:
                sl_price = va["vah"] + (0.8 * atr)
                risk = sl_price - current_price
                reward = current_price - target_price
                
                if risk <= 0 or (reward / risk) < 1.5:
                    self.last_rejection_reason = f"Poor RR ({reward/risk:.2f} if risk > 0 else 'Invalid')"
                    return None
                    
                self._last_signal_bar = len(m5)
                return TradeSignal(
                    direction="SELL",
                    price=current_price,
                    confidence=0.80,
                    reasons=["VBP:VAH_EXTENSION", "VOL:CLIMAX", f"POC:{va['poc']:.2f}"],
                    stop_loss=sl_price,
                    take_profit=target_price, # Main Target
                    tp1_price=target_price,
                    tp2_price=va["val"]   # Full Rotation Target
                )
            elif direction == "BUY" and last.close > last.low:
                sl_price = va["val"] - (0.8 * atr)
                risk = current_price - sl_price
                reward = target_price - current_price
                
                if risk <= 0 or (reward / risk) < 1.5:
                    self.last_rejection_reason = f"Poor RR ({reward/risk:.2f} if risk > 0 else 'Invalid')"
                    return None
                    
                self._last_signal_bar = len(m5)
                return TradeSignal(
                    direction="BUY",
                    price=current_price,
                    confidence=0.80,
                    reasons=["VBP:VAL_EXTENSION", "VOL:CLIMAX", f"POC:{va['poc']:.2f}"],
                    stop_loss=sl_price,
                    take_profit=target_price,
                    tp1_price=target_price,
                    tp2_price=va["vah"]
                )

    def get_parameter_grid(self) -> Dict[str, list]:
        """Institutional grid for WFO optimization."""
        return {
            "bb_std": [1.5, 2.0, 2.5],
            "vol_climax_ratio": [1.2, 1.5, 2.0],
            "adx_max_threshold": [20.0, 25.0, 30.0]
        }


    def get_metrics(self, market_data: MarketData) -> Dict[str, Any]:
        """Returns live Value Area metrics for the institutional dashboard."""
        m5 = market_data.m5_candles
        if len(m5) < self.va_lookback: return {}
        
        recent_c = m5.c[-self.va_lookback:]
        recent_v = m5.v[-self.va_lookback:]
        va = self._calculate_value_area(recent_c, recent_v)
        
        return {
            "VBP_VAH": round(va["vah"], 2),
            "VBP_VAL": round(va["val"], 2),
            "VBP_POC": round(va["poc"], 2)
        }

