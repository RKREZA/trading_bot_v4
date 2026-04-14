import numpy as np
import logging
from datetime import datetime, timezone
from typing import Optional, Dict, Any
from core.base_strategy import BaseStrategy, MarketData
from core.common.types import TradeSignal
from core.session_detector import SessionDetector

logger = logging.getLogger("trading_bot.strategy.liquidity_sweep_breakout")

class LiquiditySweepBreakoutStrategy(BaseStrategy):
    """
    Pure Liquidity Sweep (Anti-Breakout) Strategy.
    
    Core Logic:
    - Never buys high or sells low.
    - Waits for price to pierce the intraday range, then strongly reject and close inside.
    - Uses ADX to filter directions:
        - ADX < 25 (Ranging): Allow sweeps in both directions.
        - ADX >= 25 (Trending): Only allow sweeps that align with the M15 EMA trend.
    - Uses Hammer/Shooting Star math for rejection validation instead of pure body size.
    """

    def __init__(self, strategy_id: str, config: dict):
        super().__init__(strategy_id, config)
        
        strat_config = self.get_strat_config()
        self.lookback = strat_config.get("lookback", 20)
        
        # In a sweep, we care about the CLOSE position relative to the candle range (Hammer/Shooting Star)
        self.rejection_thresh = strat_config.get("rejection_thresh", 0.60) # Must close in upper/lower 40%
        self.min_confidence = float(strat_config.get("min_confidence", self.min_confidence))
        
        self.adx_trend_threshold = strat_config.get("adx_trend_threshold", 25.0)
        
        # FIX: Timestamp-based cooldown instead of broken `len(m5)` bar counting.
        # The previous approach stored `len(m5)` which is constant (full dataset size),
        # not the current sliding-window position — making the guard permanently 0 bars.
        # Cooldown duration: configurable in seconds (default 3.5 hours = 12,600s)
        self.cooldown_seconds = strat_config.get("cooldown_seconds", 12_600)  # 3.5h
        self._last_signal_time: float = 0.0  # UNIX timestamp of last queued signal
        
        # Daily trade cap: prevents over-trading on volatile days
        self.max_trades_per_day = strat_config.get("max_trades_per_day", 2)
        self._daily_trade_count: int = 0
        self._last_trade_date: str = ""  # ISO date string
        
        # Post-SL cooldown: after a stopout, add extra pause (seconds)
        self.post_sl_cooldown_seconds = strat_config.get("post_sl_cooldown_seconds", 14_400)  # 4h
        self._last_sl_time: float = 0.0  # UNIX timestamp of last SL hit
        
        # ATR spike filter: block entries when current ATR > N * 30-bar average (extreme trend days)
        self.max_atr_spike_mult = strat_config.get("max_atr_spike_mult", 2.5)
        
        self.session_multipliers = strat_config.get("session_multipliers", {
            "TOKYO": {"rej_boost": 0.0, "conf_boost": 0.0},
            "LONDON": {"rej_boost": 0.05, "conf_boost": 0.05},
            "NEW_YORK": {"rej_boost": 0.0, "conf_boost": 0.05},
            "LONDON/NY": {"rej_boost": 0.10, "conf_boost": 0.10},
            "GLOBAL": {"rej_boost": 0.0, "conf_boost": 0.0}
        })
        self._current_regime = "UNCERTAIN"

    def _get_m15_atr(self, market_data: MarketData) -> float:
        m15_atr_vals = market_data.m15_candles.atr(14)
        if len(m15_atr_vals) > 0 and not np.isnan(m15_atr_vals[-1]):
            return m15_atr_vals[-1]
        m5_atr = market_data.m5_candles.atr(14)
        if len(m5_atr) > 0 and not np.isnan(m5_atr[-1]):
            return m5_atr[-1] * 2.5
        return 15.0

    def generate_signal(self, market_data: MarketData) -> Optional[TradeSignal]:
        # TRADING ALL SESSIONS
        if not self.is_spread_safe(market_data):
            return None
            
        m5 = market_data.m5_candles
        if len(m5) < self.lookback + 1:
            self.last_rejection_reason = "Insufficient M5 data"
            return None
        
        # FIX: Timestamp-based cooldown (replaces broken `len(m5)` bar counter).
        # market_data.timestamp is always the current bar's datetime in UTC.
        now_ts = market_data.timestamp.timestamp()
        
        # 1. Standard cooldown between any two signals
        since_last = now_ts - self._last_signal_time
        if since_last < self.cooldown_seconds:
            self.last_rejection_reason = f"Signal cooldown active ({since_last/3600:.1f}h < {self.cooldown_seconds/3600:.1f}h)"
            return None
        
        # 2. Post-SL recovery window: extra pause after a stopout
        since_sl = now_ts - self._last_sl_time
        if since_sl < self.post_sl_cooldown_seconds:
            self.last_rejection_reason = f"Post-SL cooldown ({since_sl/3600:.1f}h < {self.post_sl_cooldown_seconds/3600:.1f}h)"
            return None
        
        # 3. Daily trade cap: reset counter on new calendar day
        today_str = market_data.timestamp.strftime("%Y-%m-%d")
        if today_str != self._last_trade_date:
            self._daily_trade_count = 0
            self._last_trade_date = today_str
        if self._daily_trade_count >= self.max_trades_per_day:
            self.last_rejection_reason = f"Daily trade cap reached ({self._daily_trade_count}/{self.max_trades_per_day})"
            return None

        # ── REGIME & TREND IDENTIFICATION ──
        adx_vals = market_data.m15_candles.get_indicator("adx_14")
        if len(adx_vals) == 0 or np.isnan(adx_vals[-1]):
            return None
            
        adx = adx_vals[-1]
        is_trending = adx >= self.adx_trend_threshold
        self._current_regime = "TREND" if is_trending else "RANGE"
        
        # 4. ATR Spike Filter: block entries during extreme volatility explosions.
        # On days like April 10 2026 (Trump tariff gold crash), ATR spikes 3x+
        # making sweep entries extremely likely to get overwhelmed by the trend.
        m5_atr_vals = m5.get_indicator("atr_14")
        if len(m5_atr_vals) >= 31:
            current_atr = m5_atr_vals[-1]
            avg_atr_30 = np.mean(m5_atr_vals[-31:-1])
            if avg_atr_30 > 0 and not np.isnan(current_atr):
                atr_spike_mult = current_atr / avg_atr_30
                if atr_spike_mult > self.max_atr_spike_mult:
                    self.last_rejection_reason = f"ATR Spike Block ({atr_spike_mult:.1f}x > {self.max_atr_spike_mult}x avg)"
                    return None

        m15_trend = self.get_ema_trend(market_data.m15_candles)
        
        # ── RANGE COMPUTATION ──
        scaler = self.get_regime_scaler(market_data)
        dynamic_lookback = int(self.lookback * (1.5 if scaler >= 1.5 else 1.0))
        if len(m5) < dynamic_lookback + 1:
            return None
            
        prev_range = m5[-dynamic_lookback-1:-1]
        r_high = np.max(prev_range.high)
        r_low = np.min(prev_range.low)

        last = m5[-1]
        price = market_data.current_price
        m5_range = last.high - last.low
        if m5_range == 0:
            return None
            
        # Wick Rejection Math (Hammer / Shooting Star definition)
        # Ratio of how far the close is from the extreme low/high of the candle
        close_from_low_ratio = (last.close - last.low) / m5_range
        close_from_high_ratio = (last.high - last.close) / m5_range

        session_mult = self.session_multipliers.get(market_data.session, {"rej_boost": 0, "conf_boost": 0})
        effective_rej_thresh = self.rejection_thresh - session_mult["rej_boost"] # Lower threshold = easier entry

        # Volatility Climax -> Less stringent rejection body needed
        h1 = market_data.htf_candles[-1]
        h1_v = market_data.htf_candles.v
        h1_v_avg = np.mean(h1_v[-21:-1]) if len(h1_v) >= 21 else 1
        vol_climax = h1.tick_volume > (h1_v_avg * 1.2)
        if vol_climax:
            effective_rej_thresh *= 0.9

        conf_floor = self.get_session_confidence_floor(market_data)

        # =================================================================
        # BUY SWEEP (LIQUIDITY GRAB BELOW RANGE + HAMMER REJECTION)
        # =================================================================
        swept_low = last.low < r_low
        closed_inside_from_low = last.close > r_low
        bullish_rejection = close_from_low_ratio >= effective_rej_thresh and last.close >= last.open
        
        if swept_low and closed_inside_from_low and bullish_rejection:
            # Regime Filter: If trending hard downward, sweeping the low is a falling knife. Block it.
            if is_trending and m15_trend == -1:
                self.last_rejection_reason = "Sweep: Blocked by Bearish Macro Trend"
                return None
                
            conf = min(0.98, 0.82 + session_mult["conf_boost"] + (0.05 if vol_climax else 0))
            if conf >= conf_floor:
                self._last_signal_time = now_ts  # FIX: timestamp-based cooldown
                self._daily_trade_count += 1
                sig = TradeSignal(direction="BUY", price=price, confidence=conf)
                sig.reasons.append("MECHANIC:SWEEP")
                sig.reasons.append(f"TARGET:{r_high}") 
                sig.reasons.append(f"PROTECT:{last.low}")
                return sig

        # =================================================================
        # SELL SWEEP (LIQUIDITY GRAB ABOVE RANGE + SHOOTING STAR REJECTION)
        # =================================================================
        swept_high = last.high > r_high
        closed_inside_from_high = last.close < r_high
        bearish_rejection = close_from_high_ratio >= effective_rej_thresh and last.close <= last.open
        
        if swept_high and closed_inside_from_high and bearish_rejection:
            # Regime Filter: If trending hard upward, sweeping the high is a runaway rally. Block it.
            if is_trending and m15_trend == 1:
                self.last_rejection_reason = "Sweep: Blocked by Bullish Macro Trend (SELL)"
                return None
                
            conf = min(0.98, 0.82 + session_mult["conf_boost"] + (0.05 if vol_climax else 0))
            if conf >= conf_floor:
                self._last_signal_time = now_ts  # FIX: timestamp-based cooldown
                self._daily_trade_count += 1
                sig = TradeSignal(direction="SELL", price=price, confidence=conf)
                sig.reasons.append("MECHANIC:SWEEP")
                sig.reasons.append(f"TARGET:{r_low}") 
                sig.reasons.append(f"PROTECT:{last.high}")
                return sig

        # Detailed rejection tracking for logs
        if swept_low and not bullish_rejection: self.last_rejection_reason = f"Sweep Low: Weak Hook ({close_from_low_ratio:.2f})"
        elif swept_high and not bearish_rejection: self.last_rejection_reason = f"Sweep High: Weak Hook ({close_from_high_ratio:.2f})"
        elif last.close > r_high or last.close < r_low: self.last_rejection_reason = "Breakout occurred (Ignored)"
        else: self.last_rejection_reason = "Inside Range (No Sweep)"

        return None

    # ==================================================================
    # DYNAMIC SL/TP ENGINE (Built for Traps)
    # ==================================================================
    def get_stop_loss(self, signal: TradeSignal, market_data: MarketData) -> float:
        m15_atr = self._get_m15_atr(market_data)
        
        protect_str = [r for r in signal.reasons if r.startswith("PROTECT:")]
        if protect_str:
            structural_extreme = float(protect_str[0].split(":")[1])
            # Structural placement just outside the sweep wick + 0.3 ATR buffer
            buffer = m15_atr * 0.3
            max_sl_dist = m15_atr * 1.5 # Cap the risk if the sweep wick was absolutely massive
            
            if signal.direction == "BUY":
                ideal_sl = structural_extreme - buffer
                max_sl = market_data.current_price - max_sl_dist
                return max(ideal_sl, max_sl) # Take the tighter one mathematically
            else:
                ideal_sl = structural_extreme + buffer
                max_sl = market_data.current_price + max_sl_dist
                return min(ideal_sl, max_sl)
                
        # Safe fallback
        return market_data.current_price - (m15_atr * 1.5) if signal.direction == "BUY" else market_data.current_price + (m15_atr * 1.5)

    def get_take_profit(self, signal: TradeSignal, market_data: MarketData) -> float:
        m15_atr = self._get_m15_atr(market_data)
        
        target_str = [r for r in signal.reasons if r.startswith("TARGET:")]
        if target_str:
            target_edge = float(target_str[0].split(":")[1])
            min_tp_dist = m15_atr * 1.5 # Guarantee a baseline 1.5:1 R:R floor minimum
            
            if signal.direction == "BUY":
                min_tp = market_data.current_price + min_tp_dist
                return max(target_edge, min_tp) # Target the far edge, or the minimum distance
            else:
                min_tp = market_data.current_price - min_tp_dist
                return min(target_edge, min_tp)
                
        # Safe fallback
        return market_data.current_price + (m15_atr * 1.5) if signal.direction == "BUY" else market_data.current_price - (m15_atr * 1.5)

    def on_trade_closed(self, trade_record: dict) -> None:
        """Called by the backtester after every trade close. Records SL hits for post-SL cooldown."""
        if trade_record.get("result", "").upper() == "SL":
            self._last_sl_time = float(trade_record.get("exit_time", 0.0))

    def reset_daily_stats(self) -> None:
        """Called by the backtester at the start of each calendar day."""
        self._daily_trade_count = 0

    # ==================================================================
    # Dashboard Metrics
    # ==================================================================
    def get_metrics(self, market_data: MarketData) -> Dict[str, Any]:
        if not market_data.htf_candles or len(market_data.htf_candles) < 22:
            return {}
            
        adx_vals = market_data.m15_candles.get_indicator("adx_14")
        adx = adx_vals[-1] if len(adx_vals) > 0 and not np.isnan(adx_vals[-1]) else 0.0
        regime = "TREND" if adx >= self.adx_trend_threshold else "RANGE"
        
        m15_trend = self.get_ema_trend(market_data.m15_candles)
        trend_label = {1: "BULL", -1: "BEAR", 0: "FLAT"}.get(m15_trend, "?")
        
        m5 = market_data.m5_candles
        last = m5[-1] if len(m5) > 0 else None
        cr = 0
        if last and (last.high - last.low) > 0:
            cr = (last.close - last.low) / (last.high - last.low)
        
        return {
            "Regime": f"{regime} ({adx:.1f})",
            "M15 Trend": trend_label,
            "M15 ATR": f"{self._get_m15_atr(market_data):.1f}", 
            "M5 Hook": f"{cr:.2f}"
        }

    def get_thresholds(self) -> Dict[str, Any]:
        return {
            "Regime": f"ADX <> {self.adx_trend_threshold}",
            "M15 Trend": "Filter",
            "M15 ATR": "Vol Tracker",
            "M5 Hook": f"> {self.rejection_thresh:.2f}"
        }
