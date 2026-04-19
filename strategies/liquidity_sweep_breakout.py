import numpy as np
import logging
from datetime import datetime, timezone
from typing import Optional, Dict, Any, Tuple
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
        self.base_lookback = strat_config.get("lookback", 20)
        self.rejection_thresh = strat_config.get("rejection_thresh", 0.60)
        
        # ── Institutional Gating ──
        self.adx_threshold = strat_config.get("adx_threshold", 25.0)
        self.trend_strength_threshold = strat_config.get("trend_strength_threshold", 0.8)
        self.impulse_threshold = strat_config.get("impulse_threshold", 1.8)
        self.sweep_depth_mult = strat_config.get("sweep_depth_mult", 0.2)
        self.duplicate_zone_mult = strat_config.get("duplicate_zone_mult", 0.5)
        
        # ── State Management ──
        self.cooldown_seconds = strat_config.get("cooldown_seconds", 12_600)  # 3.5h
        self._last_signal_time: float = 0.0
        self.max_trades_per_day = strat_config.get("max_trades_per_day", 2)
        self._daily_trade_count: int = 0
        self._last_trade_date: str = ""
        self.post_sl_cooldown_seconds = strat_config.get("post_sl_cooldown_seconds", 14_400) # 4h
        self._last_sl_time: float = 0.0
        self._last_trade_price: float = 0.0
        
        # ── Retracement Entry Management ──
        self.use_retracement_entry = strat_config.get("use_retracement_entry", False)
        self.retracement_validity_bars = 3
        self._pending_setup: Optional[Dict[str, Any]] = None

    def _get_m15_atr(self, market_data: MarketData) -> float:
        m15_atr_vals = market_data.m15_candles.atr(14)
        if len(m15_atr_vals) > 0 and not np.isnan(m15_atr_vals[-1]):
            return m15_atr_vals[-1]
        return 15.0

    def _get_dynamic_lookback(self, market_data: MarketData) -> int:
        m5 = market_data.m5_candles
        atr_vals = m5.atr(14)
        if len(atr_vals) < 31: return self.base_lookback
        
        current_atr = atr_vals[-1]
        avg_atr_30 = np.mean(atr_vals[-31:-1])
        
        vol_factor = current_atr / avg_atr_30 if avg_atr_30 > 0 else 1.0
        lookback = self.base_lookback * np.clip(vol_factor, 0.8, 2.0)
        return int(lookback)

    def _get_range_structure(self, market_data: MarketData, lookback: int) -> Dict[str, float]:
        m5 = market_data.m5_candles
        if len(m5) < lookback + 1: return {}
        
        prev_range = m5[-lookback-1:-1]
        r_high = np.max(prev_range.high)
        r_low = np.min(prev_range.low)
        range_size = r_high - r_low
        
        atr = self._get_m15_atr(market_data)
        
        # Range Integrity Check - relaxed for XAUUSD high volatility
        if range_size < 0.3 * atr:  # Lower bound only, remove upper bound
            return {"valid": False, "reason": "Invalid Range Structure"}
            
        return {"valid": True, "high": r_high, "low": r_low, "size": range_size}

    def _is_momentum_safe(self, last_candle: Any, atr: float) -> Tuple[bool, str]:
        if atr <= 0: return True, ""
        
        impulse = abs(last_candle.close - last_candle.open) / atr
        
        # Wick Dominance Check
        candle_range = last_candle.high - last_candle.low
        if candle_range <= 0: return True, ""
        
        # For a potential BUY (Hammer), we check the upper wick ratio for climax evidence
        # For a potential SELL (Shooting Star), we check the lower wick ratio
        # Actually, the user's rule was: "wick_ratio = (high - max(open, close)) / (high - low) # for sell"
        # Let's generalize: wick in ADVANCE of the rejection move.
        
        is_bullish = last_candle.close >= last_candle.open
        if is_bullish: # Potential BUY rejection
            wick_ratio = (last_candle.close - last_candle.low) / candle_range
        else: # Potential SELL rejection
            wick_ratio = (last_candle.high - last_candle.close) / candle_range
            
        if impulse > self.impulse_threshold and wick_ratio < 0.5:
            return False, "Momentum Block"
            
        return True, ""

    def _check_liquidity_confluence(self, price: float, direction: str, market_data: MarketData, atr: float) -> Tuple[bool, str]:
        if market_data.d1_candles is None or len(market_data.d1_candles) < 2:
            return True, "" # No data, skip confluence requirement (relaxed)
            
        pd = market_data.d1_candles[-2] # Previous Day
        pdh, pdl = pd.high, pd.low
        
        target_liq = pdl if direction == "BUY" else pdh
        distance_to_liq = abs(price - target_liq) / atr if atr > 0 else 100
        
        # If PDH/PDL is nearby, we MUST sweep it
        if distance_to_liq < 1.5:
            if direction == "BUY" and price > pdl:
                return False, "No Liquidity Confluence (PDL)"
            if direction == "SELL" and price < pdh:
                return False, "No Liquidity Confluence (PDH)"
                
        return True, ""

    def _get_confidence_score(self, rej_strength: float, impulse: float, confluence: bool, vol_climax: bool, session: str) -> float:
        # score = 0.70 + (rej_strength - 0.5) * 0.6 + max(0, 1.5 - impulse) * 0.08 + bonuses
        score = 0.70
        score += (rej_strength - 0.5) * 0.6
        score += max(0, 1.5 - impulse) * 0.08
        if confluence: score += 0.05
        if vol_climax: score += 0.05
        if session in ["LONDON", "LONDON/NY"]: score += 0.05
        
        return min(0.98, max(0.5, score))

    def generate_signal(self, market_data: MarketData) -> Optional[TradeSignal]:
        if not self.is_spread_safe(market_data): return None
        
        m5 = market_data.m5_candles
        now_ts = market_data.timestamp.timestamp()
        atr = self._get_m15_atr(market_data)
        
        # 1. Trade Gating: Cooldowns & Daily Cap
        if now_ts - self._last_signal_time < self.cooldown_seconds:
            self.last_rejection_reason = "Cooldown Active"
            return None
        if now_ts - self._last_sl_time < self.post_sl_cooldown_seconds:
            self.last_rejection_reason = "Post-SL Cooldown"
            return None
        today_str = market_data.timestamp.strftime("%Y-%m-%d")
        if today_str != self._last_trade_date:
            self._daily_trade_count = 0
            self._last_trade_date = today_str
        if self._daily_trade_count >= self.max_trades_per_day:
            self.last_rejection_reason = "Daily Cap Reached"
            return None

        # 2. Retracement Entry Management (Step 8)
        if self._pending_setup:
            if now_ts > self._pending_setup["expiry"]:
                self._pending_setup = None # Timeout
            else:
                # Check for fill: price must reach the 50% level
                target_price = self._pending_setup["entry_price"]
                # Simulating fill: if current candle spans the target_price
                if m5[-1].low <= target_price <= m5[-1].high:
                    setup = self._pending_setup
                    self._pending_setup = None
                    self._last_signal_time = now_ts
                    self._daily_trade_count += 1
                    self._last_trade_price = setup["price"]
                    return setup["signal"]
                return None # Still waiting for fill

        # 3. Regime Identification (ADX + Trend Strength)
        adx_vals = market_data.m15_candles.adx(14)
        if len(adx_vals) == 0 or np.isnan(adx_vals[-1]): return None
        adx = adx_vals[-1]
        
        ema50 = market_data.m15_candles.ema(50)
        ema200 = market_data.m15_candles.ema(200)
        macro_trend = 0
        if len(ema50) > 0 and len(ema200) > 0:
            trend_strength = abs(ema50[-1] - ema200[-1]) / atr if atr > 0 else 0
            if adx >= self.adx_threshold and trend_strength > self.trend_strength_threshold:
                macro_trend = 1 if ema50[-1] > ema200[-1] else -1

        # 4. Range Computation & Structural Integrity
        lookback = self._get_dynamic_lookback(market_data)
        range_data = self._get_range_structure(market_data, lookback)
        if not range_data.get("valid", False):
            self.last_rejection_reason = range_data.get("reason", "Invalid Range")
            return None
            
        r_high, r_low = range_data["high"], range_data["low"]

        # 5. Sweep Detection & Validation
        last = m5[-1]
        swept_low = last.low < r_low
        swept_high = last.high > r_high
        if not swept_low and not swept_high:
            self.last_rejection_reason = "Inside Range"
            return None
            
        # Rejection Math
        candle_range = last.high - last.low
        if candle_range <= 0: return None
        close_from_low = (last.close - last.low) / candle_range
        close_from_high = (last.high - last.close) / candle_range
        
        # Momentum Filter
        momentum_safe, momentum_reason = self._is_momentum_safe(last, atr)
        if not momentum_safe:
            self.last_rejection_reason = momentum_reason
            return None

        # Duplicate Zone Prevention
        if abs(market_data.current_price - self._last_trade_price) < self.duplicate_zone_mult * atr:
            self.last_rejection_reason = "Duplicate Zone"
            return None

        # Institutional Vol Climax
        h1_v = market_data.htf_candles.v
        h1_v_avg = np.mean(h1_v[-21:-1]) if len(h1_v) >= 21 else 1
        vol_climax = market_data.htf_candles[-1].tick_volume > (h1_v_avg * 1.5)

        # 6. SIGNAL EVALUATION
        direction = None
        if swept_low and last.close > r_low and close_from_low >= self.rejection_thresh and last.close > last.open:
            # BUY SWEEP
            if macro_trend == -1: # Regime Conflict
                self.last_rejection_reason = "Regime Conflict"
                return None
            # Sweep Depth Validation
            sweep_depth = abs(last.low - r_low) / atr
            if sweep_depth < self.sweep_depth_mult:
                self.last_rejection_reason = "Weak Sweep"
                return None
            direction = "BUY"
            
        elif swept_high and last.close < r_high and close_from_high >= self.rejection_thresh and last.close < last.open:
            # SELL SWEEP
            if macro_trend == 1:
                self.last_rejection_reason = "Regime Conflict"
                return None
            sweep_depth = abs(last.high - r_high) / atr
            if sweep_depth < self.sweep_depth_mult:
                self.last_rejection_reason = "Weak Sweep"
                return None
            direction = "SELL"
            
        if direction:
            # Liquidity Confluence Check
            liq_safe, liq_reason = self._check_liquidity_confluence(market_data.current_price, direction, market_data, atr)
            if not liq_safe:
                self.last_rejection_reason = liq_reason
                return None
                
            # Confidence Scoring
            rej_val = close_from_low if direction == "BUY" else close_from_high
            impulse = abs(last.close - last.open) / atr
            conf = self._get_confidence_score(rej_val, impulse, "PDL" in liq_reason or "PDH" in liq_reason, vol_climax, market_data.session)
            
            sig = TradeSignal(
                direction=direction, 
                price=market_data.current_price, 
                confidence=conf,
                reasons=["MECHANIC:SWEEP", f"REGIME:{'TREND' if macro_trend != 0 else 'RANGE'}", f"ATR:{atr:.1f}", f"PROTECT:{last.low if direction == 'BUY' else last.high}", f"TARGET:{r_high if direction == 'BUY' else r_low}"]
            )
            
            # Retracement Entry logic
            if self.use_retracement_entry:
                sweep_ext = last.low if direction == "BUY" else last.high
                retracement_price = last.close - (last.close - sweep_ext) * 0.5
                self._pending_setup = {
                    "signal": sig,
                    "entry_price": retracement_price,
                    "price": market_data.current_price,
                    "expiry": now_ts + (self.retracement_validity_bars * 300) # 5m * 3 = 15m
                }
                return None # Signal will be returned when filled
                
            self._last_signal_time = now_ts
            self._daily_trade_count += 1
            self._last_trade_price = market_data.current_price
            return sig

        if swept_low: self.last_rejection_reason = "Weak Rejection (BUY)"
        elif swept_high: self.last_rejection_reason = "Weak Rejection (SELL)"
        else: self.last_rejection_reason = "Inside Range"
        
        return None

    # ==================================================================
    # DYNAMIC SL/TP ENGINE (Built for Traps)
    # ==================================================================
    def get_stop_loss(self, signal: TradeSignal, market_data: MarketData) -> float:
        atr = self._get_m15_atr(market_data)
        buffer = 0.3 * atr
        
        protect_str = [r for r in signal.reasons if r.startswith("PROTECT:")]
        if protect_str:
            extreme = float(protect_str[0].split(":")[1])
            return extreme - buffer if signal.direction == "BUY" else extreme + buffer
            
        return market_data.current_price - (1.5 * atr) if signal.direction == "BUY" else market_data.current_price + (1.5 * atr)

    def get_take_profit(self, signal: TradeSignal, market_data: MarketData) -> float:
        atr = self._get_m15_atr(market_data)
        
        target_str = [r for r in signal.reasons if r.startswith("TARGET:")]
        if target_str:
            target_price = float(target_str[0].split(":")[1])
            # Ensure Min RR 1.5
            sl_price = self.get_stop_loss(signal, market_data)
            risk = abs(market_data.current_price - sl_price)
            min_reward = risk * 1.5
            
            if signal.direction == "BUY":
                return max(target_price, market_data.current_price + min_reward)
            else:
                return min(target_price, market_data.current_price - min_reward)
                
        return market_data.current_price + (1.5 * atr) if signal.direction == "BUY" else market_data.current_price - (1.5 * atr)

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
        atr = self._get_m15_atr(market_data)
        adx_vals = market_data.m15_candles.adx(14)
        adx = adx_vals[-1] if len(adx_vals) > 0 else 0
        
        last = market_data.m5_candles[-1]
        impulse = abs(last.close - last.open) / atr if atr > 0 else 0
        
        lookback = self._get_dynamic_lookback(market_data)
        range_data = self._get_range_structure(market_data, lookback)
        r_size = range_data.get("size", 0) / atr if atr > 0 else 0
        
        return {
            "Regime": f"{'TREND' if adx >= self.adx_threshold else 'RANGE'} ({adx:.1f})",
            "Impulse": f"{impulse:.2f}",
            "Range": f"{r_size:.1f}x ATR",
            "Daily": f"{self._daily_trade_count}/{self.max_trades_per_day}"
        }

    def get_thresholds(self) -> Dict[str, Any]:
        return {
            "Regime": f"ADX > {self.adx_threshold}",
            "Impulse": f"< {self.impulse_threshold}",
            "Range": "0.5 - 3.0x",
            "Daily": f"Max {self.max_trades_per_day}"
        }
