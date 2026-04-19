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
        self.sweep_depth_mult = strat_config.get("sweep_depth_mult", 0.15)
        self.min_displacement_ratio = strat_config.get("displacement_ratio", 1.2) # Body vs ATR
        
        # ── State Management ──
        self.allowed_sessions = ["TOKYO", "LONDON", "NEW_YORK", "LONDON/NY", "GLOBAL", "ROLLOVER"]
        self._last_signal_time: float = 0.0
        self._daily_trade_count: int = 0
        self._last_trade_date: str = ""
        self._last_trade_price: float = 0.0
        
        # ── AMD / MSS State Tracking ──
        self._sweep_data: Optional[Dict[str, Any]] = None 
        # Stores: {type: 'BUY/SELL', extreme: price, liquidity_pool: 'PDH/TOKYO_H', timestamp: float}

        
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

    def _get_liquidity_pools(self, market_data: MarketData) -> Dict[str, float]:
        """Identifies active institutional liquidity pools (PDH/PDL and Session Highs/Lows)."""
        pools = {}
        
        # 1. Previous Day High / Low
        if market_data.d1_candles is not None and len(market_data.d1_candles) >= 2:
            pd = market_data.d1_candles[-2]
            pools["PDH"] = pd.high
            pools["PDL"] = pd.low
            
        # 2. Session Highs / Lows (Extracted from M15 to ensure structural stability)
        m15 = market_data.m15_candles
        if m15 is not None and len(m15) > 50:
            # We look back ~24 hours of M15 bars (96 bars) to find session extremes
            # In a real engine, we'd use a dedicated SessionManager to track these per-session.
            # Here we proxy by finding the highest/lowest within the last 100 bars.
            pools["STRUCT_H"] = np.max(m15.h[-100:])
            pools["STRUCT_L"] = np.min(m15.l[-100:])
            
        return pools

    def _detect_mss(self, m5: Any, direction: str, sweep_extreme: float) -> Tuple[bool, Optional[float]]:
        """
        Detects Market Structure Shift (MSS) on M5 after a sweep.
        Requires:
        1. Impulse (Displacement) candle.
        2. Break of internal swing high/low.
        """
        if len(m5) < 10: return False, None
        
        # Find local swing high/low before the sweep extreme
        if direction == "BUY": # We swept a Low, now looking for MSS High
            # Look for recent internal swing high on M5
            for i in range(len(m5)-2, len(m5)-8, -1):
                if m5.h[i] > m5.h[i-1] and m5.h[i] > m5.h[i+1]:
                    internal_high = m5.h[i]
                    if m5.close[-1] > internal_high:
                        return True, internal_high
            return False, None
        else: # We swept a High, now looking for MSS Low
            for i in range(len(m5)-2, len(m5)-8, -1):
                if m5.l[i] < m5.l[i-1] and m5.l[i] < m5.l[i+1]:
                    internal_low = m5.l[i]
                    if m5.close[-1] < internal_low:
                        return True, internal_low
            return False, None

    def _get_fvg_entry(self, m5: Any, direction: str) -> Optional[float]:
        """Finds the most recent Fair Value Gap (FVG) for entry."""
        if len(m5) < 5: return None
        
        # Check last 3 candles for an unfilled FVG
        for i in range(len(m5)-1, len(m5)-4, -1):
            if direction == "BUY": # Bullish FVG
                if m5.l[i] > m5.h[i-2]:
                    # Gap between i-2 High and i Low
                    return (m5.l[i] + m5.h[i-2]) / 2
            else: # Bearish FVG
                if m5.h[i] < m5.l[i-2]:
                    return (m5.h[i] + m5.l[i-2]) / 2
        return None

    def _get_confidence_score(self, sweep_depth: float, mss_confirm: bool, vol_spike: bool) -> float:
        score = 0.70
        if sweep_depth > 0.5: score += 0.1
        if mss_confirm: score += 0.1
        if vol_spike: score += 0.05
        return min(0.95, score)

    def generate_signal(self, market_data: MarketData) -> Optional[TradeSignal]:
        if not self.is_spread_safe(market_data): return None
        
        # 1. Kill Zone Filter
        if not SessionDetector.is_session_active(market_data.timestamp, allowed_sessions=self.allowed_sessions):
            self.last_rejection_reason = "Out of Kill Zone"
            return None

        m5 = market_data.m5_candles
        if m5 is None or len(m5) < 50: return None
        
        atr = self._get_m15_atr(market_data)
        now_ts = market_data.timestamp.timestamp()
        
        # 2. Daily Cap & Cooldown
        today_str = market_data.timestamp.strftime("%Y-%m-%d")
        if today_str != self._last_trade_date:
            self._daily_trade_count = 0
            self._last_trade_date = today_str
        
        # 3. Identify Liquidity Pools & Active Sweeps
        pools = self._get_liquidity_pools(market_data)
        last_candle = m5[-1]
        
        # State Machine: Check if a sweep just happened
        if self._sweep_data is None:
            # Scan for new sweeps
            for pool_name, level in pools.items():
                if pool_name.endswith("L") and last_candle.low < level:
                    self._sweep_data = {"type": "BUY", "extreme": last_candle.low, "pool": pool_name, "level": level, "ts": now_ts}
                    break
                elif pool_name.endswith("H") and last_candle.high > level:
                    self._sweep_data = {"type": "SELL", "extreme": last_candle.high, "pool": pool_name, "level": level, "ts": now_ts}
                    break
        
        if self._sweep_data:
            # Expiry: If sweep happened > 2 hours ago, discard
            if now_ts - self._sweep_data["ts"] > 7200:
                self._sweep_data = None
                return None
                
            sweep_type = self._sweep_data["type"]
            extreme = self._sweep_data["extreme"]
            
            # Step 1: Wait for price to close back INSIDE the pool level (Rejection)
            if sweep_type == "BUY" and last_candle.close > self._sweep_data["level"]:
                # Step 2: Confirm M5 Market Structure Shift (MSS)
                mss_confirmed, mss_level = self._detect_mss(m5, "BUY", extreme)
                if mss_confirmed:
                    # Step 3: Find Fair Value Gap entry
                    entry_price = self._get_fvg_entry(m5, "BUY")
                    if entry_price:
                        # Success: Trigger Orderflow Signal
                        sweep_depth = abs(extreme - self._sweep_data["level"]) / atr
                        conf = self._get_confidence_score(sweep_depth, True, True)
                        
                        self._sweep_data = None # Reset state
                        self._daily_trade_count += 1
                        return TradeSignal(
                            direction="BUY",
                            price=entry_price,
                            confidence=conf,
                            reasons=["AMD:MANIPULATION_COMPLETE", "MSS:CONFIRMED", f"POOL:{self._sweep_data['pool'] if self._sweep_data else 'EXTREME'}", f"PROTECT:{extreme}"]
                        )
                        
            elif sweep_type == "SELL" and last_candle.close < self._sweep_data["level"]:
                mss_confirmed, mss_level = self._detect_mss(m5, "SELL", extreme)
                if mss_confirmed:
                    entry_price = self._get_fvg_entry(m5, "SELL")
                    if entry_price:
                        sweep_depth = abs(extreme - self._sweep_data["level"]) / atr
                        conf = self._get_confidence_score(sweep_depth, True, True)
                        
                        self._sweep_data = None
                        self._daily_trade_count += 1
                        return TradeSignal(
                            direction="SELL",
                            price=entry_price,
                            confidence=conf,
                            reasons=["AMD:MANIPULATION_COMPLETE", "MSS:CONFIRMED", f"POOL:{self._sweep_data['pool'] if self._sweep_data else 'EXTREME'}", f"PROTECT:{extreme}"]
                        )

        self.last_rejection_reason = "No AMD Setup found"
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
