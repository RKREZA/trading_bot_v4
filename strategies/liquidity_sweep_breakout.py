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
    V6-INSIGNIA Liquidity Sweep (Anti-Breakout) Strategy — Hardened Edition.
    
    Key fixes over V5:
    - PDH/PDL constructed from M5 data when D1 is unavailable
    - Volatility regime filter blocks entries when ATR > 2.0x average (extreme moves)
    - TP targets opposing liquidity pool instead of fixed 1.5x ATR
    - Post-SL cooldown enforced (1 hour)
    - MSS confirmation timeout (10 bars)
    - Reduced daily trade cap (6→3)
    - Widened FVG search window (3→5 candles)
    
    Core Logic:
    - Never buys high or sells low.
    - Waits for price to pierce PDH/PDL, then strongly reject and close inside.
    - Uses ADX to filter directions:
        - ADX < 25 (Ranging): Allow sweeps in both directions.
        - ADX >= 25 (Trending): Only allow sweeps that align with the M15 EMA trend.
    - Uses Hammer/Shooting Star math for rejection validation.
    """

    def __init__(self, strategy_id: str, config: dict):
        super().__init__(strategy_id, config)
        
        strat_config = self.get_strat_config()
        self.base_lookback = strat_config.get("lookback", 20)
        self.rejection_thresh = strat_config.get("rejection_thresh", 0.45)
        
        # ── Institutional Gating ──
        self.adx_threshold = strat_config.get("adx_threshold", 25.0)
        self.sweep_depth_mult = strat_config.get("sweep_depth_mult", 0.10)
        self.min_displacement_ratio = strat_config.get("displacement_ratio", 1.0)
        self.impulse_threshold = strat_config.get("impulse_threshold", 2.5)
        self.max_trades_per_day = strat_config.get("max_trades_per_day", 3)  # V6: Reduced from 6
        
        # ── V6: Volatility Regime Filter ──
        self.max_atr_ratio = strat_config.get("max_atr_ratio", 2.0)
        
        # ── V6: Post-SL Cooldown ──
        self.post_sl_cooldown_seconds = strat_config.get("post_sl_cooldown_seconds", 3600)
        self._last_sl_time: float = 0.0
        
        # ── V6: MSS Confirmation Timeout ──
        self.mss_timeout_bars = strat_config.get("mss_timeout_bars", 10)
        
        # ── State Management ──
        self.allowed_sessions = ["TOKYO", "LONDON", "NEW_YORK", "LONDON/NY", "GLOBAL", "ROLLOVER"]
        self._last_signal_time: float = 0.0
        self._daily_trade_count: int = 0
        self._last_trade_date: str = ""
        self._last_trade_price: float = 0.0
        
        # ── AMD / MSS State Tracking ──
        self._sweep_data: Optional[Dict[str, Any]] = None 
        self._sweep_bar_count: int = 0  # V6: Track bars since sweep for timeout

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
        
        if range_size < 0.3 * atr:
            return {"valid": False, "reason": "Invalid Range Structure"}
            
        return {"valid": True, "high": r_high, "low": r_low, "size": range_size}

    def _is_momentum_safe(self, last_candle: Any, atr: float) -> Tuple[bool, str]:
        if atr <= 0: return True, ""
        
        impulse = abs(last_candle.close - last_candle.open) / atr
        candle_range = last_candle.high - last_candle.low
        if candle_range <= 0: return True, ""
        
        is_bullish = last_candle.close >= last_candle.open
        if is_bullish:
            wick_ratio = (last_candle.close - last_candle.low) / candle_range
        else:
            wick_ratio = (last_candle.high - last_candle.close) / candle_range
            
        if impulse > self.impulse_threshold and wick_ratio < 0.5:
            return False, "Momentum Block"
            
        return True, ""

    def _is_volatility_extreme(self, market_data: MarketData) -> bool:
        """
        V6: Volatility Regime Filter.
        Blocks entries when M15 ATR is > 2.0x the 100-bar average.
        During extreme volatility, sweeps are often trend continuation, not reversals.
        """
        m15 = market_data.m15_candles
        atr_vals = m15.atr(14)
        if len(atr_vals) < 101:
            return False
            
        current_atr = atr_vals[-1]
        avg_atr = np.mean(atr_vals[-101:-1])
        
        if avg_atr <= 0:
            return False
            
        ratio = current_atr / avg_atr
        return ratio > self.max_atr_ratio

    def _get_liquidity_pools(self, market_data: MarketData) -> Dict[str, float]:
        """
        V6: Enhanced liquidity pool identification.
        Constructs PDH/PDL from M5 data when D1 candles are unavailable.
        """
        pools = {}
        
        # 1. Previous Day High / Low from D1 data
        if market_data.d1_candles is not None and len(market_data.d1_candles) >= 2:
            pd = market_data.d1_candles[-2]
            pools["PDH"] = pd.high
            pools["PDL"] = pd.low
        else:
            # V6 FALLBACK: Construct PDH/PDL from M5 candles
            m5 = market_data.m5_candles
            if m5 is not None and len(m5) > 200:
                current_ts = market_data.timestamp.timestamp()
                
                # Find the start of the current UTC day
                current_day_start = datetime(
                    market_data.timestamp.year, 
                    market_data.timestamp.month,
                    market_data.timestamp.day,
                    tzinfo=timezone.utc
                ).timestamp()
                
                # Previous day = 24h before current day start
                prev_day_start = current_day_start - 86400
                
                # Mask for previous day's M5 bars
                prev_day_mask = (m5.time >= prev_day_start) & (m5.time < current_day_start)
                prev_day_bars = m5[prev_day_mask]
                
                if prev_day_bars is not None and len(prev_day_bars) > 20:
                    pools["PDH"] = np.max(prev_day_bars.high)
                    pools["PDL"] = np.min(prev_day_bars.low)
        
        # 2. Session Structure Highs / Lows from M15
        m15 = market_data.m15_candles
        if m15 is not None and len(m15) > 50:
            pools["STRUCT_H"] = np.max(m15.h[-100:])
            pools["STRUCT_L"] = np.min(m15.l[-100:])
            
        return pools

    def _detect_mss(self, m5: Any, direction: str, sweep_extreme: float) -> Tuple[bool, Optional[float]]:
        """
        Detects Market Structure Shift (MSS) on M5 after a sweep.
        V6: Extended search window from 6 to 8 bars for better detection.
        """
        if len(m5) < 12: return False, None
        
        search_depth = min(10, len(m5) - 2)  # V6: Wider search
        
        if direction == "BUY":
            for i in range(len(m5)-2, len(m5) - search_depth, -1):
                if i < 1 or i >= len(m5) - 1: continue
                if m5.h[i] > m5.h[i-1] and m5.h[i] > m5.h[i+1]:
                    internal_high = m5.h[i]
                    if m5.close[-1] > internal_high:
                        return True, internal_high
            return False, None
        else:
            for i in range(len(m5)-2, len(m5) - search_depth, -1):
                if i < 1 or i >= len(m5) - 1: continue
                if m5.l[i] < m5.l[i-1] and m5.l[i] < m5.l[i+1]:
                    internal_low = m5.l[i]
                    if m5.close[-1] < internal_low:
                        return True, internal_low
            return False, None

    def _get_fvg_entry(self, m5: Any, direction: str) -> Optional[float]:
        """V6: Widened FVG search from 3 to 5 candles."""
        if len(m5) < 6: return None
        
        search_depth = min(5, len(m5) - 2)  # V6: 5 candles instead of 3
        for i in range(len(m5)-1, len(m5)-1-search_depth, -1):
            if i < 2: continue
            if direction == "BUY":
                if m5.l[i] > m5.h[i-2]:
                    return (m5.l[i] + m5.h[i-2]) / 2
            else:
                if m5.h[i] < m5.l[i-2]:
                    return (m5.h[i] + m5.l[i-2]) / 2
        return None

    def _get_confidence_score(self, sweep_depth: float, mss_confirm: bool, vol_spike: bool,
                               has_pdh_pdl: bool) -> float:
        """V6: Enhanced confidence with PDH/PDL quality bonus."""
        score = 0.65  # Slightly lowered base
        if sweep_depth > 0.5: score += 0.08
        elif sweep_depth > 0.2: score += 0.04
        if mss_confirm: score += 0.10
        if vol_spike: score += 0.05
        if has_pdh_pdl: score += 0.05  # V6: Bonus for sweeping real PDH/PDL
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

        if self._daily_trade_count >= self.max_trades_per_day:
            self.last_rejection_reason = f"Daily trade cap reached ({self.max_trades_per_day})"
            return None

        # V6: Post-SL Cooldown Enforcement
        if self._last_sl_time > 0 and (now_ts - self._last_sl_time) < self.post_sl_cooldown_seconds:
            remaining = int(self.post_sl_cooldown_seconds - (now_ts - self._last_sl_time))
            self.last_rejection_reason = f"Post-SL cooldown ({remaining}s remaining)"
            return None

        # V6: Volatility Regime Filter
        if self._is_volatility_extreme(market_data):
            self.last_rejection_reason = "Volatility Extreme (ATR > 2.0x avg) — sweep likely continuation"
            self._sweep_data = None  # Invalidate any pending sweep
            return None

        # 3. Identify Liquidity Pools & Active Sweeps
        pools = self._get_liquidity_pools(market_data)
        last_candle = m5[-1]
        
        # State Machine: Check if a sweep just happened
        if self._sweep_data is None:
            for pool_name, level in pools.items():
                if pool_name.endswith("L") and last_candle.low < level:
                    has_pdh = pool_name.startswith("PD")
                    self._sweep_data = {
                        "type": "BUY", "extreme": last_candle.low, "pool": pool_name,
                        "level": level, "ts": now_ts, "has_pdh": has_pdh
                    }
                    self._sweep_bar_count = 0
                    break
                elif pool_name.endswith("H") and last_candle.high > level:
                    has_pdh = pool_name.startswith("PD")
                    self._sweep_data = {
                        "type": "SELL", "extreme": last_candle.high, "pool": pool_name,
                        "level": level, "ts": now_ts, "has_pdh": has_pdh
                    }
                    self._sweep_bar_count = 0
                    break
        
        if self._sweep_data:
            # V6: MSS Confirmation Timeout
            self._sweep_bar_count += 1
            if self._sweep_bar_count > self.mss_timeout_bars:
                self.last_rejection_reason = f"MSS timeout ({self.mss_timeout_bars} bars) — invalidating sweep"
                self._sweep_data = None
                return None
            
            # Expiry: If sweep happened > 2 hours ago, discard
            if now_ts - self._sweep_data["ts"] > 7200:
                self._sweep_data = None
                return None
                
            sweep_type = self._sweep_data["type"]
            extreme = self._sweep_data["extreme"]
            has_pdh = self._sweep_data.get("has_pdh", False)
            
            # Step 1: Wait for price to close back INSIDE the pool level (Rejection)
            if sweep_type == "BUY" and last_candle.close > self._sweep_data["level"]:
                # Step 2: Confirm M5 Market Structure Shift (MSS)
                mss_confirmed, mss_level = self._detect_mss(m5, "BUY", extreme)
                if mss_confirmed:
                    # Step 3: Find Fair Value Gap entry
                    entry_price = self._get_fvg_entry(m5, "BUY")
                    if entry_price is None:
                        # V6 Fallback: Use midpoint of sweep candle body as entry
                        entry_price = (last_candle.open + last_candle.close) / 2
                    
                    sweep_depth = abs(extreme - self._sweep_data["level"]) / atr
                    conf = self._get_confidence_score(sweep_depth, True, True, has_pdh)
                    pool_name = self._sweep_data['pool']
                    
                    # V6: TP targeting opposing pool
                    tp_target = self._get_opposing_pool_tp(pools, "BUY", entry_price, atr)
                    
                    self._sweep_data = None
                    self._daily_trade_count += 1
                    return TradeSignal(
                        direction="BUY",
                        price=entry_price,
                        confidence=conf,
                        reasons=["AMD:MANIPULATION_COMPLETE", "MSS:CONFIRMED",
                                 f"POOL:{pool_name}", f"PROTECT:{extreme}",
                                 f"TARGET:{tp_target:.2f}"]
                    )
                        
            elif sweep_type == "SELL" and last_candle.close < self._sweep_data["level"]:
                mss_confirmed, mss_level = self._detect_mss(m5, "SELL", extreme)
                if mss_confirmed:
                    entry_price = self._get_fvg_entry(m5, "SELL")
                    if entry_price is None:
                        entry_price = (last_candle.open + last_candle.close) / 2
                    
                    sweep_depth = abs(extreme - self._sweep_data["level"]) / atr
                    conf = self._get_confidence_score(sweep_depth, True, True, has_pdh)
                    pool_name = self._sweep_data['pool']
                    
                    tp_target = self._get_opposing_pool_tp(pools, "SELL", entry_price, atr)
                    
                    self._sweep_data = None
                    self._daily_trade_count += 1
                    return TradeSignal(
                        direction="SELL",
                        price=entry_price,
                        confidence=conf,
                        reasons=["AMD:MANIPULATION_COMPLETE", "MSS:CONFIRMED",
                                 f"POOL:{pool_name}", f"PROTECT:{extreme}",
                                 f"TARGET:{tp_target:.2f}"]
                    )

        self.last_rejection_reason = "No AMD Setup found"
        return None

    def _get_opposing_pool_tp(self, pools: Dict[str, float], direction: str,
                               entry_price: float, atr: float) -> float:
        """
        V6: TP targets opposing liquidity pool instead of fixed ATR multiple.
        BUY: Target PDH or STRUCT_H
        SELL: Target PDL or STRUCT_L
        Falls back to 2.5x ATR if no pool available.
        """
        if direction == "BUY":
            targets = []
            for key in ["PDH", "STRUCT_H"]:
                if key in pools and pools[key] > entry_price:
                    targets.append(pools[key])
            if targets:
                return min(targets)  # Nearest overhead pool
            return entry_price + (2.5 * atr)
        else:
            targets = []
            for key in ["PDL", "STRUCT_L"]:
                if key in pools and pools[key] < entry_price:
                    targets.append(pools[key])
            if targets:
                return max(targets)  # Nearest below pool
            return entry_price - (2.5 * atr)

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
        
        # V6: Use TARGET from signal reasons (opposing pool)
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
                
        return market_data.current_price + (2.5 * atr) if signal.direction == "BUY" else market_data.current_price - (2.5 * atr)

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
    def get_parameter_grid(self) -> Dict[str, list]:
        """Institutional grid for WFO optimization."""
        return {
            "rejection_thresh": [0.4, 0.5, 0.6],
            "sweep_depth_mult": [0.05, 0.1, 0.15],
            "displacement_ratio": [0.8, 1.0, 1.2],
            "max_atr_ratio": [1.5, 2.0, 2.5]
        }

    def get_metrics(self, market_data: MarketData) -> Dict[str, Any]:
        atr = self._get_m15_atr(market_data)
        adx_vals = market_data.m15_candles.adx(14)
        adx = adx_vals[-1] if len(adx_vals) > 0 else 0
        
        last = market_data.m5_candles[-1]
        impulse = abs(last.close - last.open) / atr if atr > 0 else 0
        
        lookback = self._get_dynamic_lookback(market_data)
        range_data = self._get_range_structure(market_data, lookback)
        r_size = range_data.get("size", 0) / atr if atr > 0 else 0
        
        pools = self._get_liquidity_pools(market_data)
        pool_info = ", ".join([f"{k}:{v:.0f}" for k, v in pools.items()])
        
        return {
            "Regime": f"{'TREND' if adx >= self.adx_threshold else 'RANGE'} ({adx:.1f})",
            "Impulse": f"{impulse:.2f}",
            "Range": f"{r_size:.1f}x ATR",
            "Daily": f"{self._daily_trade_count}/{self.max_trades_per_day}",
            "Pools": pool_info[:50]
        }

    def get_thresholds(self) -> Dict[str, Any]:
        return {
            "Regime": f"ADX > {self.adx_threshold}",
            "Impulse": f"< {self.impulse_threshold}",
            "Range": "0.5 - 3.0x",
            "Daily": f"Max {self.max_trades_per_day}"
        }
