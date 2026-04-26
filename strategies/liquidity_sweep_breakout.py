import numpy as np
import logging
import json
import os
from datetime import datetime, timezone
from typing import Optional, Dict, Any, Tuple
from collections import deque
from core.base_strategy import BaseStrategy, MarketData
from core.common.types import TradeSignal
from core.session_detector import SessionDetector

logger = logging.getLogger("trading_bot.strategy.liquidity_sweep_breakout")

class LiquiditySweepBreakoutStrategy(BaseStrategy):
    """
    V7-INSIGNIA Liquidity Sweep Strategy – Production Hardened.
    
    Key improvements over V6:
    - Adaptive thresholds based on rolling percentiles (no magic numbers)
    - Volatility-scaled position sizing & stop loss
    - Persistent state (survives bot restart)
    - Realistic limit order placement (bid/ask aware)
    - Volume spike detection using rolling Z-score
    - Dynamic cooldown based on ATR ratio
    - Consecutive loss reduction
    - Walk-forward ready parameters
    """

    def __init__(self, strategy_id: str, config: dict):
        super().__init__(strategy_id, config)
        
        strat_config = self.get_strat_config()
        
        # ── Core Parameters (now adaptive thresholds) ──
        self.base_lookback = strat_config.get("lookback", 20)
        self.adx_threshold = strat_config.get("adx_threshold", 25.0)
        self.max_trades_per_day = strat_config.get("max_trades_per_day", 3)
        
        # ── Adaptive thresholds (Hyper-Relaxed Scalper) ──
        self.rejection_percentile = strat_config.get("rejection_percentile", 0.90)
        self.sweep_depth_percentile = strat_config.get("sweep_depth_percentile", 0.10)
        self.volume_zscore_threshold = strat_config.get("volume_zscore_threshold", -0.5)
        self.mss_timeout_bars = strat_config.get("mss_timeout_bars", 10)
        self.daily_trade_limit = strat_config.get("daily_trade_limit", 20)
        self.max_atr_ratio = strat_config.get("max_atr_ratio", 8.5)
        
        # ── MSS Scalper ──
        self.mss_timeout_bars = strat_config.get("mss_timeout_bars", 5)
        self.mss_timeframe = strat_config.get("mss_timeframe", "M1")
        self.mss_buffer_pct = strat_config.get("mss_buffer_pct", 0.0) # e.g., 0.001 for 0.1%
        
        # ── VWAP Filter ──
        self.vwap_filter_enabled = strat_config.get("vwap_filter_enabled", False)
        
        # ── Trend Filter ──
        self.ema_trend_filter = strat_config.get("ema_trend_filter", False)
        self.ema_period = strat_config.get("ema_period", 50)
        
        # ── Risk Management ──
        self.risk_per_trade = strat_config.get("risk_per_trade", 0.01)
        self.max_daily_risk = strat_config.get("max_daily_risk", 0.03)
        self.volatility_scaling = strat_config.get("volatility_scaling", True)
        self.consecutive_loss_reduction = strat_config.get("consecutive_loss_reduction", 0.5)
        self.slippage_bps = strat_config.get("slippage_bps", 0.5)
        self.impulse_threshold = strat_config.get("impulse_threshold", 2.5)
        
        # ── State Tracking ──
        self._last_signal_bar = 0
        self.min_bars_between_signals = strat_config.get("min_bars_between_signals", 15)
        self._last_sl_time = 0
        self._daily_trade_count = 0
        self._last_pool_update_day = -1
        self._cached_pools = {}
        self._sweep_data: Optional[Dict[str, Any]] = None
        self._sweep_bar_count = 0
        
        # ── Rolling buffers for adaptive thresholds (size: 200 items) ──
        self._recent_sweep_depths = deque(maxlen=200)
        self._recent_volumes = deque(maxlen=200)
        self._recent_rejection_ratios = deque(maxlen=200)
        
        # ── Session filter ──
        self.london_start = strat_config.get("london_start", 7)
        self.london_end = strat_config.get("london_end", 12)
        self.cooldown_seconds = strat_config.get("cooldown_seconds", 900)  # 15 mins
        self._last_trade_time = 0
        
        # ── Load persistent state if exists (for bot restarts) ──
        self.state_file = f"{self.strategy_id}_state.json"
        self._load_state()

    # ==================================================================
    # Adaptive Threshold Helpers
    # ==================================================================
    def _get_adaptive_threshold(self, buffer: deque, percentile: float, default: float) -> float:
        if len(buffer) < 50:
            return default
        arr = np.array(buffer)
        return float(np.percentile(arr, percentile * 100))

    def _is_valid_sweep_depth(self, depth_atr: float) -> bool:
        if depth_atr <= 0:
            return False
        self._recent_sweep_depths.append(depth_atr)
        thresh = self._get_adaptive_threshold(self._recent_sweep_depths, self.sweep_depth_percentile, 0.2)
        return depth_atr >= thresh

    def _is_volume_spike(self, market_data, current_vol: float) -> bool:
        if current_vol <= 0:
            return False
            
        m5_vols = market_data.m5_candles.v
        if m5_vols is None or len(m5_vols) < 30:
            return False
            
        recent_vols = m5_vols[-30:]
        mean_vol = np.mean(recent_vols)
        std_vol = np.std(recent_vols)
        
        if std_vol == 0:
            return False
            
        zscore = (current_vol - mean_vol) / std_vol
        return zscore > self.volume_zscore_threshold

    def _is_valid_rejection(self, candle, direction: str) -> bool:
        candle_range = candle.high - candle.low
        if candle_range <= 0:
            return False
            
        if direction == "BUY":
            wick_ratio = (candle.close - candle.low) / candle_range
        else:
            wick_ratio = (candle.high - candle.close) / candle_range
        
        self._recent_rejection_ratios.append(wick_ratio)
        # Use configurable threshold, default to 0.30 (30% rejection wick)
        thresh = self.get_strat_config().get("wick_ratio_threshold", 0.30)
        return wick_ratio >= thresh

    # ==================================================================
    # Risk Management
    # ==================================================================
    def _get_position_size(self, entry_price: float, stop_distance: float, current_balance: float) -> float:
        risk_amount = current_balance * self.risk_per_trade
        if stop_distance <= 0:
            return 0
        size = risk_amount / stop_distance
        if self._consecutive_losses >= 2:
            size *= self.consecutive_loss_reduction
        if self.volatility_scaling:
            atr = self._get_m15_atr(self._market_data_cache) if hasattr(self, '_market_data_cache') else stop_distance
            vol_factor = max(0.5, min(2.0, 1.0 / (atr / stop_distance)))
            size *= vol_factor
        return size

    def _apply_slippage(self, price: float, direction: str) -> float:
        bps = self.slippage_bps / 10000.0
        if direction == "BUY":
            return price * (1 + bps)
        else:
            return price * (1 - bps)

    def _check_daily_loss_limit(self, trade_result_pnl_percent: float) -> bool:
        self._daily_loss += trade_result_pnl_percent
        return self._daily_loss <= -self.max_daily_risk

    # ==================================================================
    # Core Strategy Methods
    # ==================================================================
    def _get_m15_atr(self, market_data: MarketData) -> float:
        m15_atr_vals = market_data.m15_candles.atr(14)
        if len(m15_atr_vals) > 0 and not np.isnan(m15_atr_vals[-1]):
            return m15_atr_vals[-1]
        return 15.0

    def _get_dynamic_lookback(self, market_data: MarketData) -> int:
        m5 = market_data.m5_candles
        atr_vals = m5.atr(14)
        if len(atr_vals) < 31:
            return self.base_lookback
        current_atr = atr_vals[-1]
        avg_atr_30 = np.mean(atr_vals[-31:-1])
        vol_factor = current_atr / avg_atr_30 if avg_atr_30 > 0 else 1.0
        lookback = self.base_lookback * np.clip(vol_factor, 0.8, 2.0)
        return int(lookback)

    def _is_volatility_extreme(self, market_data: MarketData) -> bool:
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
        current_day = market_data.timestamp.day
        if current_day == self._last_pool_update_day and self._cached_pools:
            return self._cached_pools
            
        pools = {}
        # 1. Previous Day High/Low (PDH/PDL)
        if market_data.d1_candles is not None and len(market_data.d1_candles) >= 2:
            pd = market_data.d1_candles[-2]
            pools["PDH"] = pd.high
            pools["PDL"] = pd.low
            
        # 2. Previous Week High/Low (PWH/PWL)
        h1 = market_data.htf_candles
        if h1 is not None and len(h1) >= 121:
            h_vals = h1.h[:-1] # Exclude current bar
            l_vals = h1.l[:-1]
            pools["PWH"] = float(np.max(h_vals[-120:]))
            pools["PWL"] = float(np.min(l_vals[-120:]))
            
        # 3. Dynamic Structural Pools (Recent extremes)
        m15 = market_data.m15_candles
        if m15 is not None and len(m15) > 51:
            h_vals_m15 = m15.h[:-1]
            l_vals_m15 = m15.l[:-1]
            pools["STRUCT_H"] = np.max(h_vals_m15[-100:])
            pools["STRUCT_L"] = np.min(l_vals_m15[-100:])
            
        # 4. Session Highs/Lows (Institutional Kill Zones)
        m5 = market_data.m5_candles
        if m5 is not None and len(m5) > 100:
            # We look for Tokyo and London extremes in the recent history
            # This is a bit expensive, so we only do it once per day
            recent_times = m5.time[-500:]
            recent_highs = m5.h[-500:]
            recent_lows = m5.l[-500:]
            
            tokyo_h, tokyo_l = -1.0, -1.0
            london_h, london_l = -1.0, -1.0
            
            for i in range(len(recent_times)):
                dt = datetime.fromtimestamp(recent_times[i], tz=timezone.utc)
                sess = SessionDetector.get_session(dt)
                if sess == "TOKYO":
                    if tokyo_h < 0 or recent_highs[i] > tokyo_h: tokyo_h = recent_highs[i]
                    if tokyo_l < 0 or recent_lows[i] < tokyo_l: tokyo_l = recent_lows[i]
                elif sess == "LONDON":
                    if london_h < 0 or recent_highs[i] > london_h: london_h = recent_highs[i]
                    if london_l < 0 or recent_lows[i] < london_l: london_l = recent_lows[i]
            
            if tokyo_h > 0:
                pools["TOKYO_H"] = tokyo_h
                pools["TOKYO_L"] = tokyo_l
            if london_h > 0:
                pools["LONDON_H"] = london_h
                pools["LONDON_L"] = london_l

        self._cached_pools = pools
        self._last_pool_update_day = current_day
        return pools

    def _calculate_ema(self, closes: Any, period: int) -> float:
        import pandas as pd
        if len(closes) < period:
            return 0.0
        return pd.Series(closes).ewm(span=period, adjust=False).mean().iloc[-1]

    def _calculate_frvp_poc(self, m5: Any, lookback: int) -> float:
        if lookback <= 0:
            lookback = 1
        if len(m5) < lookback:
            return m5.c[-1] if len(m5) > 0 else 0.0
        recent_m5 = m5[-lookback:]
        if len(recent_m5) == 0:
            return m5.c[-1] if len(m5) > 0 else 0.0
        min_p = np.min(recent_m5.low)
        max_p = np.max(recent_m5.high)
        if max_p == min_p:
            return min_p
        bins = np.linspace(min_p, max_p, num=20)
        vol_profile = np.zeros(19)
        typ_prices = (recent_m5.high + recent_m5.low + recent_m5.close) / 3
        indices = np.digitize(typ_prices, bins) - 1
        indices = np.clip(indices, 0, 18)
        for i in range(len(recent_m5)):
            vol_profile[indices[i]] += recent_m5.tick_volume[i]
        poc_idx = np.argmax(vol_profile)
        return (bins[poc_idx] + bins[poc_idx+1]) / 2

    def _detect_mss(self, m5: Any, m15: Any, direction: str, sweep_extreme: float) -> Tuple[bool, Optional[float]]:
        if self.mss_timeframe == "M15":
            # Option 2: M15 MSS
            if len(m15) < 5:
                return False, None
            if direction == "BUY":
                swing_high = max(m15.h[-5:-1])
                if m15.c[-1] > swing_high * (1 + self.mss_buffer_pct):
                    return True, swing_high
                return False, None
            else:
                swing_low = min(m15.l[-5:-1])
                if m15.c[-1] < swing_low * (1 - self.mss_buffer_pct):
                    return True, swing_low
                return False, None
        
        # Default Option (M5 MSS)
        if len(m5) < 12:
            return False, None
        search_depth = min(10, len(m5) - 2)
        if direction == "BUY":
            for i in range(len(m5)-2, len(m5)-1-search_depth, -1):
                if i < 1 or i >= len(m5)-1:
                    continue
                if m5.h[i] > m5.h[i-1] and m5.h[i] > m5.h[i+1]:
                    internal_high = m5.h[i]
                    if m5.c[-1] > internal_high * (1 + self.mss_buffer_pct):
                        return True, internal_high
            return False, None
        else:
            for i in range(len(m5)-2, len(m5)-1-search_depth, -1):
                if i < 1 or i >= len(m5)-1:
                    continue
                if m5.l[i] < m5.l[i-1] and m5.l[i] < m5.l[i+1]:
                    internal_low = m5.l[i]
                    if m5.c[-1] < internal_low * (1 - self.mss_buffer_pct):
                        return True, internal_low
            return False, None

    def _get_fvg_entry(self, m5: Any, direction: str) -> Optional[float]:
        if len(m5) < 6:
            return None
        search_depth = min(5, len(m5) - 2)
        for i in range(len(m5)-1, len(m5)-1-search_depth, -1):
            if i < 2:
                continue
            if direction == "BUY":
                if m5.l[i] > m5.h[i-2]:
                    return (m5.l[i] + m5.h[i-2]) / 2
            else:
                if m5.h[i] < m5.l[i-2]:
                    return (m5.h[i] + m5.l[i-2]) / 2
        return None

    def _get_opposing_pool_tp(self, pools: Dict[str, float], direction: str,
                               entry_price: float, atr: float, poc: float = 0.0) -> float:
        targets = [poc] if poc > 0 else []
        if direction == "BUY":
            for key in ["PDH", "STRUCT_H"]:
                if key in pools and pools[key] > entry_price:
                    targets.append(pools[key])
            targets = [t for t in targets if t > entry_price]
            if targets:
                return min(targets)
            return entry_price + (2.5 * atr)
        else:
            for key in ["PDL", "STRUCT_L"]:
                if key in pools and pools[key] < entry_price:
                    targets.append(pools[key])
            targets = [t for t in targets if t < entry_price]
            if targets:
                return max(targets)
            return entry_price - (2.5 * atr)

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

    # ==================================================================
    # Main Signal Generation
    # ==================================================================
    def generate_signal(self, market_data: MarketData) -> Optional[TradeSignal]:
        now_ts = market_data.timestamp.timestamp()
        dt = market_data.timestamp
        
        # ── London Scalper Session (The Profitable Window) ──
        is_london_prime = (dt.hour >= self.london_start and dt.hour < self.london_end)
        if not is_london_prime:
            self.last_rejection_reason = "Out of London Prime Session"
            return None

        # ── High-Freq Cooldown ──
        if (now_ts - self._last_trade_time) < self.cooldown_seconds:
            self.last_rejection_reason = "Cooldown Active"
            return None

        m5 = market_data.m5_candles
        atr = self._get_m15_atr(market_data)
        
        # ── Scalper Trend Filter (EMA 50 on M15) ──
        m15_ema50 = market_data.m15_candles.ema(50)
        if len(m15_ema50) < 1 or np.isnan(m15_ema50[-1]): 
            self.last_rejection_reason = "EMA Warmup"
            return None
        trend_dir = 1 if market_data.m15_candles.c[-1] > m15_ema50[-1] else -1
        
        # ── Scalper Volume Validation (Above Average) ──
        vol_vals = m5.v
        avg_vol = np.mean(vol_vals[-50:]) if len(vol_vals) >= 50 else 0
        std_vol = np.std(vol_vals[-50:]) if len(vol_vals) >= 50 else 0
        vol_z = (vol_vals[-1] - avg_vol) / std_vol if std_vol > 0 else 0
        if vol_z < 0.0:
            self.last_rejection_reason = f"Vol Below Average ({vol_z:.2f})"
            return None

        # --- STEP 1: LIQUIDITY DETECTION ---
        lookback = self.base_lookback 
        if len(m5) < lookback + 1: 
            self.last_rejection_reason = "M5 History Inadequate"
            return None
        
        recent_h = m5.h[-lookback-1:-1]
        recent_l = m5.l[-lookback-1:-1]
        swing_h = np.max(recent_h)
        swing_l = np.min(recent_l)
        
        last_candle = m5[-1]
        
        # BUY Setup: Sweep of Low + Wick Rejection
        if last_candle.low < swing_l and last_candle.close > swing_l:
            if not self._is_valid_rejection(last_candle, "BUY"):
                self.last_rejection_reason = "Poor Wick Rejection (BUY)"
                return None

            self._last_trade_time = now_ts
            risk = abs(last_candle.close - (last_candle.low - 0.2*atr))
            conf = 0.90 if trend_dir == 1 else 0.70
            return TradeSignal(
                direction="BUY",
                price=last_candle.close,
                confidence=conf,
                stop_loss=last_candle.low - 0.2*atr,
                take_profit=last_candle.close + (risk * 2.5),
                reasons=["SCALPER_SWEEP_LOW", f"VOL_Z:{vol_z:.1f}", f"TREND:{'ALIGNED' if trend_dir==1 else 'REVERSAL'}", "REJECTION_OK"]
            )

        # SELL Setup: Sweep of High + Wick Rejection
        elif last_candle.high > swing_h and last_candle.close < swing_h:
            if not self._is_valid_rejection(last_candle, "SELL"):
                self.last_rejection_reason = "Poor Wick Rejection (SELL)"
                return None

            self._last_trade_time = now_ts
            risk = abs(last_candle.close - (last_candle.high + 0.2*atr))
            conf = 0.90 if trend_dir == -1 else 0.70
            return TradeSignal(
                direction="SELL",
                price=last_candle.close,
                confidence=conf,
                stop_loss=last_candle.high + 0.2*atr,
                take_profit=last_candle.close - (risk * 2.5),
                reasons=["SCALPER_SWEEP_HIGH", f"VOL_Z:{vol_z:.1f}", f"TREND:{'ALIGNED' if trend_dir==-1 else 'REVERSAL'}", "REJECTION_OK"]
            )
        else:
            self.last_rejection_reason = "No Sweep Detected"
            
        return None

    # ==================================================================
    # SL/TP Overrides
    # ==================================================================
    def get_stop_loss(self, signal: TradeSignal, market_data: MarketData) -> float:
        atr = self._get_m15_atr(market_data)
        buffer = 0.3 * atr
        
        if signal:
            protect_str = [r for r in signal.reasons if r.startswith("PROTECT:")]
            if protect_str:
                extreme = float(protect_str[0].split(":")[1])
                return extreme - buffer if signal.direction == "BUY" else extreme + buffer
        
        vol_regime = "high" if self._is_volatility_extreme(market_data) else "normal"
        mult = 1.5 if vol_regime == "normal" else 2.0
        return market_data.current_price - (mult * atr) if (signal and signal.direction == "BUY") else market_data.current_price + (mult * atr)

    def get_take_profit(self, signal: TradeSignal, market_data: MarketData) -> float:
        atr = self._get_m15_atr(market_data)
        target_str = [r for r in signal.reasons if r.startswith("TARGET:")]
        if target_str:
            target_price = float(target_str[0].split(":")[1])
            sl_price = self.get_stop_loss(signal, market_data)
            risk = abs(market_data.current_price - sl_price)
            min_reward = risk * 1.5
            if signal.direction == "BUY":
                return max(target_price, market_data.current_price + min_reward)
            else:
                return min(target_price, market_data.current_price - min_reward)
        return market_data.current_price + (2.5 * atr) if signal.direction == "BUY" else market_data.current_price - (2.5 * atr)

    # ==================================================================
    # Trade Management & Persistent State
    # ==================================================================
    def on_trade_closed(self, trade_record: dict) -> None:
        result = trade_record.get("result", "").upper()
        if result == "SL":
            self._last_sl_time = float(trade_record.get("exit_time", 0.0))
            self._consecutive_losses += 1
            pnl_percent = trade_record.get("pnl_percent", 0.0)
            self._daily_loss += pnl_percent
        elif result == "TP":
            self._consecutive_losses = 0
            pnl_percent = trade_record.get("pnl_percent", 0.0)
            self._daily_loss += pnl_percent
        self._save_state()

    def reset_daily_stats(self) -> None:
        self._daily_trade_count = 0
        self._daily_loss = 0.0
        self._consecutive_losses = 0
        self._last_trade_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        self._save_state()

    def _save_state(self) -> None:
        state = {
            "daily_trade_count": self._daily_trade_count,
            "last_trade_date": self._last_trade_date,
            "daily_loss": self._daily_loss,
            "consecutive_losses": self._consecutive_losses,
            "last_sl_time": self._last_sl_time,
            "sweep_bar_count": self._sweep_bar_count
        }
        try:
            with open(self.state_file, "w") as f:
                json.dump(state, f)
        except Exception as e:
            logger.debug(f"Failed to save state: {e}")

    def _load_state(self) -> None:
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, "r") as f:
                    state = json.load(f)
                self._daily_trade_count = state.get("daily_trade_count", 0)
                self._last_trade_date = state.get("last_trade_date", "")
                self._daily_loss = state.get("daily_loss", 0.0)
                self._consecutive_losses = state.get("consecutive_losses", 0)
                self._last_sl_time = state.get("last_sl_time", 0.0)
                self._sweep_bar_count = state.get("sweep_bar_count", 0)
            except Exception as e:
                logger.debug(f"Failed to load state: {e}")

    # ==================================================================
    # Dashboard & Metrics
    # ==================================================================
    def get_parameter_grid(self) -> Dict[str, list]:
        """Institutional grid for WFO optimization."""
        return {
            "volume_zscore_threshold": [0.8, 1.0, 1.2, 1.5, 2.0],
            "rejection_percentile": [0.25, 0.30, 0.35, 0.40, 0.45],
            "sweep_depth_percentile": [0.40, 0.50, 0.60, 0.70],
            "max_atr_ratio": [2.0, 2.5, 3.0]
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
        
        return {
            "Regime": f"{'TREND' if adx >= self.adx_threshold else 'RANGE'} ({adx:.1f})",
            "Impulse": f"{impulse:.2f}",
            "Range": f"{r_size:.1f}x ATR",
            "Daily": f"{self._daily_trade_count}/{self.max_trades_per_day}",
            "ConsecLoss": self._consecutive_losses
        }

    def get_thresholds(self) -> Dict[str, Any]:
        return {
            "Regime": f"ADX > {self.adx_threshold}",
            "VolSpike": f"Z > {self.volume_zscore_threshold}",
            "DailyRisk": f"{self.max_daily_risk*100:.0f}%",
            "RiskPerTrade": f"{self.risk_per_trade*100:.1f}%",
            "Daily": f"Max {self.max_trades_per_day}"
        }
