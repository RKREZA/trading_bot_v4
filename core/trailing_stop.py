import logging
import threading
from typing import Dict, Any, Optional

logger = logging.getLogger("trading_bot.trailing_stop")


class TrailingStopManager:
    """
    Adaptive 5-Phase Trailing Stop — targets 1:5 to 1:15 R:R.

    Phase 1 (0 → 1.5R):   Wide Chandelier (ATR*4.0) — lets trade breathe
    Phase 2 (≥ 1.5R):     Break-even lock-in (+small offset)
    Phase 3 (≥ 3.0R):     Structural trail — prior candle low/high
    Phase 4 (≥ 5.0R):     Momentum runner — ATR*1.5 tight trail
    Phase 5 (≥ 10.0R):    Ultra-tight ATR*0.8 — milk the full run
    """

    def __init__(self, config: dict, connection: Any):
        self.config        = config
        self.connection    = connection

    @staticmethod
    def calculate_new_sl(is_buy: bool, entry: float, current_sl: float,
                          best_price: float, atr: float, risk: float,
                          config: dict, last_candle: Optional[dict] = None,
                          symbol: Optional[str] = None, session: Optional[str] = None
                          ) -> Optional[float]:
        """
        5-Phase adaptive trailing stop calculation.

        Returns the new SL price if it should be moved, else None.
        The SL can only ever improve (never widen).
        """
        if risk <= 0 or atr <= 0:
            return None

        profit_points = (best_price - entry) if is_buy else (entry - best_price)
        rr = profit_points / risk

        ts = config.get("trailing_stop", {})
        new_sl = current_sl

        # ── Phase 1: Wide Chandelier (0 → Phase2 threshold) ──────────────
        p1_threshold = ts.get("phase1_rr_threshold", 1.5)
        
        # Determine multiplier: Symbol Session -> Common Session -> Global TS
        c_mult = ts.get("phase1_wider_mult", 4.0) # Default
        
        if session:
            # Check symbol-specific override
            if symbol:
                sym_sessions = config.get("symbols_config", {}).get(symbol, {}).get("sessions", {})
                if session in sym_sessions:
                    c_mult = sym_sessions[session].get("trailing_atr_mult", c_mult)
                else:
                    # Fallback to global session config
                    c_mult = config.get("session_config", {}).get(session, {}).get("trailing_atr_mult", c_mult)
            else:
                # Use global session config directly
                c_mult = config.get("session_config", {}).get(session, {}).get("trailing_atr_mult", c_mult)

        if rr >= p1_threshold:
            c_mult = ts.get("phase1_tighter_mult", 2.5)
            
        chandelier = (best_price - atr * c_mult) if is_buy \
                     else (best_price + atr * c_mult)

        if (is_buy and chandelier > new_sl) or \
           (not is_buy and (new_sl == 0 or chandelier < new_sl)):
            new_sl = chandelier

        # ── Phase 2: Break-Even ───────────────────────────────────────────
        p2_threshold = ts.get("phase2_rr_threshold", 1.5)
        p2_offset    = ts.get("phase2_be_offset_pct", 0.1)
        if rr >= p2_threshold:
            be = entry + risk * p2_offset if is_buy \
                 else entry - risk * p2_offset
            if (is_buy and be > new_sl) or \
               (not is_buy and (new_sl == 0 or be < new_sl)):
                new_sl = be

        # ── Phase 3: Structural Lock-in (prior candle extreme) ───────────
        p3_threshold = ts.get("phase3_rr_threshold", 3.0)
        if rr >= p3_threshold and last_candle:
            structural = last_candle['low'] if is_buy else last_candle['high']
            if (is_buy and structural > new_sl) or \
               (not is_buy and (new_sl == 0 or structural < new_sl)):
                new_sl = structural

        # ── Phase 4: Momentum Runner Trail ───────────────────────────────
        # Activates at 5R+: tight chandelier (ATR*1.5) — rides big moves
        p4_threshold  = ts.get("phase4_rr_threshold", 5.0)
        p4_trail_mult = ts.get("phase4_trail_mult", 1.5)
        if rr >= p4_threshold:
            p4_sl = (best_price - atr * p4_trail_mult) if is_buy \
                    else (best_price + atr * p4_trail_mult)
            if (is_buy and p4_sl > new_sl) or \
               (not is_buy and (new_sl == 0 or p4_sl < new_sl)):
                new_sl = p4_sl

        # ── Phase 5: Ultra-Tight Milk Trail ──────────────────────────────
        # Activates at 10R+: ATR*0.8 — squeezes every last pip
        p5_threshold  = ts.get("phase5_rr_threshold", 10.0)
        p5_trail_mult = ts.get("phase5_trail_mult", 0.8)
        if rr >= p5_threshold:
            p5_sl = (best_price - atr * p5_trail_mult) if is_buy \
                    else (best_price + atr * p5_trail_mult)
            if (is_buy and p5_sl > new_sl) or \
               (not is_buy and (new_sl == 0 or p5_sl < new_sl)):
                new_sl = p5_sl

        return new_sl if new_sl != current_sl else None
