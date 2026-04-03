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

    def __init__(self, config: dict, connection: Any,
                 position_meta: Dict[int, Any], state_lock: threading.Lock):
        self.config        = config
        self.connection    = connection
        self.position_meta = position_meta
        self.state_lock    = state_lock

    def manage_positions(self, symbol: str, current_bid: float,
                         current_ask: float, atr: float,
                         last_candle: Optional[dict] = None):
        """
        Performs trailing stop logic for all active positions on a symbol.
        """
        if not self.config.get("trailing_stop_enabled", True):
            return

        positions = self.connection.get_positions(symbol)
        if not positions:
            return

        magic = self.config.get("magic_number", 234000)

        for pos in positions:
            if pos.magic != magic:
                continue

            ticket = pos.ticket
            with self.state_lock:
                if ticket not in self.position_meta:
                    self.position_meta[ticket] = {
                        "ticket":     ticket,
                        "best_price": pos.price_current,
                        "risk":       abs(pos.price_open - pos.sl) if pos.sl > 0 else 0,
                    }
                meta = self.position_meta[ticket]

            is_buy        = (pos.type == 0)
            current_price = current_bid if is_buy else current_ask

            with self.state_lock:
                if (is_buy  and current_price > meta["best_price"]) or \
                   (not is_buy and current_price < meta["best_price"]):
                    meta["best_price"] = current_price

            new_sl = self.calculate_new_sl(
                is_buy, pos.price_open, pos.sl,
                meta["best_price"], atr, meta["risk"],
                self.config, last_candle
            )

            if new_sl and new_sl != pos.sl:
                if (is_buy and new_sl > pos.sl) or \
                   (not is_buy and (pos.sl == 0 or new_sl < pos.sl)):
                    if self.connection.modify_sl_tp(ticket, symbol, new_sl, pos.tp):
                        rr = (abs(new_sl - pos.price_open) / meta["risk"]) if meta["risk"] > 0 else 0
                        logger.info(
                            "[Trailing] #%d → SL=%.2f (Phase at ~%.1fR)",
                            ticket, new_sl, rr
                        )

    @staticmethod
    def calculate_new_sl(is_buy: bool, entry: float, current_sl: float,
                         best_price: float, atr: float, risk: float,
                         config: dict, last_candle: Optional[dict] = None
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
        c_mult = ts.get("phase1_wider_mult", 4.0) if rr < p1_threshold \
                 else ts.get("phase1_tighter_mult", 2.5)
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
