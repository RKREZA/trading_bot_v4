import logging
import time
import random
from typing import Optional
from core.common.types import TradeSignal

logger = logging.getLogger("trading_bot.execution")


class ExecutionEngine:
    """
    Institutional Execution Layer.
    Simulates spread, slippage, and execution latency with optional deterministic RNG.
    """

    def __init__(self, config: dict):
        self.config = config
        exe_cfg = config.get("execution", {}) if isinstance(config.get("execution", {}), dict) else {}
        bt_cfg = config.get("backtest", {}) if isinstance(config.get("backtest", {}), dict) else {}

        self.latency_ms = int(exe_cfg.get("latency_ms", 150))
        self.max_spread_points = float(exe_cfg.get("max_spread_points", 50.0))

        base_slip = float(exe_cfg.get("slippage_points", 0.5))
        self.entry_slippage_points = float(exe_cfg.get("entry_slippage_points", base_slip))
        self.tp_exit_slippage_points = float(exe_cfg.get("tp_exit_slippage_points", max(0.0, base_slip * 0.5)))
        self.sl_exit_slippage_points = float(exe_cfg.get("sl_exit_slippage_points", max(0.0, base_slip * 1.25)))
        self.forced_exit_slippage_points = float(exe_cfg.get("forced_exit_slippage_points", base_slip))

        self.deterministic = bool(bt_cfg.get("deterministic", False) or exe_cfg.get("deterministic", False))
        self.random_seed = exe_cfg.get("random_seed", bt_cfg.get("random_seed"))

        from core.news_filter import InstitutionalNewsFilter
        self.news_filter = InstitutionalNewsFilter(config)

        self._rng = random.Random()
        self.reset_rng(self.random_seed)

    def reset_rng(self, seed: Optional[int] = None):
        if seed is None:
            self._rng.seed()
        else:
            self._rng.seed(int(seed))

    def execute_order(
        self,
        signal: TradeSignal,
        symbol: str,
        current_price: float,
        spread: float,
        point: float,
        timestamp: float = None,
    ) -> Optional[dict]:
        if signal.direction == "NONE":
            return None

        spread_points = spread / point if point > 0 else 0
        if spread_points > self.max_spread_points:
            logger.warning(f"EX_ENGINE: Spread too high ({spread_points:.1f} points). Rejecting signal.")
            return None

        # Institutional Guard: News Filter (Step 15 Refinement)
        ts = timestamp if timestamp else time.time()
        blocking_event = self.news_filter.is_blocked(symbol, ts)
        if blocking_event:
            logger.warning(f"[NEWS BLOCK] Trade rejected for {symbol} due to {blocking_event}")
            return None

        entry_slip = self.sample_slippage_points(point, event="entry")
        fill_price = current_price + entry_slip if signal.direction == "BUY" else current_price - entry_slip

        logger.info(
            f"EX_ENGINE: Order Filled. {signal.direction} {symbol} @ {fill_price:.5f} "
            f"(Entry Slippage: {entry_slip / point:.2f} points)"
        )

        return {
            "symbol": symbol,
            "direction": signal.direction,
            "fill_price": fill_price,
            "sl": signal.stop_loss,
            "tp": signal.take_profit,
            "timestamp": ts,
            "latency_ms": self.latency_ms,
            "entry_slippage_points": entry_slip / point if point > 0 else 0.0,
        }

    def sample_slippage_points(self, point: float, event: str = "entry") -> float:
        if point <= 0:
            return 0.0

        if event == "tp_exit":
            slippage_points = self.tp_exit_slippage_points
        elif event == "sl_exit":
            slippage_points = self.sl_exit_slippage_points
        elif event == "forced_exit":
            slippage_points = self.forced_exit_slippage_points
        else:
            slippage_points = self.entry_slippage_points

        return self._rng.uniform(0.0, max(0.0, slippage_points)) * point
