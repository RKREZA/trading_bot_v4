import logging
import time
import random
from typing import Optional
from core.types import TradeSignal

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
        self.max_spread_pips = float(exe_cfg.get("max_spread_pips", 5.0))

        base_slip = float(exe_cfg.get("slippage_pips", 0.1))
        self.entry_slippage_pips = float(exe_cfg.get("entry_slippage_pips", base_slip))
        self.tp_exit_slippage_pips = float(exe_cfg.get("tp_exit_slippage_pips", max(0.0, base_slip * 0.5)))
        self.sl_exit_slippage_pips = float(exe_cfg.get("sl_exit_slippage_pips", max(0.0, base_slip * 1.25)))
        self.forced_exit_slippage_pips = float(exe_cfg.get("forced_exit_slippage_pips", base_slip))

        self.deterministic = bool(bt_cfg.get("deterministic", False) or exe_cfg.get("deterministic", False))
        self.random_seed = exe_cfg.get("random_seed", bt_cfg.get("random_seed"))

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

        spread_pips = spread / point if point > 0 else 0
        if spread_pips > self.max_spread_pips:
            logger.warning(f"EX_ENGINE: Spread too high ({spread_pips:.1f} pips). Rejecting signal.")
            return None

        if self._is_news_blocked():
            logger.warning("EX_ENGINE: Trading blocked due to high-impact news.")
            return None

        entry_slippage_points = self.sample_slippage_points(point, event="entry")
        fill_price = current_price + entry_slippage_points if signal.direction == "BUY" else current_price - entry_slippage_points

        logger.info(
            f"EX_ENGINE: Order Filled. {signal.direction} {symbol} @ {fill_price:.5f} "
            f"(Entry Slippage: {entry_slippage_points / point:.2f} pips)"
        )

        return {
            "symbol": symbol,
            "direction": signal.direction,
            "fill_price": fill_price,
            "sl": signal.stop_loss,
            "tp": signal.take_profit,
            "timestamp": timestamp if timestamp else time.time(),
            "latency_ms": self.latency_ms,
            "entry_slippage_pips": entry_slippage_points / point if point > 0 else 0.0,
        }

    def sample_slippage_points(self, point: float, event: str = "entry") -> float:
        if point <= 0:
            return 0.0

        if event == "tp_exit":
            slippage_pips = self.tp_exit_slippage_pips
        elif event == "sl_exit":
            slippage_pips = self.sl_exit_slippage_pips
        elif event == "forced_exit":
            slippage_pips = self.forced_exit_slippage_pips
        else:
            slippage_pips = self.entry_slippage_pips

        return self._rng.uniform(0.0, max(0.0, slippage_pips)) * point

    def _is_news_blocked(self) -> bool:
        return False
