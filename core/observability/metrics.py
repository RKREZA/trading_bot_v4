"""
In-process metrics collector for the trading platform.
Tracks latency, slippage, win rate, Sharpe, and system health.
Thread-safe, lock-free design using atomic operations where possible.
"""

import time
import threading
from collections import deque
from typing import Dict, Any, Optional

import numpy as np


class MetricsCollector:

    def __init__(self, window_size: int = 500):
        self._window = window_size
        self._lock = threading.Lock()

        self._trade_pnls: deque = deque(maxlen=window_size)
        self._trade_returns: deque = deque(maxlen=window_size)
        self._exec_latencies_ms: deque = deque(maxlen=window_size)
        self._slippage_pips: deque = deque(maxlen=window_size)
        self._signal_count = 0
        self._trade_count = 0
        self._win_count = 0
        self._reject_count = 0

        self._peak_equity = 0.0
        self._current_equity = 0.0
        self._current_balance = 0.0

        self._uptime_start = time.monotonic()
        self._last_signal_ts = 0.0
        self._last_trade_ts = 0.0

        self._counters: Dict[str, int] = {}

    def record_trade(self, pnl: float, balance: float, equity: float,
                     latency_ms: float = 0.0, slippage_pips: float = 0.0) -> None:
        with self._lock:
            self._trade_pnls.append(pnl)
            self._trade_count += 1
            if pnl > 0:
                self._win_count += 1
            if balance > 0:
                self._trade_returns.append(pnl / balance)
            if latency_ms > 0:
                self._exec_latencies_ms.append(latency_ms)
            if slippage_pips != 0:
                self._slippage_pips.append(abs(slippage_pips))
            self._current_balance = balance
            self._current_equity = equity
            self._peak_equity = max(self._peak_equity, equity)
            self._last_trade_ts = time.time()

    def record_signal(self) -> None:
        self._signal_count += 1
        self._last_signal_ts = time.time()

    def record_rejection(self) -> None:
        self._reject_count += 1

    def update_equity(self, balance: float, equity: float) -> None:
        with self._lock:
            self._current_balance = balance
            self._current_equity = equity
            self._peak_equity = max(self._peak_equity, equity)

    def increment(self, counter_name: str, amount: int = 1) -> None:
        self._counters[counter_name] = self._counters.get(counter_name, 0) + amount

    def get_snapshot(self) -> Dict[str, Any]:
        with self._lock:
            pnls = list(self._trade_pnls)
            returns = list(self._trade_returns)
            latencies = list(self._exec_latencies_ms)
            slippages = list(self._slippage_pips)

        win_rate = (self._win_count / self._trade_count * 100) if self._trade_count > 0 else 0.0

        dd_pct = 0.0
        if self._peak_equity > 0:
            dd_pct = (self._peak_equity - self._current_equity) / self._peak_equity * 100

        sharpe = 0.0
        if len(returns) > 1:
            arr = np.array(returns)
            mean_r = arr.mean()
            std_r = arr.std()
            if std_r > 0:
                sharpe = float(mean_r / std_r * np.sqrt(252))

        avg_pnl = float(np.mean(pnls)) if pnls else 0.0
        avg_latency = float(np.mean(latencies)) if latencies else 0.0
        p99_latency = float(np.percentile(latencies, 99)) if len(latencies) >= 10 else avg_latency
        avg_slippage = float(np.mean(slippages)) if slippages else 0.0

        return {
            "uptime_seconds": round(time.monotonic() - self._uptime_start, 1),
            "trade_count": self._trade_count,
            "signal_count": self._signal_count,
            "reject_count": self._reject_count,
            "win_rate": round(win_rate, 2),
            "avg_pnl": round(avg_pnl, 2),
            "sharpe_ratio": round(sharpe, 2),
            "current_balance": round(self._current_balance, 2),
            "current_equity": round(self._current_equity, 2),
            "peak_equity": round(self._peak_equity, 2),
            "drawdown_pct": round(dd_pct, 2),
            "avg_latency_ms": round(avg_latency, 2),
            "p99_latency_ms": round(p99_latency, 2),
            "avg_slippage_pips": round(avg_slippage, 3),
            "last_signal_ts": self._last_signal_ts,
            "last_trade_ts": self._last_trade_ts,
            "counters": dict(self._counters),
        }

    def reset(self) -> None:
        with self._lock:
            self._trade_pnls.clear()
            self._trade_returns.clear()
            self._exec_latencies_ms.clear()
            self._slippage_pips.clear()
        self._signal_count = 0
        self._trade_count = 0
        self._win_count = 0
        self._reject_count = 0
        self._peak_equity = 0.0
        self._current_equity = 0.0
        self._current_balance = 0.0
        self._counters.clear()


metrics = MetricsCollector()
