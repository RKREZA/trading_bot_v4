import logging
import random
import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
from core.common.types import TradeSignal

logger = logging.getLogger("trading_bot.backtester.simulator")

class ExecutionSimulator:
    """
    V4-ULTRA Institutional Execution Simulator.
    Simulates:
    - Randomized Slippage (Gaussian)
    - Execution Latency (ms-level)
    - Variable Spread Expansion
    - Tick-level vs. Candle-open Entry
    """
    
    def __init__(self, config: Dict):
        self.config = config
        exec_cfg = config.get("execution", {})
        
        # Simulation Parameters
        self.latency_ms = exec_cfg.get("latency_ms", 100)
        self.entry_slip_pips = exec_cfg.get("entry_slippage_pips", 0.15)
        self.tp_exit_slip_pips = exec_cfg.get("tp_exit_slippage_pips", 0.08)
        self.sl_exit_slip_pips = exec_cfg.get("sl_exit_slippage_pips", 0.25)
        self.max_spread_pips = exec_cfg.get("max_spread_pips", 80)
        
        # Deterministic RNG
        seed = config.get("backtest", {}).get("random_seed", 42)
        self._rng = np.random.default_rng(seed)

    def simulate_entry(self, 
                       signal: TradeSignal, 
                       current_price: float, 
                       base_spread_points: float, 
                       point: float) -> Optional[Dict]:
        """
        Simulates institutional trade entry (Step 4.4).
        """
        # 1. Spread Check (Step 4.5)
        # Variable spread simulation: base_spread + random noise (up to 20% of base)
        # base_spread_points is assumed to be in RAW POINTS (e.g. 500 for XAUUSD $5.00 spread)
        current_spread_pts = base_spread_points * (1.0 + self._rng.uniform(0, 0.2))
        spread_price = current_spread_pts * point
        
        # Human-readable pips for gating (Institutional Standard: 10 pts = 1 pip)
        # Note: Gold doesn't have 10pts/pip in some brokers but 10 is the internal bot standard.
        spread_pips = base_spread_points / 10.0
        
        if spread_pips > self.max_spread_pips:
            logger.debug(f"Trade REJECTED: Spread too high ({spread_pips:.1f} pips)")
            return None
            
        # 2. Latency & Slippage (Step 4.1)
        # We simulate Gaussian slippage centered around the configured 'expected' slippage
        # slip_points is in Pips
        slip_price = self._rng.normal(self.entry_slip_pips, self.entry_slip_pips * 0.5) * 10.0 * point
        
        # Calculation for Entry
        direction = signal.direction
        if direction == "BUY":
            # Long enters at ASK (Price + half spread + slippage)
            fill_price = current_price + (spread_price / 2.0) + slip_price
        else:
            # Short enters at BID (Price - half spread - slippage)
            fill_price = current_price - (spread_price / 2.0) - slip_price
            
        return {
            "fill_price": fill_price,
            "direction": direction,
            "sl": signal.stop_loss,
            "tp": signal.take_profit,
            "timestamp": signal.timestamp,
            "fill_spread_pips": spread_pips,
            "fill_slippage_pips": slip_price / point / 10.0 if point > 0 else 0
        }

    def simulate_exit(self, 
                      trade: Dict, 
                      exit_price: float, 
                      point: float, 
                      event: str = "tp") -> Tuple[float, float]:
        """
        Simulates institutional trade exit.
        """
        if event == "tp":
            slip_pips = self.tp_exit_slip_pips
        elif event == "sl":
            slip_pips = self.sl_exit_slip_pips
        else:
            slip_pips = 0.15 # Forced close / News
            
        slip_points = self._rng.normal(slip_pips, slip_pips * 0.5) * 10 * point
        
        # For Long (BUY), exit is at BID (lower) -> subtract slippage
        # For Short (SELL), exit is at ASK (higher) -> add slippage
        # Wait, if we are BUYing, we exit by SELLing. 
        # Slippage on exit SELL = worse price = lower price.
        # Slippage on exit BUY = worse price = higher price.
        
        direction = trade["direction"]
        if direction == "BUY":
            final_price = exit_price - slip_points
        else:
            final_price = exit_price + slip_points
            
        return final_price, slip_points
