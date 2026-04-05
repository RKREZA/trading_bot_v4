import logging
import time
import random
from typing import Optional, Dict, Any
from core.common.types import TradeSignal

class OrderManager:
    """
    Institutional Order Execution Manager.
    Simulates realistic market execution including latency, slippage, and spread validation.
    Independently runnable and testable.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        exe_cfg = config.get("execution", {})
        
        self.latency_ms = int(exe_cfg.get("latency_ms", 150))
        self.max_spread_pips = float(exe_cfg.get("max_spread_pips", 5.0))
        
        # Slippage Tiers
        base_slip = float(exe_cfg.get("slippage_pips", 0.1))
        self.entry_slip = float(exe_cfg.get("entry_slippage_pips", base_slip))
        self.tp_slip = float(exe_cfg.get("tp_exit_slippage_pips", base_slip * 0.5))
        self.sl_slip = float(exe_cfg.get("sl_exit_slippage_pips", base_slip * 1.5))
        
        # Deterministic RNG for reproducibility (Institutional requirement)
        self.deterministic = config.get("backtest", {}).get("deterministic", False)
        self.seed = config.get("backtest", {}).get("random_seed", 42)
        self._rng = random.Random(self.seed if self.deterministic else None)
        
        self.logger = logging.getLogger("trading_bot.execution")

    def execute_signal(self, 
                       signal: TradeSignal, 
                       symbol: str, 
                       price_data: Dict[str, float],
                       is_news_blocked: bool = False) -> Optional[Dict[str, Any]]:
        """
        Processes a TradeSignal with institutional realism (Spread, News, Latency).
        """
        if signal.direction == "NONE":
            return None

        # 1. News Blockade (Step 6.3)
        if is_news_blocked:
            self.logger.warning(f"Execution REJECTED: News Event Active for {symbol}")
            return None

        bid = price_data.get('bid')
        ask = price_data.get('ask')
        point = price_data.get('point', 0.00001)
        
        if not bid or not ask:
            self.logger.warning(f"Execution failed: Missing price data for {symbol}")
            return None

        # 2. Variable Spread Simulation (Step 6.1)
        base_spread_pips = (ask - bid) / point
        effective_spread = self._get_effective_spread(base_spread_pips)
        
        if effective_spread > self.max_spread_pips:
            self.logger.warning(f"Execution Rejected: Spread too high ({effective_spread:.1f} > {self.max_spread_pips})")
            return None

        # Update ask to reflect effective spread for fill calculation
        spread_diff = (effective_spread - base_spread_pips) * point
        ask_adj = ask + spread_diff
        bid_adj = bid - spread_diff # Optional: split it.

        # 3. Simulate Latency (Step 6.2)
        if not self.deterministic:
            time.sleep(self._rng.uniform(0.5, 1.5) * (self.latency_ms / 1000.0))

        # 4. Simulate Entry Slippage
        slip_points = self._sample_slippage(point, tier="entry")
        fill_price = ask_adj + slip_points if signal.direction == "BUY" else bid_adj - slip_points
        
        actual_slippage_pips = slip_points / point if point > 0 else 0
        
        self.logger.info(f"ORDER_EXECUTED: {signal.direction} {symbol} @ {fill_price:.5f} (Slip: {actual_slippage_pips:.2f} pips)")

        return {
            "ticket": self._rng.randint(1000000, 9999999),
            "symbol": symbol,
            "direction": signal.direction,
            "fill_price": fill_price,
            "actual_slippage_pips": actual_slippage_pips,
            "sl": signal.stop_loss,
            "tp": signal.take_profit,
            "lot": signal.volume if hasattr(signal, 'volume') else 0.0,
            "timestamp": time.time(),
            "latency_ms": self.latency_ms,
            "is_error": False
        }

    def simulate_exit(self, ticket: int, exit_type: str, price: float, point: float) -> Dict[str, Any]:
        """Simulates an exit event (SL/TP) with appropriate slippage."""
        slip_points = self._sample_slippage(point, tier=exit_type)
        
        # TP usually gets better/neutral execution, SL usually gets worse (negative) slippage
        if exit_type == "tp":
            exit_price = price - slip_points # Slight positive slippage simulated as price hit
        else:
            exit_price = price - slip_points # Negative slippage for SL
            
        return {
            "ticket": ticket,
            "exit_price": exit_price,
            "exit_type": exit_type,
            "exit_time": time.time()
        }

    def _get_effective_spread(self, base_spread: float) -> float:
        """
        Institutional Realism: Stochastic Spread Expansion.
        Simulates the widening of spreads during volatility.
        """
        # Volatility Multiplier: 1.0 to 2.5x base spread
        vol_mult = self._rng.uniform(1.0, 2.5)
        # Apply a 'Low Liquidity' boost occasionally
        if self._rng.random() < 0.1: vol_mult *= 1.5
        
        return base_spread * vol_mult

    def _sample_slippage(self, point: float, tier: str = "entry") -> float:
        """Samples slippage based on configured tiers."""
        pips = {"entry": self.entry_slip, "tp": self.tp_slip, "sl": self.sl_slip}.get(tier, self.entry_slip)
        # Random variance 0 to 100% of the tier limit
        return self._rng.uniform(0.0, pips) * point

if __name__ == "__main__":
    # Standalone Test logic
    logging.basicConfig(level=logging.INFO)
    test_config = {
        "execution": {"latency_ms": 0, "slippage_pips": 0.2},
        "backtest": {"deterministic": True, "random_seed": 42}
    }
    
    manager = OrderManager(test_config)
    signal = TradeSignal(direction="BUY", stop_loss=1.0900, take_profit=1.1100)
    prices = {"bid": 1.1000, "ask": 1.1001, "point": 0.0001}
    
    print("\n--- OrderManager Standalone Test ---")
    order = manager.execute_signal(signal, "EURUSD", prices)
    print(f"Executed Order: {order}")
    
    exit_res = manager.simulate_exit(order['ticket'], "sl", 1.0900, 0.0001)
    print(f"Simulated SL Exit: {exit_res}")
