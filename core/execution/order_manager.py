import logging
import time
import random
from typing import Optional, Dict, Any
from core.common.types import TradeSignal

class OrderManager:
    """
    V4-ULTRA Unified Execution Engine.
    Handles both LIVE MT5 execution and historical SIMULATION.
    Centralizes: Latency, Slippage, Spread Validation, and Retry Logic.
    """

    def __init__(self, config: Dict[str, Any], connection=None):
        self.config = config
        self.connection = connection # MT5Connection if live
        exe_cfg = config.get("execution", {})
        
        self.latency_ms = int(exe_cfg.get("latency_ms", 150))
        self.max_spread_pts = float(exe_cfg.get("max_spread_points", 500.0))
        
        # Slippage Tiers (Points)
        base_slip = float(exe_cfg.get("slippage_points", 1.0))
        self.entry_slip = float(exe_cfg.get("entry_slippage_points", base_slip))
        self.tp_slip = float(exe_cfg.get("tp_exit_slippage_points", base_slip * 0.5))
        self.sl_slip = float(exe_cfg.get("sl_exit_slippage_points", base_slip * 1.5))
        
        # Deterministic RNG for reproducibility (Institutional requirement)
        self.deterministic = config.get("backtest", {}).get("deterministic", False)
        self.seed = config.get("backtest", {}).get("random_seed", 42)
        self._rng = random.Random(self.seed if self.deterministic else None)
        
        self.logger = logging.getLogger("trading_bot.execution")

    def execute_signal(self, 
                       signal: TradeSignal, 
                       symbol: str, 
                       price_data: Dict[str, float],
                       is_news_blocked: bool = False,
                       magic: int = None,
                       comment: str = "V4-ULTRA",
                       timestamp: float = None) -> Optional[Dict[str, Any]]:
        """
        Processes a TradeSignal with institutional realism (Spread, News, Latency).
        Routes to Live MT5 if connection is present, otherwise Simulates.
        """
        if signal.direction == "NONE":
            return None

        # 1. News Blockade
        if is_news_blocked:
            self.logger.warning(f"Execution REJECTED: News Event Active for {symbol}")
            return None

        # 2. Institutional Live Path
        if self.connection and not self.config.get("backtest", {}).get("enabled", False):
            # Live execution via MT5Connection
            return self.connection.place_order(
                symbol=symbol,
                signal=signal,
                lot_size=getattr(signal, 'volume', 0.01),
                magic=magic,
                comment=comment
            )

        # 3. Simulation Path (Audit Bug #7 Fix)
        bid = price_data.get('bid')
        ask = price_data.get('ask')
        point = price_data.get('point', 0.00001)
        
        if not bid or not ask:
            self.logger.warning(f"Execution failed: Missing price data for {symbol}")
            return None

        # Variable Spread Simulation
        base_spread_pts = (ask - bid) / point
        effective_spread = self._get_effective_spread(base_spread_pts)
        
        if effective_spread > self.max_spread_pts:
            self.logger.warning(f"Execution Rejected: Spread too high ({effective_spread:.1f} > {self.max_spread_pts})")
            return None

        # Simulate Latency
        if not self.deterministic:
            time.sleep(self._rng.uniform(0.5, 1.5) * (self.latency_ms / 1000.0))

        # Simulate Entry Slippage
        slip_points = self._sample_slippage(point, tier="entry")
        fill_price = ask + slip_points if signal.direction == "BUY" else bid - slip_points
        
        actual_slippage_pips = slip_points / point if point > 0 else 0
        
        return {
            "ticket": self._rng.randint(1000000, 9999999),
            "symbol": symbol,
            "direction": signal.direction,
            "fill_price": fill_price,
            "actual_slippage_pips": actual_slippage_pips,
            "sl": signal.stop_loss,
            "tp": signal.take_profit,
            "lots": getattr(signal, 'volume', 0.0),
            "timestamp": timestamp if timestamp is not None else time.time(),
            "is_error": False
        }

    def simulate_exit(self, ticket: int, exit_type: str, price: float, point: float, direction: str = "BUY", exit_time: float = None) -> Dict[str, Any]:
        """Simulates an exit event (SL/TP) with appropriate slippage."""
        slip_points = self._sample_slippage(point, tier=exit_type)
        
        # TP usually gets better/neutral execution, SL usually gets worse (negative) slippage
        if exit_type == "tp":
            # Better price: Higher for BUY exit (selling), Lower for SELL exit (buying)
            exit_price = price + slip_points if direction == "BUY" else price - slip_points
        else:
            # Worse price: Lower for BUY exit, Higher for SELL exit
            exit_price = price - slip_points if direction == "BUY" else price + slip_points
            
        return {
            "ticket": ticket,
            "exit_price": exit_price,
            "exit_type": exit_type,
            "exit_time": exit_time if exit_time is not None else time.time()
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
    order = manager.execute_signal(signal, "XAUUSDm", prices)
    print(f"Executed Order: {order}")
    
    exit_res = manager.simulate_exit(order['ticket'], "sl", 1.0900, 0.0001, direction="BUY")
    print(f"Simulated SL Exit: {exit_res}")
