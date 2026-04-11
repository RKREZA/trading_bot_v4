import numpy as np
import hashlib
import logging
from typing import Dict, Any, Tuple
from core.common.types import ExecutionIntent, MarketSnapshot, ExecutionOutcome

logger = logging.getLogger("trading_bot.kernel")

class StochasticKernel:
    """
    V5-INSIGNIA Institutional Execution Kernel (Grade A+).
    Pure Function: Outcome = f(Intent, Snapshot, Seed).
    Implements Tempered Impact, Hybrid Latency, and Symmetric OBI.
    """

    def __init__(self, global_seed: int = 42):
        self.global_seed = global_seed

    def execute(self, intent: ExecutionIntent, snapshot: MarketSnapshot) -> ExecutionOutcome:
        """
        Executes an immutable intent against a frozen market snapshot.
        Deterministic given the seed (global_seed + intent_hash).
        """
        # 1. Deterministic Seed Control (Rule 2.1 & 3.1)
        # Using domain separation to prevent collision
        seed_hex = hashlib.sha256(f"KERNEL|{self.global_seed}|{intent.intent_hash}".encode()).hexdigest()
        seed = int(seed_hex[:15], 16) % (2**32)
        rng = np.random.default_rng(seed)
        
        # Rule 2.2: RNG Warmup (Discard biased initial states)
        for _ in range(5):
            rng.random()

        # 2. Market Impact Model (Normalized Tempered Exponential)
        # x = order_size / liquidity_depth
        # We estimate depth from tick volume and volatility
        liquidity_depth = max(10.0, snapshot.metadata.get("liquidity_depth", 100.0))
        x = intent.volume / liquidity_depth
        x = min(x, 7.0) # Edge-case clamp
        
        # Tempered Impact Function
        impact_scale = np.log1p(x)
        impact_factor = ((np.exp(x) - 1) / (np.exp(x) + 1)) * impact_scale
        base_impact_pts = snapshot.metadata.get("base_impact_points", 1.0)
        microstructure_loss = base_impact_pts * impact_factor * snapshot.point

        # 3. Hybrid Latency Model (Normal + Truncated Cauchy)
        mu = snapshot.metadata.get("latency_mu", 150.0) # ms
        sigma = snapshot.metadata.get("latency_sigma", 50.0)
        gamma = snapshot.metadata.get("latency_gamma", 10.0) # Cauchy scale
        
        # Latency Floor/Cap
        latency_floor = 50.0
        latency_cap = mu * 5.0
        
        baseline = rng.normal(mu, sigma)
        spike = 0.0
        if rng.random() < 0.05: # 5% probability of infrastructure spike
            # Cauchy truncated at latency_cap
            spike = abs(rng.standard_cauchy() * gamma)
            
        latency_ms = np.clip(baseline + spike, latency_floor, latency_cap)
        
        # 4. Hybrid OBI Model (Directional Asymmetry)
        # We enforce saturation at 0.85
        obi_val = np.clip(snapshot.metadata.get("obi", 0.0), -0.85, 0.85)
        
        # Directional Skew: If OBI supports the trade, slippage is reduced.
        # If OBI opposes (e.g. buying into selling pressure), slippage is increased.
        is_buy = intent.direction == "BUY"
        # OBI > 0 means buying pressure.
        # So if is_buy and OBI > 0, alignment = +1. 
        # If is_buy and OBI < 0, alignment = -1.
        alignment = (1 if is_buy else -1) * np.sign(obi_val)
        obi_penalty_mult = 1.0 + (abs(obi_val) * (1.0 if alignment < 0 else -0.5))
        
        # 5. Final Friction Assembly
        # Total Friction = Execution_Drag (Spread/Slip) + Microstructure (Impact)
        base_slip_pts = snapshot.metadata.get("base_slippage_points", 0.5)
        # Latency slip: 0.5 pts per 100ms
        latency_slip_pts = (latency_ms / 100.0) * 0.5
        
        execution_drag_pts = (base_slip_pts * obi_penalty_mult) + latency_slip_pts
        execution_drag = execution_drag_pts * snapshot.point
        
        total_friction = execution_drag + microstructure_loss
        
        # Rule 3.3: Bit-Level Price Calculation
        alpha_price = np.float64(snapshot.ask if is_buy else snapshot.bid)
        fill_price = np.float64(alpha_price + total_friction if is_buy else alpha_price - total_friction)
        actual_slippage_pips = np.float64((total_friction / snapshot.point) if snapshot.point > 0 else 0.0)

        return ExecutionOutcome(
            ticket=rng.integers(1000000, 9999999),
            fill_price=float(fill_price),
            actual_slippage_pips=float(actual_slippage_pips),
            actual_latency_ms=float(latency_ms),
            alpha_price=float(alpha_price),
            microstructure_loss=float(microstructure_loss),
            execution_drag=float(execution_drag),
            timestamp=snapshot.timestamp + (latency_ms / 1000.0),
            intent_hash=intent.intent_hash,
            is_error=False
        )
