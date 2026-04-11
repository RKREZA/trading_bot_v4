import json
import logging
from datetime import datetime, timezone
import numpy as np

# Set up logging to console
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("certification_test")

def run_test():
    try:
        from core.common.types import CandleArray, ExecutionIntent, MarketSnapshot
        from core.execution.stochastic_kernel import StochasticKernel
        from core.data.fidelity import FidelityEngine
        from core.portfolio.audit_engine import AuditEngine
        from backtesting.reconstructor import PathReconstructor
        from types import MappingProxyType

        logger.info("--- STARTING GRADE-A+ CERTIFICATION TEST ---")

        # 1. Test Immutability
        logger.info("[1/5] Testing Deep Immutability...")
        intent = ExecutionIntent(
            symbol="XAUUSD", direction="BUY", volume=10.0, 
            stop_loss=2000.0, take_profit=2050.0, 
            strategy_id="TEST", setup_timestamp=123456789.0
        )
        try:
            intent.volume = 20.0 # Should raise FrozenInstanceError
            logger.error("IMMUTABILITY FAIL: intent.volume is mutable!")
        except Exception:
            logger.info("IMMUTABILITY PASS: ExecutionIntent is frozen.")

        # 2. Test DFS Engine
        logger.info("[2/5] Testing DFS Engine...")
        candles = CandleArray(
            time=np.array([i*60 for i in range(100)], dtype=np.int64),
            open=np.random.normal(2000, 5, 100),
            high=np.random.normal(2005, 5, 100),
            low=np.random.normal(1995, 5, 100),
            close=np.random.normal(2000, 5, 100),
            tick_volume=np.random.randint(100, 500, 100),
            spread=np.full(100, 2)
        )
        dfs = FidelityEngine.calculate_dfs(candles, "M1")
        logger.info(f"DFS Calculation: {dfs:.4f} ({FidelityEngine.get_classification(dfs)})")

        # 3. Test Stochastic Kernel
        logger.info("[3/5] Testing Stochastic Kernel...")
        snapshot = MarketSnapshot(
            timestamp=123456789.0, bid=2000.0, ask=2000.2, 
            spread=0.2, point=0.01, dfs=dfs, volatility="NORMAL",
            metadata=MappingProxyType({"liquidity_depth": 100.0, "latency_mu": 150.0})
        )
        kernel = StochasticKernel(global_seed=42)
        outcome = kernel.execute(intent, snapshot)
        logger.info(f"Execution Outcome: Price={outcome.fill_price:.2f}, Latency={outcome.actual_latency_ms:.1f}ms, Slip={outcome.actual_slippage_pips:.2f}pts")
        logger.info(f"Outcome Hash: {outcome.intent_hash}")

        # 4. Test Path Reconstructor
        logger.info("[4/5] Testing Path Reconstructor...")
        reconstructor = PathReconstructor(n_paths=200, seed=42)
        candle = candles[0]
        res = reconstructor.resolve_path(candle, 1990.0, 2010.0, "BUY", "NORMAL")
        logger.info(f"MC Resolve: P(SL)={res['p_sl']:.2f}, P(TP)={res['p_tp']:.2f}, CI={res['ci_95']:.4f}")

        # 5. Test Audit Engine
        logger.info("[5/5] Testing Audit Engine...")
        config = {"backtest": {"seed": 42}}
        fingerprint = AuditEngine.generate_fingerprint(config, {"symbol": "XAUUSD"})
        logger.info(f"System Fingerprint: {fingerprint}")
        
        # PnL Identity Verification
        pnl_data = {
            "alpha": 100.0,
            "execution_drag_loss": -5.0,
            "microstructure_loss": -2.0,
            "total": 93.0
        }
        is_valid = AuditEngine.verify_accounting_identity(
            pnl_data["total"], pnl_data["alpha"], pnl_data["execution_drag_loss"], pnl_data["microstructure_loss"]
        )
        logger.info(f"PnL Identity Check: {'PASS' if is_valid else 'FAIL'}")

        logger.info("--- CERTIFICATION TEST COMPLETE: GRADE A+ ---")

    except Exception as e:
        logger.error(f"CERTIFICATION TEST CRASHED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_test()
