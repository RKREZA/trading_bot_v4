"""Quick import and smoke test for the multi-strategy framework."""
import sys
import os
import traceback

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    # 1. Core imports
    from core.base_strategy import BaseStrategy, MarketData, TaggedSignal
    from core.order_tagger import OrderTagger
    from core.position_tracker import PositionTracker
    from core.performance_tracker import PerformanceTracker
    from core.strategy_runtime import StrategyRuntime, StrategyState
    print("[OK] Core imports")

    # 2. Strategy imports
    from strategies import create_strategy, STRATEGY_REGISTRY, SniperStrategy, SMCStrategy
    print(f"[OK] Strategy Registry: {list(STRATEGY_REGISTRY.keys())}")

    # 3. Create strategies via factory
    sniper = create_strategy("sniper_v1", "SNIPER", {"enabled": True})
    smc = create_strategy("smc_v1", "SMC", {"enabled": False})
    print(f"[OK] Created: {sniper}")
    print(f"[OK] Created: {smc}")

    # 4. OrderTagger
    comment = OrderTagger.create_comment("sniper_v1", "abc123def456")
    print(f"[OK] OrderTagger.create_comment -> '{comment}'")
    parsed = OrderTagger.parse_comment(comment)
    print(f"[OK] OrderTagger.parse_comment -> {parsed}")
    assert parsed[0] == "sniper_v1"
    assert OrderTagger.is_tagged(comment)
    assert not OrderTagger.is_tagged("Bot V3")
    print("[OK] OrderTagger assertions passed")

    # 5. PositionTracker
    pt = PositionTracker("test_strategy")
    pt.add_position(12345, {"entry_price": 2000.0, "direction": "BUY"})
    assert pt.open_count == 1
    assert pt.has_open_position
    assert pt.get_position(12345)["entry_price"] == 2000.0
    pt.update_position(12345, {"best_price": 2010.0})
    assert pt.get_position(12345)["best_price"] == 2010.0
    removed = pt.remove_position(12345)
    assert removed is not None
    assert pt.open_count == 0
    print("[OK] PositionTracker assertions passed")

    # 6. PerformanceTracker
    perf = PerformanceTracker("test_perf", 1000.0)
    perf.record_trade({"ticket": 1, "pnl": 50.0, "result": "TP"})
    perf.record_trade({"ticket": 2, "pnl": -20.0, "result": "SL"})
    perf.record_trade({"ticket": 3, "pnl": 30.0, "result": "TP"})
    assert perf.total_trades == 3
    assert perf.win_count == 2
    assert perf.loss_count == 1
    assert abs(perf.balance - 1060.0) < 0.01
    summary = perf.get_summary()
    assert summary["total_trades"] == 3
    print(f"[OK] PerformanceTracker: balance=${perf.balance:.2f}, WR={perf.win_rate:.1f}%")

    # 7. Finalize metrics
    metrics = perf.finalize()
    assert metrics["total_trades"] == 3
    assert metrics["net_profit"] == 60.0
    print(f"[OK] Finalized metrics: PF={metrics['profit_factor']}, WR={metrics['win_rate']}%")

    # 8. StrategyRuntime
    config = {
        "risk": {"risk_per_trade_pct": 1.0, "max_daily_trades": 5},
        "session_config": {}
    }
    runtime = StrategyRuntime(sniper, config, 1000.0)
    assert runtime.strategy_id == "sniper_v1"
    assert runtime.enabled
    state = runtime.get_state()
    assert state["strategy_id"] == "sniper_v1"
    print(f"[OK] StrategyRuntime: {runtime}")

    # 9. Verify BaseStrategy is abstract
    try:
        BaseStrategy("test", {})
        assert False, "Should not be able to instantiate ABC"
    except TypeError:
        print("[OK] BaseStrategy is properly abstract")

    # 10. Strategy isolation check
    sniper2 = create_strategy("sniper_v2", "SNIPER", {"enabled": True})
    rt1 = StrategyRuntime(sniper, config, 1000.0)
    rt2 = StrategyRuntime(sniper2, config, 1000.0)
    rt1.state.daily_trades = 5
    assert rt2.state.daily_trades == 0, "Strategy state leaked!"
    print("[OK] Strategy isolation verified")

    # 11. Orchestrator import
    from core.strategy_orchestrator import StrategyOrchestrator
    print("[OK] StrategyOrchestrator imported")

    # 12. Backtester import
    from core.backtester import BacktestEngine, MultiStrategyBacktestEngine
    print("[OK] BacktestEngine + MultiStrategyBacktestEngine imported")

    print("")
    print("=" * 50)
    print("  ALL TESTS PASSED")
    print("=" * 50)

except Exception as e:
    print(f"FAILED: {e}")
    traceback.print_exc()
    sys.exit(1)
