import sys
import os
import logging
from comprehensive_backtest import ComprehensiveBacktestSuite
from core.config.loader import ConfigLoader
from strategies import create_strategy
from backtesting.backtester import PortfolioBacktester
from core.performance_tracker import PerformanceTracker

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("test_mss_refinements")

def main():
    # =========================================================================
    # 1. REPLACE THESE WITH THE BEST PARAMETERS FROM WFO ONCE IT FINISHES
    # =========================================================================
    wfo_best_params = {
        "volume_zscore_threshold": 1.2,
        "rejection_percentile": 0.35,
        "sweep_depth_percentile": 0.60,
        "max_atr_ratio": 2.5,
    }

    # =========================================================================
    # 2. MSS REFINEMENT VARIATIONS
    # =========================================================================
    variations = [
        {"name": "Baseline (10 bars, M5)", "mss_timeout_bars": 10, "mss_timeframe": "M5", "mss_buffer_pct": 0.0},
        {"name": "Longer timeout (15 bars)", "mss_timeout_bars": 15, "mss_timeframe": "M5", "mss_buffer_pct": 0.0},
        {"name": "M15 MSS", "mss_timeout_bars": 10, "mss_timeframe": "M15", "mss_buffer_pct": 0.0},
        {"name": "Buffer 0.1%", "mss_timeout_bars": 10, "mss_timeframe": "M5", "mss_buffer_pct": 0.001},
        {"name": "EMA Trend Filter", "mss_timeout_bars": 15, "ema_trend_filter": True, "ema_period": 50},
    ]

    symbol = "XAUUSDm"
    config_loader = ConfigLoader()
    base_config = config_loader.global_config
    symbol_config = config_loader.get_symbol_config(symbol)
    
    suite = ComprehensiveBacktestSuite()
    
    # Load a testing window (e.g. 3 months for out-of-sample testing)
    print("Loading test data...")
    m5 = suite.load_real_data(symbol=symbol, timeframe="M5", n_bars=25000)
    m15 = suite.load_real_data(symbol=symbol, timeframe="M15", n_bars=8500)
    h1 = suite.load_real_data(symbol=symbol, timeframe="H1", n_bars=2200)
    m1 = suite.load_real_data(symbol=symbol, timeframe="M1", n_bars=150000)
    
    m5, m15, h1, m1 = suite._align_timeframes(m1, m5, m15, h1)

    print("\n" + "=" * 80)
    print(" MSS REFINEMENT TESTING SUITE ")
    print("=" * 80)

    results = []

    for var in variations:
        print(f"\n--- Testing Variation: {var['name']} ---")
        
        # Merge global config + symbol config + wfo best params + variation params
        test_config = {**base_config, **symbol_config}
        if "symbols_config" not in test_config:
            test_config["symbols_config"] = {symbol: {}}
        
        # Apply specific overrides for testing
        test_config["risk_governance"] = test_config.get("risk_governance", {})
        test_config["risk_governance"]["min_tick_density"] = 1
        test_config["max_spread_points"] = 1500
        test_config["backtest"] = test_config.get("backtest", {})
        test_config["backtest"]["disable_checkpoint"] = True
        
        # Apply WFO best params & variation params to the symbol configuration directly
        if symbol not in test_config["symbols_config"]:
            test_config["symbols_config"][symbol] = {}
            
        for k, v in wfo_best_params.items():
            test_config[k] = v
        for k, v in var.items():
            if k != "name":
                test_config[k] = v
                
        strategy_id = "liquiditysweepbreakout_v7"
        strat = create_strategy(strategy_id, "LIQUIDITYSWEEPBREAKOUT", test_config)
        
        bt = PortfolioBacktester(test_config)
        
        # We suppress the normal backtester prints to keep output clean, 
        # but since we can't easily mute logger, we'll let it print.
        history, equity_history = bt.run(symbol, [strat], m5, h1, m15, m5, m1)
        
        if not history:
            print(f"[{var['name']}] Status: NO TRADES")
            results.append({"name": var["name"], "trades": 0, "pf": 0.0, "win_rate": 0.0, "max_dd": 0.0})
            continue
            
        metrics = PerformanceTracker.calculate_metrics(history, 10000, equity_history)
        
        trades = metrics.get("total_trades", len(history))
        
        pf_raw = metrics.get("profit_factor", 0.0)
        try:
            pf = float(pf_raw)
        except (ValueError, TypeError):
            pf = 0.0
            
        win_rate = float(metrics.get("win_rate", "0%").strip("%"))
        max_dd = float(metrics.get("max_drawdown", "0%").strip("%"))
        
        print(f"[{var['name']}] Trades: {trades} | PF: {pf:.2f} | Win Rate: {win_rate:.1f}% | Max DD: {max_dd:.2f}%")
        
        results.append({
            "name": var["name"],
            "trades": trades,
            "pf": pf,
            "win_rate": win_rate,
            "max_dd": max_dd
        })

    print("\n" + "=" * 80)
    print(" MSS REFINEMENT SUMMARY RESULTS ")
    print("=" * 80)
    print(f"{'Variation':<30} | {'Trades':<8} | {'PF':<6} | {'Win %':<6} | {'Max DD %':<8}")
    print("-" * 70)
    for res in results:
        print(f"{res['name']:<30} | {res['trades']:<8} | {res['pf']:<6.2f} | {res['win_rate']:<6.1f} | {res['max_dd']:<8.2f}")

if __name__ == "__main__":
    main()
