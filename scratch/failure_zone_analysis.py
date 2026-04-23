"""
V5-INSIGNIA Failure Zone Forensic Analysis
==========================================
Deep-dive into Jul-Aug 2025 and Jan-Feb 2026 failure zones.
Logs every signal, rejection reason, regime state, and per-trade forensics.
"""
import sys, os
sys.path.append(os.getcwd())

import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
import json
import logging

from core.common.types import CandleArray
from core.regime_detector import RegimeDetector
from core.regime_store import MemoryRegimeStore
from core.session_detector import SessionDetector
from core.base_strategy import MarketData
from core.indicator_engine import IndicatorEngine
from backtesting.backtester import PortfolioBacktester
from core.performance_tracker import PerformanceTracker
from strategies import create_strategy
from core.config.loader import ConfigLoader

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(name)s] %(message)s')
logger = logging.getLogger("failure_zone_analysis")

FAILURE_ZONES = {
    "Jul-Aug 2025": {
        "start": datetime(2025, 7, 1, tzinfo=timezone.utc),
        "end": datetime(2025, 9, 1, tzinfo=timezone.utc),
    },
    "Jan-Feb 2026": {
        "start": datetime(2026, 1, 1, tzinfo=timezone.utc),
        "end": datetime(2026, 3, 1, tzinfo=timezone.utc),
    },
}

def load_data(symbol, timeframe, n_bars=None):
    path = f"data_cache/{symbol}/{timeframe}.parquet"
    if not os.path.exists(path):
        logger.error(f"Missing data: {path}")
        return None
    df = pd.read_parquet(path)
    if n_bars:
        df = df.tail(n_bars)
    return CandleArray(
        time=df['time'].values,
        open=df['open'].values,
        high=df['high'].values,
        low=df['low'].values,
        close=df['close'].values,
        tick_volume=df['tick_volume'].values,
        spread=df['spread'].values,
    )

def slice_by_time(candles, start_ts, end_ts):
    if candles is None: return None
    mask = (candles.time >= start_ts) & (candles.time < end_ts)
    return candles[mask]

def analyze_regime_distribution(m5, h1, start_ts, end_ts):
    """Calculate what % of time was in each regime."""
    detector = RegimeDetector()
    store = MemoryRegimeStore()
    
    regimes = {"TREND": 0, "RANGE": 0, "LIQUIDITY_EVENT": 0, "EXPANSION": 0, "TRANSITION": 0}
    total = 0
    
    m5._indicators = IndicatorEngine.precalculate_all("XAUUSDm", "M5", m5)
    h1._indicators = IndicatorEngine.precalculate_all("XAUUSDm", "H1", h1)
    
    step = max(1, len(m5.time) // 200)  # Sample ~200 points for speed
    
    for i in range(200, len(m5.time), step):
        m5.set_limit(i)
        t = m5.time[i]
        dt = datetime.fromtimestamp(float(t), tz=timezone.utc)
        
        h1_idx = np.searchsorted(h1.time, t, side='right')
        if h1_idx > 0: h1.set_limit(h1_idx)
        
        class Shim: pass
        shim = Shim()
        shim.m5_candles = m5
        shim.htf_candles = h1
        shim.session = SessionDetector.get_session(dt, 0)
        shim.timestamp = dt
        
        state = store.load("ANALYSIS")
        try:
            info, new_state, _ = detector.detect(shim, state, f"ANALYSIS:{i}", "ANALYSIS", is_live=False)
            store.save("ANALYSIS", new_state)
            regime_name = info.market_type.name if hasattr(info.market_type, 'name') else str(info.market_type)
            regimes[regime_name] = regimes.get(regime_name, 0) + 1
            total += 1
        except Exception as e:
            logger.debug(f"Regime detection error at bar {i}: {e}")
    
    # Reset limits
    m5.set_limit(len(m5.time))
    h1.set_limit(len(h1.time))
    
    if total > 0:
        return {k: round(v / total * 100, 1) for k, v in regimes.items()}
    return regimes

def run_strategy_on_zone(strategy_name, zone_name, m5, m15, h1, m1, config):
    """Runs a single strategy on a failure zone and returns detailed forensics."""
    logger.info(f"  Running {strategy_name} on {zone_name}...")
    
    sid = f"{strategy_name.lower()}_v4"
    st_type = strategy_name.upper()
    
    try:
        strategy = create_strategy(sid, st_type, config)
    except Exception as e:
        logger.error(f"  Failed to create {strategy_name}: {e}")
        return None
    
    bt_config = config.copy()
    bt_config["backtest"] = bt_config.get("backtest", {}).copy()
    bt_config["backtest"]["initial_balance_per_strategy"] = 5000.0
    bt_config["backtest"]["deterministic"] = True
    bt_config["backtest"]["random_seed"] = 42
    bt_config["backtest"]["disable_checkpoint"] = True
    bt_config["backtest"]["debug_signals"] = True
    
    if "risk_governance" not in bt_config: bt_config["risk_governance"] = {}
    bt_config["risk_governance"]["min_tick_density"] = 1
    bt_config["max_spread_points"] = 1500
    
    # Re-enable strategy for backtest
    if strategy_name in bt_config:
        bt_config[strategy_name]["enabled"] = True
    
    bt = PortfolioBacktester(bt_config)
    
    try:
        history, equity = bt.run("XAUUSDm", [strategy], m5, h1, m15, m5, m1)
        
        if not history:
            return {"status": "NO_TRADES", "strategy": strategy_name, "zone": zone_name, "trades": 0}
        
        metrics = PerformanceTracker.calculate_metrics(history, 5000.0, equity)
        
        # Per-trade forensics
        trade_details = []
        for t in history:
            trade_details.append({
                "direction": t.get("direction"),
                "entry": round(t.get("fill_price", 0), 2),
                "exit": round(t.get("exit_price", 0), 2),
                "pnl": round(t.get("pnl", 0), 2),
                "result": t.get("result"),
                "lots": round(t.get("lots", 0), 4),
                "sl": round(t.get("sl", 0), 2),
                "tp": round(t.get("tp", 0), 2),
            })
        
        # Rejection stats
        rejections = bt.rejection_stats.get(sid, {})
        
        return {
            "status": "SUCCESS",
            "strategy": strategy_name,
            "zone": zone_name,
            "trades": len(history),
            "net_pnl": round(sum(t["pnl"] for t in history), 2),
            "win_rate": metrics.get("win_rate", "0%"),
            "sharpe": metrics.get("sharpe_ratio", 0),
            "max_dd": metrics.get("max_drawdown", "0%"),
            "profit_factor": metrics.get("profit_factor", 0),
            "trade_details": trade_details,
            "top_rejections": dict(sorted(rejections.items(), key=lambda x: x[1], reverse=True)[:10]),
        }
        
    except Exception as e:
        logger.error(f"  Error running {strategy_name}: {e}")
        import traceback
        traceback.print_exc()
        return {"status": "ERROR", "strategy": strategy_name, "zone": zone_name, "error": str(e)}


def main():
    print("=" * 80)
    print(" V5-INSIGNIA FAILURE ZONE FORENSIC ANALYSIS")
    print("=" * 80)
    
    loader = ConfigLoader()
    config = loader.global_config
    
    # Load all timeframes
    logger.info("Loading market data...")
    m5_all = load_data("XAUUSDm", "M5")
    m15_all = load_data("XAUUSDm", "M15")
    h1_all = load_data("XAUUSDm", "H1")
    m1_all = load_data("XAUUSDm", "M1")
    
    if m5_all is None:
        logger.error("Cannot proceed without M5 data.")
        return
    
    strategies = ["TrendFollowing", "LiquiditySweepBreakout", "SmartMeanReversion"]
    all_results = {}
    
    for zone_name, zone in FAILURE_ZONES.items():
        print(f"\n{'='*60}")
        print(f"  ZONE: {zone_name} ({zone['start'].date()} to {zone['end'].date()})")
        print(f"{'='*60}")
        
        start_ts = zone["start"].timestamp()
        end_ts = zone["end"].timestamp()
        
        # Slice data (with 200-bar warmup for indicators)
        warmup_ts = start_ts - (200 * 300)  # 200 M5 bars before
        
        m5 = slice_by_time(m5_all, warmup_ts, end_ts)
        m15 = slice_by_time(m15_all, warmup_ts, end_ts)
        h1 = slice_by_time(h1_all, warmup_ts - 3600*200, end_ts)  # Extra warmup for H1
        m1 = slice_by_time(m1_all, warmup_ts, end_ts)
        
        if m5 is None or len(m5) < 300:
            logger.warning(f"  Insufficient M5 data for {zone_name}")
            continue
        
        # 1. Regime Distribution Analysis
        print(f"\n  [1] REGIME DISTRIBUTION")
        print(f"  {'-'*40}")
        m5_regime = slice_by_time(m5_all, start_ts, end_ts)
        h1_regime = slice_by_time(h1_all, start_ts - 3600*200, end_ts)
        
        if m5_regime is not None and h1_regime is not None:
            regime_dist = analyze_regime_distribution(m5_regime, h1_regime, start_ts, end_ts)
            for regime, pct in sorted(regime_dist.items(), key=lambda x: x[1], reverse=True):
                bar = "█" * int(pct / 2)
                print(f"    {regime:20} {pct:5.1f}%  {bar}")
        
        # 2. Price Action Context
        print(f"\n  [2] PRICE ACTION CONTEXT")
        print(f"  {'-'*40}")
        zone_m5 = slice_by_time(m5_all, start_ts, end_ts)
        if zone_m5 is not None and len(zone_m5) > 0:
            price_start = zone_m5.open[0]
            price_end = zone_m5.close[-1]
            price_high = np.max(zone_m5.high)
            price_low = np.min(zone_m5.low)
            price_range = price_high - price_low
            pct_change = (price_end - price_start) / price_start * 100
            print(f"    Open:  {price_start:.2f}")
            print(f"    Close: {price_end:.2f} ({pct_change:+.2f}%)")
            print(f"    High:  {price_high:.2f}")
            print(f"    Low:   {price_low:.2f}")
            print(f"    Range: {price_range:.2f} ({price_range/price_start*100:.2f}%)")
        
        # 3. Per-Strategy Forensics
        print(f"\n  [3] STRATEGY PERFORMANCE")
        print(f"  {'-'*40}")
        
        zone_results = {}
        for strat_name in strategies:
            result = run_strategy_on_zone(strat_name, zone_name, m5, m15, h1, m1, config)
            zone_results[strat_name] = result
            
            if result and result["status"] == "SUCCESS":
                print(f"    {strat_name:30} | Trades: {result['trades']:3} | PnL: ${result['net_pnl']:>8.2f} | WR: {result['win_rate']:>6} | PF: {result['profit_factor']:>5}")
                
                # Show top rejections
                if result.get("top_rejections"):
                    print(f"      Top Rejections:")
                    for reason, count in list(result["top_rejections"].items())[:5]:
                        print(f"        [{count:5}] {reason[:70]}")
                        
                # Show worst trades
                if result.get("trade_details"):
                    worst = sorted(result["trade_details"], key=lambda x: x["pnl"])[:3]
                    print(f"      Worst Trades:")
                    for t in worst:
                        print(f"        {t['direction']} @ {t['entry']} → {t['exit']} | PnL: ${t['pnl']:.2f} ({t['result']})")
            else:
                status = result.get("status", "UNKNOWN") if result else "ERROR"
                print(f"    {strat_name:30} | {status}")
        
        all_results[zone_name] = zone_results
    
    # Save full results
    output_path = "backtest_results/failure_zone_forensics.json"
    os.makedirs("backtest_results", exist_ok=True)
    
    # Make serializable
    serializable = {}
    for zone, strats in all_results.items():
        serializable[zone] = {}
        for sname, data in strats.items():
            if data:
                serializable[zone][sname] = {k: v for k, v in data.items()}
    
    with open(output_path, "w") as f:
        json.dump(serializable, f, indent=2, default=str)
    
    print(f"\n\n{'='*80}")
    print(f" FORENSIC ANALYSIS COMPLETE — Results saved to {output_path}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
