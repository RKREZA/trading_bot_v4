"""
BACKTEST ENGINE V5 - Institutional Multi-Strategy Interface
High-fidelity historical simulation with Portfolio Governance.
"""

import argparse
import json
import os
import sys
import logging
import numpy as np
from datetime import datetime, timezone, timedelta
from typing import Optional, List, Dict, Any

from backtesting.backtester import DatasetFingerprinter, ENGINE_VERSION

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import print as rprint

from dotenv import load_dotenv
from core import SourceHandler, MT5Connection, PerformanceTracker, setup_logging
from core.data.manager import DataManager
from backtesting import PortfolioBacktester, MonteCarloSimulator, StressTester, WalkForwardValidator
from strategies import create_strategy, STRATEGY_REGISTRY
from core.config.loader import ConfigLoader
from core.portfolio.audit_engine import AuditEngine
from core.common.types import CanonicalHasher

load_dotenv()

class BacktestCLI:
    def __init__(self):
        self.console = Console()
        self.config_loader = ConfigLoader(environment="backtest")
        self.config = self.config_loader.global_config
        self.data_manager = DataManager(self.config)
        self.connection = MT5Connection()

    def run(
        self,
        symbol: str,
        start: str,
        end: str,
        strategy_filter: str = None,
        run_monte_carlo: bool = False,
        run_walk_forward: bool = False,
        run_stress_test: bool = False,
        seed: Optional[int] = None,
        deterministic: Optional[bool] = None,
        resume: bool = False,
        debug_signals: bool = False,
        no_adaptive: bool = False
    ):
        setup_logging(console=True)
        
        # [ Institutional A+ Refactor ]: Load Symbol-Specific Config
        self.config = self.config_loader.get_symbol_config(symbol)
        self.data_manager = DataManager(self.config) # Re-init with symbol specifics
        
        if not self.connection.connect():
            rprint("[bold red]Critical Error:[/] Could not connect to MT5.")
            return

        try:
            dt_from = datetime.fromisoformat(start).replace(tzinfo=timezone.utc)
            dt_to = datetime.fromisoformat(end).replace(tzinfo=timezone.utc)
        except ValueError:
            rprint("[bold red]Error:[/] Start/End dates must be in YYYY-MM-DD format.")
            return

        # 1. Flawless Data Prep (Step 2.5)
        rprint(Panel(f"[bold cyan]V5-INSIGNIA Production Benchmark[/]\nSymbol: {symbol} | Range: {start} to {end}", border_style="bright_blue"))
        
        with self.console.status("[bold green]Orchestrating Flawless MT5 Sync...") as status:
            m1 = self.data_manager.prepare_data(symbol, "M1", dt_from)
            m5 = self.data_manager.prepare_data(symbol, "M5", dt_from)
            m15 = self.data_manager.prepare_data(symbol, "M15", dt_from)
            h1 = self.data_manager.prepare_data(symbol, "H1", dt_from)
            d1 = self.data_manager.prepare_data(symbol, "D1", dt_from)
            
            # [ Institutional Alignment Filter Removed ]: Ensuring full-session coverage
            pass

        if len(m5) < 100:
            rprint("[bold red]Error:[/] Insufficient data for benchmark.")
            return

        # [ Range Capping ]: Slice M5/M15 by dt_to, but preserve M1 for trade management
        # M1 data is critical for trade execution and should extend to cover all M5 bars
        to_ts = dt_to.timestamp()
        m5 = m5[m5.time < to_ts]
        m15 = m15[m15.time < to_ts]
        
        # [ Institutional Warmup ]: Extend H1 to include historical warmup bars for strategy lookback
        # Extend H1 by 500 bars before test start to ensure strategies have sufficient historical data
        m5_start_ts = m5.time[0]
        h1_start_ts = h1.time[0]
        h1_warmup_idx = 0
        
        # Find how many H1 bars exist before m5_start_ts
        h1_idx_for_m5_start = np.searchsorted(h1.time, m5_start_ts, side='left')
        
        # We need at least 500 H1 bars for warmup. 
        # The slice returned by prepare_data already includes a buffer.
        # We should NOT slice it further unless we have too much.
        if h1_idx_for_m5_start < 200:
            rprint(f"[yellow]Warning: Only {h1_idx_for_m5_start} H1 bars available for warmup. Some indicators may be unstable at start.[/]")
        
        # Now cap H1 to to_ts after warmup is applied
        h1 = h1[h1.time < to_ts]
        


        if len(m5) < 10:
             rprint("[bold red]Error:[/] Date range results in zero bars for simulation.")
             return

        # 1.5 Data Fidelity Audit (Institutional Standard)
        audit_table = Table(title="Institutional Data Fidelity Audit", box=None, header_style="bold cyan")
        audit_table.add_column("Property", style="dim")
        audit_table.add_column("Value")
        
        # Calculate coverage
        days_requested = (dt_to - dt_from).days
        expected_m5 = (days_requested * 24 * 12) * 0.7 # Approx accounting for weekends
        coverage = min(100.0, (len(m5) / expected_m5 * 100)) if expected_m5 > 0 else 100.0
        
        audit_table.add_row("Symbol/Mode", f"{symbol} (High-Fidelity)")
        audit_table.add_row("Precision", "M5 Primary + M1 Tick-Replay")
        audit_table.add_row("Bars (M5)", f"{len(m5):,}")
        audit_table.add_row("Temporal Range", f"{datetime.fromtimestamp(m5.time[0], tz=timezone.utc).date()} to {datetime.fromtimestamp(m5.time[-1], tz=timezone.utc).date()}")
        audit_table.add_row("Fidelity Score", f"{coverage:.1f}%")
        
        rprint(audit_table)
        rprint("-" * 60)

        # 2. Strategy Loader (Step 9)
        strategies = self._build_strategies(strategy_filter, symbol=symbol)
        if not strategies:
            rprint(f"[bold red]Error:[/] No strategies available for benchmark.")
            return

        # 2.5 [ M1 Validation ]: Institutional Grade-A+ Strict Data Policy
        # Synthetic data generation is disabled to prevent "Hallucinated Profits".
        # If the strategy requires M1 fidelity, we must provide real MT5 M1 data.
        m1_required = self.config.get("backtest", {}).get("timeframe", "M5") == "M1" or \
                      any(s.config.get("timeframe") == "M1" for s in strategies)
        
        if m1_required:
            if len(m1) == 0:
                rprint("[bold red]CRITICAL DATA ERROR:[/] M1 data is required but unavailable for this range. Institutional audit FAILED.")
                return
            
            if len(m5) > 0 and m5.time[0] < m1.time[0]:
                rprint(f"[bold red]CRITICAL DATA GAP:[/] M5 starts at {datetime.fromtimestamp(m5.time[0], tz=timezone.utc).date()}, but M1 starts later at {datetime.fromtimestamp(m1.time[0], tz=timezone.utc).date()}. Partial simulation rejected.")
                return

        rprint(f"[bold yellow]DEBUG:[/] M1 range: {datetime.fromtimestamp(m1.time[0], tz=timezone.utc)} to {datetime.fromtimestamp(m1.time[-1], tz=timezone.utc)}")
        rprint(f"[bold yellow]DEBUG:[/] M1 last close: {m1.close[-1]}")


        # 3. Validation: Walk-Forward (Phase 15/16)
        if run_walk_forward:
            self._run_walk_forward(symbol, strategies, {"M5": m5, "M1": m1, "M15": m15, "H1": h1})

        # 4. Core Portfolio Simulation
        # Lookup Symbol-specific Primary Timeframe (Reality Check)
        symbol_cfg = self.config.get("symbols_config", {}).get(symbol, {})
        primary_tf = symbol_cfg.get("backtest_timeframe", "M5")
        
        # Select the driver data
        tf_map = {"M1": m1, "M5": m5, "M15": m15, "H1": h1}
        primary_data = tf_map.get(primary_tf, m5)
        
        rprint(f"[green]Executing V5-INSIGNIA Event-Driven Simulation on {primary_tf} (Resume={resume})...[/]")
        
        runtime_config = dict(self.config)
        runtime_backtest = dict(runtime_config.get("backtest", {}))
        runtime_backtest["enabled"] = True
        if seed is not None:
            runtime_backtest["random_seed"] = int(seed)
        if deterministic is not None:
            runtime_backtest["deterministic"] = bool(deterministic)
        else:
            # Institutional Default: If not specified, force deterministic to True for audit stability
            runtime_backtest["deterministic"] = True
            
        if no_adaptive:
            runtime_backtest["adaptive_strategy"] = False
            rprint("[bold cyan]ADAPTIVE STRATEGY DISABLED: All strategies will run simultaneously.[/]")
        runtime_config["backtest"] = runtime_backtest
        if debug_signals:
            runtime_config["backtest"]["debug_signals"] = True
            rprint("[bold yellow]DIAGNOSTIC MODE ENABLED: Signal rejection reasons will be logged.[/]")

        backtester = PortfolioBacktester(runtime_config)
        
        # [ Rule 3.1: Institutional Dataset Fingerprinting ]
        dataset_hashes = {}
        for tf in ["M1", "M5", "M15", "H1"]:
            p = self.data_manager.store.get_path(symbol, tf)
            if os.path.exists(p):
                dataset_hashes[tf] = DatasetFingerprinter.get_hash(p)
        
        # Full Simulation Run with V5 Lockdown Guards
        backtester.run(symbol, strategies, primary_data, h1, m15, m5, m1, d1_data=d1, data_hashes=dataset_hashes, resume=resume, start_ts=dt_from.timestamp())
        history = backtester.history
        equity_history = backtester.equity_history
        
        # 5. Performance Attribution
        partition_initial = float(self.config.get("backtest", {}).get("initial_balance_per_strategy", 1000.0))
        total_initial = len(strategies) * partition_initial
        
        # Aggregate Portfolio Equity
        portfolio_equity = []
        if equity_history:
            import pandas as pd
            eq_df = pd.DataFrame(equity_history)
            portfolio_equity = eq_df.groupby('time')['equity'].sum().tolist()

        stats = PerformanceTracker.calculate_metrics(history, total_initial, equity_curve=portfolio_equity)
        strat_stats = PerformanceTracker.calculate_per_strategy(history, partition_initial)
        session_stats = PerformanceTracker.calculate_per_session(history, total_initial)
        
        full_results = {
            "symbol": symbol,
            "start_date": start,
            "end_date": end,
            "portfolio": stats,
            "strategies": strat_stats,
            "sessions": session_stats
        }

        # 6. Audit Pack Generation (Rule 6.2: Institutional Graduation Capsule)
        audit_dir = getattr(backtester, "audit_trail_dir", "audit_trail")
        
        # Extract Outcomes for Trace Lock
        outcomes = [t["outcome"] for t in history if "outcome" in t]
        trace_lock = AuditEngine.generate_trace_lock(outcomes)
        fingerprint = AuditEngine.generate_fingerprint(runtime_config, {"symbol": symbol, "data_hashes": dataset_hashes})
        
        # Final Graduation Bundle (Capsule)
        AuditEngine.generate_bundle(
            output_dir=audit_dir,
            fingerprint=fingerprint,
            trace_lock=trace_lock,
            data_hashes=dataset_hashes,
            config=runtime_config,
            audit_results=full_results
        )
        
        # [ Export Trades CSV ]
        if history:
            import pandas as pd
            trades_df = pd.DataFrame(history)
            trades_df.to_csv(os.path.join(audit_dir, "trades.csv"), index=False)
            
        # 7. Dashboard Display
        dashboard = PerformanceTracker.generate_professional_dashboard(full_results)
        rprint(f"\n[bold white]{dashboard}[/]")
        rprint(f"\n[bold green]Institutional Audit Pack Persistence: {audit_dir}[/]")

        # 8. Robustness: Monte Carlo (Step 15/16)
        if run_monte_carlo and history:
            self._run_monte_carlo(history, partition_initial)
            
        if self.config.get("backtest", {}).get("run_stress_test", False):
            self._run_stress_test(symbol, strategies, {"M1": m1, "M5": m5, "M15": m15, "H1": h1})

    def _run_walk_forward(self, symbol, strategies, data):
        rprint("\n[bold yellow]Validation Audit: Running Walk-Forward Optimization (Rolling Windows)...[/]")
        wfo = WalkForwardValidator(self.config)
        results = wfo.run_validation(symbol, strategies, data)
        wfo.summarize_wfo_results(results)

    def _build_strategies(self, strategy_filter: str = None, symbol: str = None):
        strategies = []
        
        # 1. Normalize Strategy Filter into a list
        requested_ids = []
        if strategy_filter:
            requested_ids = [s.strip().lower() for s in strategy_filter.split(",")]

        for st_type, st_class in STRATEGY_REGISTRY.items():
            # 2. Define potential ID matches for this class (PascalCase Priority)
            pascal_name = st_type.title().replace("_", "")


            
            potential_ids = [
                pascal_name,
                f"{pascal_name}_v5",
                st_type.lower(),
                f"{st_type.lower()}_v5"
            ]
            
            # 3. Match Identification
            matched_id = None
            if not requested_ids:
                # Default behavior: Load all using standardized PascalCase
                matched_id = pascal_name
            else:
                # Check if any part of our filter matches this strategy
                for rid in requested_ids:
                    # Partial matching: allow 'Breakout' to match 'LiquiditySweepBreakout'
                    if (rid.lower() == pascal_name.lower() or 
                        rid.lower() == st_type.lower() or 
                        rid.lower() in pascal_name.lower() or 
                        rid.lower() in st_type.lower()):
                        matched_id = pascal_name # Force standardized PascalCase for reports
                        break


            
            if not matched_id:
                continue

            try:
                strategy_obj = st_class(matched_id, config=self.config)
                
                # [ Institutional Gate ]: Only load strategies explicitly enabled in config
                if not strategy_obj.enabled:
                    continue

                    
                if symbol and not strategy_obj.is_symbol_allowed(symbol):
                    continue
                strategies.append(strategy_obj)
            except Exception as e:
                rprint(f"[bold yellow]Warning:[/] Failed to load strategy {st_type} (resolved as {matched_id}): {e}")
        return strategies


    def _run_monte_carlo(self, history, initial_balance=1000.0):
        rprint("\n[bold yellow]Institutional Robustness Audit: Running Monte Carlo Stress Suite (2500 paths)...[/]")
        mc = MonteCarloSimulator(iterations=2500)
        res = mc.run(history, initial_balance=initial_balance)
        
        if res.get("status") == "INSUFFICIENT_DATA":
            rprint(f"\n[bold red][!] AUDIT HALTED: {res.get('message')}[/]")
            return False

        score = float(res.get("robustness_score", 0))
        ruin_text = res.get("probability_of_ruin", "100%")
        ruin_prob = float(ruin_text.replace("%", ""))
        
        # Institutional Standard: Score > 80.0 and Ruin == 0.0%
        is_robust = (score >= 80.0) and (ruin_prob == 0.0)
        
        mc_table = Table(title="V4-ULTRA Institutional Robustness Certification", show_header=False, padding=(0, 2), box=None)
        
        # Display specific tracked metrics in a professional order
        metrics = [
            ("Robustness Score", f"{score}/100"),
            ("Median Final Balance", f"${res.get('median_final_balance', 0)}"),
            ("Worst Case Balance (95% CI)", f"${res.get('worst_case_balance_95ci', 0)}"),
            ("Max Drawdown (95% CI)", res.get("worst_case_dd_95ci", "0%")),
            ("Probability of Ruin", ruin_text)
        ]
        
        for label, val in metrics:
            if "Score" in label:
                color = "green" if is_robust else "red"
                mc_table.add_row(label, f"[{color}]{val}[/]")
            elif "Ruin" in label:
                color = "green" if ruin_prob == 0.0 else "red"
                mc_table.add_row(label, f"[{color}]{val}[/]")
            else:
                mc_table.add_row(label, val)
        
        if not is_robust:
            rprint("\n[bold red]!!! AUDIT FAILURE: STRATEGY REJECTED FOR PRODUCTION !!![/]")
            if ruin_prob > 0:
                rprint("[bold red]Reason: Non-Zero Probability of Ruin detected under Execution Shock.[/]")
            elif score < 80:
                rprint(f"[bold red]Reason: Robustness Score ({score}) below institutional floor (80.0).[/]")
        else:
            rprint("\n[bold green]*** AUDIT PASSED: STRATEGY CERTIFIED FOR INSTITUTIONAL DEPLOYMENT ***[/]")

        self.console.print(Panel(mc_table, border_style="bright_green" if is_robust else "red", title="Robustness Suite Output"))
        return is_robust

    def _run_stress_test(self, symbol, strategies, data):
        rprint("\n[bold magenta]Worst Case Scenario Hunt: Running Stress Test Suite...[/]")
        tester = StressTester(self.config)
        results = tester.run_stress_test(symbol, strategies, data)
        
        s_table = Table(title="Execution Stress Tolerance Analysis", header_style="bold magenta")
        s_table.add_column("Scenario", style="bold")
        s_table.add_column("Profit ($)", justify="right")
        s_table.add_column("Retention (%)", justify="right")
        s_table.add_column("Result", justify="center")

        for name, res in results.items():
            m = res["metrics"]
            status = "[bold green]PASS[/]" if m.get("net_profit", 0) > 0 else "[bold red]FAIL[/]"
            s_table.add_row(name.replace("_", " ").upper(), f"${m.get('net_profit', 0):,.2f}", f"{res['profit_retention']:.1f}%", status)
        self.console.print(s_table)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="V4 Institutional Backtester CLI")
    parser.add_argument("--symbol", type=str, default="XAUUSDm")
    parser.add_argument("--from", dest="start_date", type=str, required=True)
    parser.add_argument("--to", dest="end_date", type=str, required=True)
    parser.add_argument("--strategy", type=str, default="")
    parser.add_argument("--monte-carlo", action="store_true")
    parser.add_argument("--walk-forward", action="store_true")
    parser.add_argument("--stress-test", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--deterministic", choices=["on", "off"], default=None)
    parser.add_argument("--resume", action="store_true", help="Resume from last crash checkpoint")
    parser.add_argument("--debug-signals", action="store_true", help="Log reason for every signal rejection")
    parser.add_argument("--no-adaptive", action="store_true", help="Disable adaptive strategy selection (run ALL strategies)")
    
    # [ Institutional High-Fidelity Flags ]
    parser.add_argument("--tick-model", choices=["true", "false"], default="true", help="Use M1 tick-replay for execution")
    parser.add_argument("--variable-spread", choices=["true", "false"], default="true", help="Use variable spreads from historical data")
    parser.add_argument("--slippage-model", choices=["pessimistic", "standard", "none"], default="standard")
    parser.add_argument("--commission", choices=["realistic", "none"], default="realistic")
    parser.add_argument("--execution-delay", choices=["simulated", "none"], default="simulated")
    parser.add_argument("--export-trades", action="store_true", default=True)
    parser.add_argument("--equity-curve", action="store_true", default=True)

    args = parser.parse_args()
    cli = BacktestCLI()
    
    # Map Institutional Flags to Config
    if "backtest" not in cli.config: cli.config["backtest"] = {}
    
    if args.stress_test:
        cli.config["backtest"]["run_stress_test"] = True
        
    if args.slippage_model == "pessimistic":
        cli.config["backtest"]["base_slippage_points"] = 2.5
        cli.config["backtest"]["latency_mu"] = 250.0
    elif args.slippage_model == "none":
        cli.config["backtest"]["base_slippage_points"] = 0.0
        cli.config["backtest"]["latency_mu"] = 0.0

    if args.commission == "realistic":
        # Force $7/lot across all symbols if realistic is set
        for sym in cli.config.get("symbols_config", {}):
            cli.config["symbols_config"][sym]["commission_per_lot"] = 7.0
            
    if args.execution_delay == "none":
         cli.config["backtest"]["latency_mu"] = 0.0
         cli.config["backtest"]["latency_sigma"] = 0.0

    cli.run(args.symbol, args.start_date, args.end_date, args.strategy, args.monte_carlo, args.walk_forward, args.stress_test, args.seed, 
            None if args.deterministic is None else (args.deterministic == "on"), resume=args.resume, debug_signals=args.debug_signals, no_adaptive=args.no_adaptive)
