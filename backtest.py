"""
BACKTEST ENGINE V4 - Institutional Multi-Strategy Interface
High-fidelity historical simulation with Portfolio Governance.
"""

import argparse
import json
import os
import sys
import logging
from datetime import datetime, timezone, timedelta
from typing import Optional, List, Dict, Any

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import print as rprint

from dotenv import load_dotenv
from core import SourceHandler, MT5Connection, PerformanceTracker, setup_logging
from core.data.manager import DataManager
from backtesting import PortfolioBacktester, MonteCarloSimulator, StressTester, WalkForwardValidator
from strategies import create_strategy, STRATEGY_REGISTRY

load_dotenv()

class BacktestCLI:
    def __init__(self):
        self.console = Console()
        self.config = self._load_config("config.json")
        self.data_manager = DataManager(self.config)
        self.connection = MT5Connection()
        
    def _load_config(self, path: str) -> dict:
        if not os.path.exists(path):
            rprint(f"[bold red]Error:[/] {path} not found. Using defaults.")
            return {"initial_balance": 1000.0}
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

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
        debug_signals: bool = False
    ):
        setup_logging()
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
        rprint(Panel(f"[bold cyan]V4-ULTRA Production Benchmark[/]\nSymbol: {symbol} | Range: {start} to {end}", border_style="bright_blue"))
        
        with self.console.status("[bold green]Orchestrating Flawless MT5 Sync...") as status:
            m1 = self.data_manager.prepare_data(symbol, "M1", dt_from)
            m5 = self.data_manager.prepare_data(symbol, "M5", dt_from)
            m15 = self.data_manager.prepare_data(symbol, "M15", dt_from)
            h1 = self.data_manager.prepare_data(symbol, "H1", dt_from)
            
            # [ Institutional Alignment Filter ]: Cap all TFs by available M1 execution data
            # This prevents simulation crashes on last forming M15/H1 bars.
            if len(m1) > 0:
                max_t = m1.time[-1]
                m5 = m5[m5.time <= max_t]
                m15 = m15[m15.time <= max_t]
                h1 = h1[h1.time <= max_t]

        if len(m5) < 100:
            rprint("[bold red]Error:[/] Insufficient data for benchmark.")
            return

        # [ Range Capping ]: Slice all timeframes by dt_to
        to_ts = dt_to.timestamp()
        m1 = m1[m1.time < to_ts]
        m5 = m5[m5.time < to_ts]
        m15 = m15[m15.time < to_ts]
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
        
        rprint(f"[green]Executing V4-ULTRA Event-Driven Simulation on {primary_tf} (Resume={resume})...[/]")
        
        runtime_config = dict(self.config)
        runtime_backtest = dict(runtime_config.get("backtest", {}))
        if seed is not None:
            runtime_backtest["random_seed"] = int(seed)
        if deterministic is not None:
            runtime_backtest["deterministic"] = bool(deterministic)
        runtime_config["backtest"] = runtime_backtest
        if debug_signals:
            runtime_config["backtest"]["debug_signals"] = True
            rprint("[bold yellow]DIAGNOSTIC MODE ENABLED: Signal rejection reasons will be logged.[/]")

        backtester = PortfolioBacktester(runtime_config)
        history, equity_history = backtester.run(symbol, strategies, primary_data, h1, m15, m5, m1, resume=resume)
        
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
            "portfolio": stats,
            "strategies": strat_stats,
            "sessions": session_stats
        }

        # 6. Audit Pack Generation (Step 14)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        session_dir = f"backtest_results/session_{timestamp}"
        PerformanceTracker.save_audit_pack(history, full_results, session_dir)
        
        # 7. Dashboard Display
        dashboard = PerformanceTracker.generate_professional_dashboard(full_results)
        rprint(f"\n[bold white]{dashboard}[/]")
        rprint(f"\n[bold green]Institutional Audit Pack Persistence: {session_dir}[/]")

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
        for st_type, st_class in STRATEGY_REGISTRY.items():
            sid = f"{st_type.lower()}_v4"
            if strategy_filter and sid != strategy_filter and st_type != strategy_filter.upper():
                continue
            try:
                strategy_obj = st_class(sid, config=self.config)
                if symbol and not strategy_obj.is_symbol_allowed(symbol):
                    continue
                strategies.append(strategy_obj)
            except Exception as e:
                rprint(f"[bold yellow]Warning:[/] Failed to load {sid}: {e}")
        return strategies

    def _run_monte_carlo(self, history, initial_balance=1000.0):
        rprint("\n[bold yellow]Institutional Robustness Audit: Running Monte Carlo Stress Suite (2500 paths)...[/]")
        mc = MonteCarloSimulator(iterations=2500)
        res = mc.run(history, initial_balance=initial_balance)
        
        if res.get("status") == "INSUFFICIENT_DATA":
            rprint(f"\n[bold red]⚠ AUDIT HALTED: {res.get('message')}[/]")
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
    parser.add_argument("--strategy", type=str, default=None)
    parser.add_argument("--monte-carlo", action="store_true")
    parser.add_argument("--walk-forward", action="store_true")
    parser.add_argument("--stress-test", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--deterministic", choices=["on", "off"], default=None)
    parser.add_argument("--resume", action="store_true", help="Resume from last crash checkpoint")
    parser.add_argument("--debug-signals", action="store_true", help="Log reason for every signal rejection")

    args = parser.parse_args()
    cli = BacktestCLI()
    if args.stress_test:
        cli.config.setdefault("backtest", {})["run_stress_test"] = True

    cli.run(args.symbol, args.start_date, args.end_date, args.strategy, args.monte_carlo, args.walk_forward, args.stress_test, args.seed, 
            None if args.deterministic is None else (args.deterministic == "on"), resume=args.resume, debug_signals=args.debug_signals)
