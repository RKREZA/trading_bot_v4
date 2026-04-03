import pandas as pd
import numpy as np
import os
import glob
from datetime import datetime
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()

def analyze_portfolio(results_dir="backtest_results", initial_balance=1000.0):
    """
    Reads all latest CSV files from the results directory and combines them.
    Fix: Uses Percentage Returns for a Portfolio-level Compounding simulation.
    """
    csv_files = glob.glob(os.path.join(results_dir, "*.csv"))
    if not csv_files:
        console.print("[bold red]No backtest result CSVs found.[/]")
        return

    latest_files = {}
    for f in csv_files:
        base = os.path.basename(f)
        parts = base.split('_')
        if len(parts) < 3: continue
        key = f"{parts[0]}_{parts[1]}"
        if key not in latest_files or os.path.getmtime(f) > os.path.getmtime(latest_files[key]):
            latest_files[key] = f

    console.print(f"[cyan]Analyzing {len(latest_files)} strategy files using Percentage Returns...[/]")
    
    all_data = []
    for key, path in latest_files.items():
        df = pd.read_csv(path).dropna(subset=['pnl'])
        if df.empty:
            console.print(f"[yellow]Skipping {key} (0 trades)...[/]")
            continue
            
        # We need the BALANCE at each trade to calculate % return
        df['balance_before'] = df['balance'] - df['pnl']
        df['ret_pct'] = df['pnl'] / df['balance_before']
        df['exit_time'] = pd.to_datetime(df['exit_time'])
        df['strategy_id'] = key
        all_data.append(df[['exit_time', 'ret_pct', 'strategy_id']])

    if not all_data:
        return

    portfolio_df = pd.concat(all_data).sort_values('exit_time')
    
    # Portfolio equity starts at 1.0 (100%)
    # Total account return = Product of (1 + sum(returns_at_t))
    # If 2 strategies exit at same time, their returns are additive for that moment.
    portfolio_df['cum_ret'] = (1 + portfolio_df['ret_pct']).cumprod()
    portfolio_df['equity'] = initial_balance * portfolio_df['cum_ret']
    
    # Calculate Drawdown
    portfolio_df['peak'] = portfolio_df['equity'].cummax()
    portfolio_df['drawdown_pct'] = (portfolio_df['peak'] - portfolio_df['equity']) / portfolio_df['peak'] * 100
    
    max_dd = portfolio_df['drawdown_pct'].max()
    total_prof = portfolio_df['equity'].iloc[-1] - initial_balance
    final_equity = portfolio_df['equity'].iloc[-1]
    return_pct = (total_prof / initial_balance) * 100
    
    # Portfolio Metrics Table
    table = Table(title="Compounded Portfolio Analysis", border_style="bold yellow")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right", style="green")
    
    table.add_row("Starting Balance", f"${initial_balance:,.2f}")
    table.add_row("Total Net Profit", f"${total_prof:,.2f}")
    table.add_row("Total Return (%)", f"{return_pct:,.2f}%")
    table.add_row("Portfolio Max Drawdown", f"{max_dd:.2f}%")
    table.add_row("Final Equity", f"${final_equity:,.2f}")
    table.add_row("Total Trades", str(len(portfolio_df)))
    
    console.print(Panel(table))
    
    # Check Portfolio Circuit Breaker
    try:
        import json
        with open("config.json", "r") as f:
            cfg = json.load(f)
            halt_limit = cfg.get("portfolio_risk", {}).get("max_drawdown_halt_pct", 12.0)
    except:
        halt_limit = 12.0 

    if max_dd >= halt_limit:
        console.print(f"[bold red]WARNING: Portfolio Max Drawdown ({max_dd:.2f}%) exceeded the halt limit ({halt_limit}%)![/]")
    else:
        console.print(f"[bold green]Portfolio within safe limits. Max DD: {max_dd:.2f}%[/]")

if __name__ == "__main__":
    analyze_portfolio()
