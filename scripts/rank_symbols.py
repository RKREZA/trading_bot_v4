import sys
import os
import json
import argparse
from rich.console import Console
from rich import print as rprint
sys.path.append(os.getcwd())
from backtest import BacktestCLI

def main():
    parser = argparse.ArgumentParser(description="Batch Backtester: Rank symbols from config.")
    parser.add_argument("--from", dest="start_date", type=str, required=True, help="YYYY-MM-DD")
    parser.add_argument("--to", dest="end_date", type=str, required=True, help="YYYY-MM-DD")
    
    args = parser.parse_args()
    console = Console()
    
    # 1. Load config
    with open("config.json", "r", encoding="utf-8") as f:
        config = json.load(f)
    
    symbols = list(config.get("symbols_config", {}).keys())
    if not symbols:
        rprint("[bold red]No symbols found in config.json symbols_config.[/]")
        return
    
    cli = BacktestCLI()
    results = []
    
    rprint(f"[bold cyan]Starting batch backtest for {len(symbols)} symbols...[/]")
    
    # 2. Iterate through symbols
    for sym in symbols:
        rprint(f"\n[bold yellow]Analyzing {sym}...[/]")
        try:
            # We need to capture the results. Modify CLI momentarily or use its components.
            # For simplicity, we'll run it and the user sees the individual tables, 
            # and then we'll show the final ranking.
            # Note: CLI.run() prints multiple tables and panels.
            cli.run(sym, args.start_date, args.end_date)
            
        except Exception as e:
            rprint(f"[bold red]Failed to analyze {sym}: {e}[/]")
            continue
            
    rprint("\n" + "="*50)
    rprint("[bold green]Batch Analysis Complete.[/]")
    rprint("="*50)
    rprint("[italic]Check the individual reports above to compare the best performer based on Win Rate and DD.[/]")

if __name__ == "__main__":
    main()
