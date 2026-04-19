#!/usr/bin/env python
"""
INSTITUTIONAL AUDIT ENGINE
========================
Automated testing and backtesting script for all strategies.
Generates comprehensive audit report.
"""

import os
import sys
import json
import logging
import subprocess
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

import MetaTrader5 as mt5
from dotenv import load_dotenv

load_dotenv()

console = Console()

class AuditEngine:
    """Institutional Audit Engine for Strategy Certification."""
    
    def __init__(self):
        self.results = {}
        self.strategies = {
            "TrendFollowing": {"enabled": False, "class": "TrendFollowingStrategy"},
            "LiquiditySweepBreakout": {"enabled": True, "class": "LiquiditySweepBreakoutStrategy"},
            "RangeBounce": {"enabled": False, "class": "RangeBounceStrategy"},
            "SmartMeanReversion": {"enabled": True, "class": "SmartMeanReversionStrategy"},
            "LiquiditySession": {"enabled": False, "class": "LiquiditySessionStrategy"},
            "SMASampleStrategy": {"enabled": True, "class": "SMASampleStrategy"}
        }
        
    def run_audit(self, symbol: str = "XAUUSDm", start: str = "2025-01-01", end: str = "2026-04-09"):
        """Run complete audit for all strategies."""
        
        console.print("\n[bold cyan]══════════════════════════════════════════════════════════════[/]")
        console.print("[bold cyan]  V5-INSIGNIA INSTITUTIONAL AUDIT ENGINE         [/]")
        console.print("[bold cyan]══════════════════════════════════════════════════════════════[/]\n")
        
        console.print(f"[bold]Symbol:[/bold] {symbol}")
        console.print(f"[bold]Period:[/bold] {start} to {end}")
        console.print(f"[bold]Strategies:[/bold] {len(self.strategies)}")
        
        # Test each strategy
        for strategy_name, config in self.strategies.items():
            with console.status(f"[bold green]Testing {strategy_name}...[/]"):
                try:
                    result = self._test_strategy(strategy_name, symbol, start, end)
                    self.results[strategy_name] = result
                except Exception as e:
                    console.print(f"[red]Error testing {strategy_name}: {e}[/]")
                    self.results[strategy_name] = {"status": "ERROR", "error": str(e)}
        
        return self._generate_report()
    
    def _test_strategy(self, strategy_name: str, symbol: str, start: str, end: str) -> Dict[str, Any]:
        """Test a single strategy."""
        
        # Enable only the target strategy in config
        config_updates = self._modify_config(strategy_name, True)
        
        if config_updates is None:
            return {"status": "CONFIG_ERROR", "error": "Failed to update config"}
        
        try:
            # Run backtest
            cmd = [
                sys.executable, "backtest.py",
                "--symbol", symbol,
                "--from", start,
                "--to", end,
                "--strategy", strategy_name,
                "--no-adaptive",
                "--variable-spread", "true",
                "--slippage-model", "standard"
            ]
            
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                timeout=300
            )
            
            if result.returncode == 0:
                # Parse output for metrics
                return self._parse_backtest_output(result.stdout)
            else:
                return {
                    "status": "FAILED",
                    "returncode": result.returncode,
                    "stderr": result.stderr[:500] if result.stderr else result.stdout[:500]
                }
                
        except subprocess.TimeoutExpired:
            return {"status": "TIMEOUT", "error": "Backtest timed out"}
        except Exception as e:
            return {"status": "ERROR", "error": str(e)}
        finally:
            # Restore original config
            self._restore_config()
    
    def _modify_config(self, strategy_name: str, enabled: bool) -> Optional[str]:
        """Modify config to enable only the target strategy."""
        try:
            import json
            config_path = "configs/symbols/XAUUSDm.json"
            
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            # Disable all, enable target
            for strat in config.get("strategies", {}):
                config["strategies"][strat]["enabled"] = False
            
            if strategy_name in config.get("strategies", {}):
                config["strategies"][strategy_name]["enabled"] = enabled
            
            # Save backup
            self._config_backup = config
            
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
            
            return "OK"
            
        except Exception as e:
            console.print(f"[red]Config error: {e}[/]")
            return None
    
    def _restore_config(self):
        """Restore original config."""
        try:
            import json
            if hasattr(self, '_config_backup'):
                config_path = "configs/symbols/XAUUSDm.json"
                with open(config_path, 'w') as f:
                    json.dump(self._config_backup, f, indent=2)
        except:
            pass
    
    def _parse_backtest_output(self, output: str) -> Dict[str, Any]:
        """Parse backtest output to extract metrics."""
        metrics = {}
        
        # Extract key metrics from output
        lines = output.split('\n')
        for line in lines:
            if 'Total Net Profit' in line or 'Net Profit' in line:
                try:
                    val = line.split('$')[-1].replace(',', '').strip()
                    metrics['net_profit'] = float(val)
                except:
                    pass
            if 'Win Rate' in line or 'GrossProfit' in line:
                try:
                    parts = line.split(':')
                    if len(parts) > 1:
                        metrics['win_rate'] = float(parts[1].strip().replace('%', ''))
                except:
                    pass
            if 'Profit Factor' in line:
                try:
                    val = line.split(':')[-1].strip()
                    metrics['profit_factor'] = float(val)
                except:
                    pass
            if 'Max Drawdown' in line or 'MaxDD' in line:
                try:
                    val = line.split(':')[-1].strip().replace('%', '')
                    metrics['max_drawdown'] = float(val)
                except:
                    pass
            if 'Sharpe Ratio' in line:
                try:
                    val = line.split(':')[-1].strip()
                    metrics['sharpe_ratio'] = float(val)
                except:
                    pass
        
        if not metrics:
            metrics = {"status": "COMPLETED", "note": "Output parsing incomplete"}
        
        return metrics
    
    def _generate_report(self) -> str:
        """Generate audit report."""
        
        console.print("\n[bold cyan]══════════════════════════════════════════════════════════════[/]")
        console.print("[bold cyan]  INSTITUTIONAL AUDIT REPORT                       [/]")
        console.print("[bold cyan]══════════════════════════════════════════════════════════════[/]\n")
        
        # Create results table
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Strategy", style="cyan", width=25)
        table.add_column("Status", width=12)
        table.add_column("Net Profit", justify="right", width=12)
        table.add_column("Win Rate", justify="right", width=10)
        table.add_column("Profit Factor", justify="right", width=10)
        table.add_column("Max DD", justify="right", width=10)
        
        for strategy, result in self.results.items():
            status = result.get('status', 'UNKNOWN')
            status_color = "green" if status == "COMPLETED" else "yellow" if status == "TIMEOUT" else "red"
            
            net_profit = f"${result.get('net_profit', 0):,.2f}" if 'net_profit' in result else "N/A"
            win_rate = f"{result.get('win_rate', 0):.1f}%" if 'win_rate' in result else "N/A"
            pf = f"{result.get('profit_factor', 0):.2f}" if 'profit_factor' in result else "N/A"
            max_dd = f"{result.get('max_drawdown', 0):.2f}%" if 'max_drawdown' in result else "N/A"
            
            table.add_row(
                strategy,
                f"[{status_color}]{status}[/{status_color}]",
                net_profit,
                win_rate,
                pf,
                max_dd
            )
        
        console.print(table)
        
        return "Audit Complete"


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="V5-INSIGNIA Audit Engine")
    parser.add_argument("--symbol", type=str, default="XAUUSDm", help="Trading symbol")
    parser.add_argument("--from", type=str, default="2025-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--to", type=str, default="2026-04-09", help="End date (YYYY-MM-DD)")
    
    args = parser.parse_args()
    
    engine = AuditEngine()
    engine.run_audit(args.symbol, args.from_, args.to)


if __name__ == "__main__":
    main()