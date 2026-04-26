import os
import sys
import time
import logging
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from rich.console import Console
from rich.table import Table
from rich.progress import track
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.connection import MT5Connection
from core.data.source_handler import SourceHandler
from core.common.types import CandleArray

console = Console()

def is_big_candle(candles, i, avg_body):
    body = abs(candles.close[i] - candles.open[i])
    total_range = candles.high[i] - candles.low[i]
    if total_range == 0: return False
    
    # Criteria 1: Big body relative to its own range (small wicks)
    wick_ratio = body / total_range
    if wick_ratio < 0.7: return False # Body must be at least 70% of range
    
    # Criteria 2: Big body relative to recent average
    if body < avg_body * 1.5: return False # Must be 50% larger than average
    
    return True

def detect_refined_n_patterns(candles):
    patterns = []
    # Calculate rolling average body size for comparison
    bodies = np.abs(candles.close - candles.open)
    avg_bodies = pd.Series(bodies).rolling(window=20).mean().fillna(0).values
    
    for i in range(20, len(candles) - 10):
        if is_big_candle(candles, i, avg_bodies[i]):
            impulse_candle = candles[i]
            is_bullish = impulse_candle.close > impulse_candle.open
            
            c_high, c_low = impulse_candle.high, impulse_candle.low
            c_range = c_high - c_low
            
            # 90% Retracement Level
            if is_bullish:
                retrace_level = c_high - (c_range * 0.9)
                # Look ahead for a candle that touches or goes below this level
                for j in range(i + 1, min(i + 20, len(candles))):
                    if candles.low[j] <= retrace_level:
                        patterns.append({
                            'type': 'N (Refined)',
                            'impulse_index': i,
                            'impulse_price': c_high,
                            'retrace_index': j,
                            'retrace_price': candles.low[j],
                            'level_retrace': retrace_level,
                            'is_bullish': True
                        })
                        break
            else:
                retrace_level = c_low + (c_range * 0.9)
                # Look ahead for a candle that touches or goes above this level
                for j in range(i + 1, min(i + 20, len(candles))):
                    if candles.high[j] >= retrace_level:
                        patterns.append({
                            'type': 'Inverse N (Refined)',
                            'impulse_index': i,
                            'impulse_price': c_low,
                            'retrace_index': j,
                            'retrace_price': candles.high[j],
                            'level_retrace': retrace_level,
                            'is_bullish': False
                        })
                        break
    return patterns

def plot_pattern(symbol, timeframe, candles, patterns, filename):
    if not patterns:
        return
    
    plt.figure(figsize=(15, 8))
    
    # Use the most recent pattern to determine zoom window
    last_pat = patterns[-1]
    start_zoom = max(0, last_pat['impulse_index'] - 20)
    end_zoom = min(len(candles), last_pat['retrace_index'] + 30)
    
    # Plot candles as vertical lines (High-Low)
    for i in range(start_zoom, end_zoom):
        color = 'green' if candles.close[i] >= candles.open[i] else 'red'
        plt.vlines(i, candles.low[i], candles.high[i], color=color, linewidth=1.5, alpha=0.8)
        plt.vlines(i, min(candles.open[i], candles.close[i]), max(candles.open[i], candles.close[i]), color=color, linewidth=6)

    for pat in patterns:
        # Only plot if within zoom window
        if pat['impulse_index'] < start_zoom or pat['retrace_index'] > end_zoom:
            continue
            
        i_idx = pat['impulse_index']
        r_idx = pat['retrace_index']
        level_retrace = pat['level_retrace']
        
        color = 'blue' if pat['is_bullish'] else 'purple'
        
        # Highlight impulse candle
        plt.axvspan(i_idx - 0.5, i_idx + 0.5, color='yellow', alpha=0.4)
        
        # Draw 90% retrace line
        plt.hlines(level_retrace, i_idx, r_idx, color='orange', linestyle='--', linewidth=2)
        plt.text(i_idx, level_retrace, ' 90% Entry Level', color='orange', fontsize=10, fontweight='bold', verticalalignment='bottom')
        
        # Draw path
        plt.annotate('', xy=(r_idx, pat['retrace_price']), xytext=(i_idx, pat['impulse_price']),
                     arrowprops=dict(arrowstyle="->", color=color, lw=3, mutation_scale=20))
        
        # Labels
        plt.text(i_idx, pat['impulse_price'], ' IMPULSE', color='black', fontsize=11, fontweight='bold', 
                 bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
        plt.text(r_idx, pat['retrace_price'], ' RETRACE HIT', color='red', fontsize=11, fontweight='bold',
                 bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

    plt.title(f"{symbol} {timeframe} - ZOOMED Refined N-Patterns", fontsize=14, fontweight='bold')
    plt.xlim(start_zoom, end_zoom)
    
    # Set Y limits with some padding
    visible_low = min(candles.low[start_zoom:end_zoom])
    visible_high = max(candles.high[start_zoom:end_zoom])
    padding = (visible_high - visible_low) * 0.1
    plt.ylim(visible_low - padding, visible_high + padding)
    
    plt.grid(True, alpha=0.3)
    
    # Save to artifacts directory (assuming it's available)
    # Since I don't know the exact path for user's artifacts for the script to use, 
    # I'll save to 'scratch/plots/' and then I can use them.
    os.makedirs("scratch/plots", exist_ok=True)
    full_path = os.path.join("scratch/plots", filename)
    plt.savefig(full_path)
    plt.close()
    return full_path

def main():
    symbol = "XAUUSDm"
    timeframes = ["M1"]
    
    conn = MT5Connection()
    if not conn.connect():
        console.print("[red]Failed to connect to MT5[/red]")
        return
    
    handler = SourceHandler()
    
    results_table = Table(title=f"90% Retrace N-Patterns for {symbol} (M1)")
    results_table.add_column("Timeframe", style="cyan")
    results_table.add_column("Pattern Type", style="magenta")
    results_table.add_column("Impulse Time", style="green")
    results_table.add_column("Retrace Time (90%+)", style="green")
    results_table.add_column("Retrace Price", style="yellow")

    all_patterns_found = []

    for tf in track(timeframes, description="Processing timeframes..."):
        candles = handler.fetch_candles(symbol, tf, count=5000)
        if len(candles) < 50:
            console.print(f"[yellow]Insufficient data for {tf}[/yellow]")
            continue
            
        patterns = detect_refined_n_patterns(candles)
        
        if patterns:
            # Only take the last 3 patterns to avoid cluttered plots
            recent_patterns = patterns[-3:]
            filename = f"refined_patterns_{symbol}_{tf}.png"
            plot_path = plot_pattern(symbol, tf, candles, recent_patterns, filename)
            
            for pat in patterns:
                # Convert index to time string
                time_i = datetime.datetime.fromtimestamp(candles.time[pat['impulse_index']]).strftime('%Y-%m-%d %H:%M')
                time_r = datetime.datetime.fromtimestamp(candles.time[pat['retrace_index']]).strftime('%Y-%m-%d %H:%M')
                
                results_table.add_row(
                    tf,
                    pat['type'],
                    time_i,
                    time_r,
                    f"{pat['retrace_price']:.2f}"
                )
                all_patterns_found.append({
                    'timeframe': tf,
                    'type': pat['type'],
                    'plot': plot_path
                })
        else:
            results_table.add_row(tf, "NONE FOUND", "-", "-", "-")

    console.print(results_table)
    
    if all_patterns_found:
        console.print(f"\n[green]Successfully generated plots in scratch/plots/[/green]")
    else:
        console.print("\n[yellow]No patterns detected in the last 500 bars for any timeframe.[/yellow]")

    conn.disconnect()

if __name__ == "__main__":
    main()
