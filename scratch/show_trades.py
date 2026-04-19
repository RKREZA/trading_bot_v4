import sys
sys.path.append('.')
from comprehensive_backtest import ComprehensiveBacktestSuite
from datetime import datetime

suite = ComprehensiveBacktestSuite()
result = suite.run_single_strategy_backtest('PureBreakoutOneMinute', 'XAUUSDm', 'BULLISH', 3.0)

# Print all trades with details
print("\n" + "="*110)
print("ALL TRADES FROM BACKTEST - PureBreakoutOneMinute")
print("="*110)

if suite.last_history:
    wins = 0
    losses = 0
    total_pnl = 0
    
    for i, trade in enumerate(suite.last_history, 1):
        ts = trade.get('timestamp', 0)
        try:
            dt = datetime.fromtimestamp(ts)
            time_str = dt.strftime('%Y-%m-%d %H:%M')
        except:
            time_str = str(ts)
        
        direction = trade.get('direction', 'N/A')
        entry = trade.get('fill_price', 0)
        exit_price = trade.get('exit_price', 0)
        volume = trade.get('lots', 0)
        pnl = trade.get('pnl', 0)
        result_type = trade.get('result', 'N/A')
        tp = trade.get('tp', 0)
        sl = trade.get('sl', 0)
        
        total_pnl += pnl
        if pnl > 0:
            wins += 1
        else:
            losses += 1
        
        result_marker = "WIN" if pnl > 0 else "LOSS"
        
        print(f"{i:>3}. {time_str} | {direction:<4} | Entry: {entry:.5f} | Exit: {exit_price:.5f} | TP: {tp:.5f} | SL: {sl:.5f} | Lots: {volume:.4f} | P&L: ${pnl:>10.2f} | {result_marker}")
    
    print("="*110)
    print(f"SUMMARY: {len(suite.last_history)} trades | {wins} wins | {losses} losses | Win Rate: {wins/len(suite.last_history)*100:.1f}%")
    print(f"Total P&L: ${total_pnl:.2f}")
    print("="*110)

print("\nFinal Metrics:")
for k, v in result.items():
    if k not in ['strategy', 'market', 'status', 'monthly_stats']:
        print(f"  {k}: {v}")
