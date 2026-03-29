import csv

def load(path):
    trades = []
    with open(path) as f:
        for r in csv.DictReader(f):
            trades.append({
                'pnl': float(r['pnl']),
                'result': r['result'],
                'direction': r['direction'],
                'time': r['time'],
            })
    return trades

def analyze(trades, label, out_f):
    closed = [t for t in trades if t['result'] in ('TP','SL','TSL')]
    w = [t for t in closed if t['pnl'] > 0]
    l = [t for t in closed if t['pnl'] <= 0]
    tp = len([t for t in closed if t['result']=='TP'])
    sl = len([t for t in closed if t['result']=='SL'])
    tsl = len([t for t in closed if t['result']=='TSL'])
    tsl_be = len([t for t in closed if t['result']=='TSL' and abs(t['pnl'])<0.5])
    total_w = sum(t['pnl'] for t in w) if w else 0
    total_l = abs(sum(t['pnl'] for t in l)) if l else 0
    pf = total_w / total_l if total_l > 0 else 0
    wr = len(w)/len(closed)*100 if closed else 0
    pnl = sum(t['pnl'] for t in closed)
    
    buys = [t for t in closed if t['direction']=='BUY']
    sells = [t for t in closed if t['direction']=='SELL']
    bw = [t for t in buys if t['pnl']>0]
    sw = [t for t in sells if t['pnl']>0]
    
    # Max loss streak
    mx, cur = 0, 0
    for t in closed:
        if t['pnl'] <= 0: cur += 1; mx = max(mx, cur)
        else: cur = 0
    
    out_f.write(f"\\n--- {label} ---\\n")
    out_f.write(f"Trades: {len(closed)}\\n")
    out_f.write(f"TP/SL/TSL: {tp}/{sl}/{tsl}  (TSL breakeven: {tsl_be})\\n")
    out_f.write(f"Win Rate: {wr:.1f}%\\n")
    out_f.write(f"Profit Factor: {pf:.2f}\\n")
    out_f.write(f"Total PnL: ${pnl:,.2f}\\n")
    if w: out_f.write(f"Avg Win: ${total_w/len(w):.2f}\\n")
    if l: out_f.write(f"Avg Loss: ${total_l/len(l):.2f}\\n")
    out_f.write(f"Max Loss Streak: {mx}\\n")
    if buys:
        out_f.write(f"BUY:  {len(buys)} trades, {len(bw)/len(buys)*100:.0f}% WR, ${sum(t['pnl'] for t in buys):,.2f}\\n")
    if sells:
        out_f.write(f"SELL: {len(sells)} trades, {len(sw)/len(sells)*100:.0f}% WR, ${sum(t['pnl'] for t in sells):,.2f}\\n")

before = load('backtest_results/XAUUSDm_trades_20260326_130208.csv')
after = load('backtest_results/XAUUSDm_trades_20260326_134552.csv')

with open('result.txt', 'w', encoding='utf-8') as f:
    analyze(before, "BEFORE (Phase 1)", f)
    analyze(after, "AFTER (Phase 2 - Win Rate Focus)", f)
