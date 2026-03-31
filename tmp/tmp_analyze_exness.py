import MetaTrader5 as mt5
from datetime import datetime, timezone, timedelta
import pandas as pd
import sys

def analyze_exness_trades():
    # Redirect output to file
    with open("exness_analysis.txt", "w") as f:
        sys.stdout = f
        
        if not mt5.initialize():
            print("Failed to initialize MT5")
            return

        # Fetch deals for the last 7 days
        now = datetime.now()
        start = now - timedelta(days=7)
        
        deals = mt5.history_deals_get(start, now)
        if deals is None or len(deals) == 0:
            print("No trades found in the last 7 days.")
            mt5.shutdown()
            return

        print(f"Total deals fetched: {len(deals)}")
        
        # Process deals
        trades = []
        for d in deals:
            if d.entry == mt5.DEAL_ENTRY_OUT:
                trades.append({
                    "ticket": d.ticket,
                    "position_id": d.position_id,
                    "symbol": d.symbol,
                    "volume": d.volume,
                    "profit": d.profit,
                    "commission": d.commission,
                    "swap": d.swap,
                    "reason": d.reason,
                    "time": datetime.fromtimestamp(d.time)
                })
                
        if not trades:
            print("No OUT deals found.")
            mt5.shutdown()
            return
            
        df = pd.DataFrame(trades)
        
        wins = df[df['profit'] > 0]
        losses = df[df['profit'] < 0]
        breakevens = df[df['profit'] == 0]
        
        print("\n--- TRADE ANALYSIS ---")
        print(f"Total Trades (OUT): {len(df)}")
        print(f"Wins: {len(wins)} ({len(wins)/len(df)*100:.1f}%)")
        print(f"Losses: {len(losses)} ({len(losses)/len(df)*100:.1f}%)")
        print(f"Breakeven: {len(breakevens)}")
        
        print(f"\nNet Profit: ${df['profit'].sum():.2f}")
        if len(wins) > 0:
            print(f"Average Win: ${wins['profit'].mean():.2f}")
        if len(losses) > 0:
            print(f"Average Loss: ${losses['profit'].mean():.2f}")
        
        print(f"Total Commission paid: ${df['commission'].sum():.2f}")
        
        # Reason distribution
        # 0 = Client, 1 = Mobile, 2 = Web, 3 = Expert, 4 = SL, 5 = TP, 6 = SO, 7 = Rollover, ...
        reason_map = {
            0: "Client", 1: "Mobile", 2: "Web", 3: "Expert (Bot)", 4: "Stop Loss", 5: "Take Profit", 
            6: "Stop Out", 7: "Rollover", 16: "Partial Close"
        }
        df['reason_str'] = df['reason'].map(lambda x: reason_map.get(x, f"Other({x})"))
        
        print("\n--- EXIT REASONS ---")
        print(df['reason_str'].value_counts().to_string())
        
        # Display all losses to identify problems
        print("\n--- ALL LOSSES ---")
        all_losses = df[df['profit'] < 0].sort_values(by='time', ascending=False)
        for _, row in all_losses.iterrows():
            print(f"Time: {row['time']}, Symbol: {row['symbol']}, Vol: {row['volume']}, Profit: ${row['profit']:.2f}, Reason: {row['reason_str']}")
            
        print("\n--- ALL TRADES ---")
        for _, row in df.sort_values(by='time', ascending=False).iterrows():
            print(f"Time: {row['time']}, Symbol: {row['symbol']}, Vol: {row['volume']}, Profit: ${row['profit']:.2f}, Reason: {row['reason_str']}")
            
        mt5.shutdown()

    sys.stdout = sys.__stdout__

if __name__ == "__main__":
    analyze_exness_trades()
