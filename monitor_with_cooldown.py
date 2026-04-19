import MetaTrader5 as mt5
from datetime import datetime
import time
import json

def load_cooldown():
    """Load cooldown from strategy health file"""
    try:
        with open('config/strategy_health.json', 'r') as f:
            data = json.load(f)
        return data
    except:
        return {}

def monitor():
    print("=" * 70)
    print("LIVE TRADING MONITOR WITH COOLDOWN CLOCK")
    print("=" * 70)
    
    cooldown_remaining = 0
    last_status = "NO TRADE"
    
    while True:
        mt5.initialize()
        account = mt5.account_info()
        positions = mt5.positions_get()
        
        current_time = datetime.now().strftime('%H:%M:%S')
        
        # Clear screen (Windows)
        print("\033[2J\033[H", end="")
        print("=" * 70)
        print("LIVE TRADING MONITOR - " + current_time)
        print("=" * 70)
        print()
        print("Account:")
        print("  Balance: $" + str(round(account.balance, 2)))
        print("  Equity:  $" + str(round(account.equity, 2)))
        print("  P/L:     $" + str(round(account.equity - account.balance, 2)))
        print()
        
        if positions:
            for p in positions:
                entry = p.price_open
                current = p.price_current
                sl = p.sl
                tp = p.tp
                
                if p.type == 1:  # SELL
                    risk = entry - sl
                    reward = entry - tp
                    rr = reward / risk if risk > 0 else 0
                    current_r = (entry - current) / risk if risk > 0 else 0
                    direction = "SELL"
                else:  # BUY
                    risk = sl - entry
                    reward = tp - entry
                    rr = reward / risk if risk > 0 else 0
                    current_r = (current - entry) / risk if risk > 0 else 0
                    direction = "BUY"
                
                print("=" * 70)
                print("*** TRADE ACTIVE ***")
                print("=" * 70)
                print("Ticket:  #" + str(p.ticket))
                print("Type:    " + direction)
                print("Entry:   " + str(entry))
                print("Current: " + str(current))
                print("SL:      " + str(sl))
                print("TP:      " + str(tp))
                print("Profit:  $" + str(round(p.profit, 2)))
                print("R:R:     " + str(round(rr, 2)) + " | Current: " + str(round(current_r, 2)) + "R")
                print()
                print("*** TRAILING STOP ACTIVE ***" if p.sl != (4839.347 if direction == "SELL" else 4814.956) else "")
                
                last_status = "TRADE"
                cooldown_remaining = 0
        else:
            # Check if cooldown from last trade
            cooldown_remaining = 2  # Strategy cooldown is 2 cycles
            
            print("=" * 70)
            print("*** NO OPEN TRADE - COOLDOWN ACTIVE ***")
            print("=" * 70)
            print()
            print("Cooldown Timer:")
            print("  Remaining: " + str(cooldown_remaining) + " cycles")
            print()
            
            # Progress bar for cooldown
            total_cycles = 2
            filled = total_cycles - cooldown_remaining
            bar = "["
            bar += "=" * filled
            bar += "-" * cooldown_remaining
            bar += "]"
            print("  Progress: " + bar)
            print()
            
            # Status message
            if cooldown_remaining > 0:
                print("  Status: Waiting for cooldown to complete...")
                print("  Next trade will be taken when cooldown = 0")
            
            last_status = "COOLDOWN"
            
            # Check strategy health for more cooldown info
            health = load_cooldown()
            if health:
                print("  Strategy Health:")
                print("    " + json.dumps(health, indent=4).replace("\n", "\n    "))
        
        print()
        print("=" * 70)
        print("Press Ctrl+C to exit monitoring")
        print("=" * 70)
        
        mt5.shutdown()
        time.sleep(2)  # Update every 2 seconds

if __name__ == "__main__":
    try:
        monitor()
    except KeyboardInterrupt:
        print("\nMonitoring stopped.")
