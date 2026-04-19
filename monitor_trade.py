import MetaTrader5 as mt5
from datetime import datetime, timedelta
import time

def monitor_trade():
    mt5.initialize()
    account = mt5.account_info()
    start_balance = account.balance
    start_equity = account.equity
    start_time = datetime.now()

    # End time = start + 1 hour
    end_time = start_time + timedelta(hours=1)

    print("=" * 60)
    print("TRADE MONITOR - 1 HOUR SESSION")
    print("=" * 60)
    print("Start:", start_time.strftime("%H:%M:%S"))
    print("Expected End:", end_time.strftime("%H:%M:%S"))
    print()

    check_count = 0
    while datetime.now() < end_time:
        mt5.initialize()
        positions = mt5.positions_get()
        account = mt5.account_info()

        for p in positions:
            entry = p.price_open
            current = p.price_current
            sl = p.sl
            tp = p.tp

            dist_to_sl = sl - current if p.type == 1 else current - sl
            dist_to_tp = current - tp if p.type == 1 else tp - current

            print("[" + datetime.now().strftime("%H:%M:%S") + "] Check #" + str(check_count + 1))
            print("  Type: SELL" if p.type == 1 else "  Type: BUY")
            print("  Entry:", entry)
            print("  Current:", current)
            print("  SL:", sl, "dist:", round(dist_to_sl, 3))
            print("  TP:", tp, "dist:", round(dist_to_tp, 3))
            print("  Profit:", round(p.profit, 2))
            print("  Balance:", round(account.balance, 2), "| Equity:", round(account.equity, 2))
            print()

        if not positions:
            print("[" + datetime.now().strftime("%H:%M:%S") + "] NO POSITIONS - Trade closed!")
            print("Final Balance:", round(account.balance, 2))
            print("Final Equity:", round(account.equity, 2))
            mt5.shutdown()
            return

        mt5.shutdown()
        check_count += 1
        time.sleep(300)  # 5 minutes

    # Final check
    mt5.initialize()
    account = mt5.account_info()
    positions = mt5.positions_get()

    print("=" * 60)
    print("1 HOUR MONITORING COMPLETE")
    print("=" * 60)
    print("Start Equity:", round(start_equity, 2))
    print("End Equity:", round(account.equity, 2))
    print("Change:", round(account.equity - start_equity, 2))
    print()

    if positions:
        for p in positions:
            print("STATUS: POSITION STILL OPEN")
            print("  Entry:", p.price_open)
            print("  Current:", p.price_current)
            print("  SL:", p.sl)
            print("  TP:", p.tp)
            print("  Profit:", round(p.profit, 2))
    else:
        print("STATUS: POSITION CLOSED")

    mt5.shutdown()

if __name__ == "__main__":
    monitor_trade()
