
import sys
import os
import time
from dotenv import load_dotenv

# Add project root to path
sys.path.append(os.getcwd())

from core.notifications.telegram_alerter import TelegramAlerter, AlertPriority, AlertCategory

def test_all_notifications():
    load_dotenv()
    
    alerter = TelegramAlerter()
    
    if not alerter.enabled:
        print("TelegramAlerter is not enabled. Check your .env file for TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID.")
        return

    print("Sending test notifications...")

    # 1. Generic Info Alert
    print("- Sending Generic Info Alert")
    alerter.send_alert("This is a <b>Generic Info</b> test alert.", priority=AlertPriority.INFO)
    time.sleep(1)

    # 2. Generic Warning Alert
    print("- Sending Generic Warning Alert")
    alerter.send_alert("This is a <b>Generic Warning</b> test alert.", priority=AlertPriority.WARNING)
    time.sleep(1)

    # 3. Trade Alert
    print("- Sending Trade Alert")
    alerter.send_trade_alert(
        direction="BUY",
        symbol="XAUUSDm",
        volume=0.1,
        price=2345.67,
        sl=2330.00,
        tp=2380.00,
        strategy="NPatternGrid"
    )
    time.sleep(1)

    # 4. Risk Alert
    print("- Sending Risk Alert")
    alerter.send_risk_alert(
        reason="Drawdown Threshold",
        details="Current drawdown at 4.5%, nearing 5% halt limit."
    )
    time.sleep(1)

    # 5. Emergency Alert
    print("- Sending Emergency Alert")
    alerter.send_emergency_alert(
        reason="Manual Kill-Switch Triggered"
    )
    time.sleep(1)

    # 6. Performance Summary
    print("- Sending Performance Summary")
    alerter.send_performance_summary(
        equity=10543.21,
        daily_pnl=143.21,
        dd=1.2,
        trades=8
    )
    time.sleep(1)

    # 7. Regime Change
    print("- Sending Regime Change")
    alerter.send_regime_change(
        old_regime="LOW_VOL_RANGE",
        new_regime="HIGH_VOL_TREND"
    )
    
    print("\nAll test notifications queued. Waiting 5 seconds for async delivery...")
    time.sleep(5)
    print("Done.")

if __name__ == "__main__":
    test_all_notifications()
