import sys
import os
import time
from dotenv import load_dotenv

# Add the root directory to path to import core
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.notifications.telegram_alerter import TelegramAlerter, AlertPriority, AlertCategory

def test_telegram_alerts():
    print("Loading .env file...")
    load_dotenv()
    
    alerter = TelegramAlerter()
    
    if not alerter.enabled:
        print("Telegram alerter is DISABLED. Please check TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in .env")
        return
        
    print("Telegram alerter is ENABLED. Sending test notifications...")
    
    # 1. Trade Alert
    print("Sending Trade Alert...")
    alerter.send_trade_alert(
        direction="BUY",
        symbol="XAUUSDm",
        volume=0.5,
        price=2345.67,
        sl=2335.00,
        tp=2365.00,
        strategy="SmartMeanReversion"
    )
    time.sleep(1) # Sleep briefly to maintain order in Telegram
    
    # 2. Regime Change
    print("Sending Regime Change Alert...")
    alerter.send_regime_change(
        old_regime="BULLISH",
        new_regime="VOLATILE"
    )
    time.sleep(1)
    
    # 3. Performance Summary
    print("Sending Performance Summary...")
    alerter.send_performance_summary(
        equity=10543.21,
        daily_pnl=143.50,
        dd=1.2,
        trades=4
    )
    time.sleep(1)
    
    # 4. Risk Alert
    print("Sending Risk Alert...")
    alerter.send_risk_alert(
        reason="Daily Drawdown Warning",
        details="Equity dropped by 3% today. Nearing daily max loss threshold."
    )
    time.sleep(1)
    
    # 5. Emergency Alert
    print("Sending Emergency Alert...")
    alerter.send_emergency_alert(
        reason="Broker Disconnected - Max Retries Exceeded. Flattening portfolio."
    )
    time.sleep(1)
    
    # 6. Generic Alert
    print("Sending Generic System Alert...")
    alerter.send_alert(
        message="<b>V5-INSIGNIA System Check</b>\nAll modules loaded correctly and ready for VPS live demo.",
        priority=AlertPriority.INFO,
        category=AlertCategory.SYSTEM
    )
    
    print("Wait 3 seconds for async queue to drain...")
    time.sleep(3)
    print("Test complete.")

if __name__ == "__main__":
    test_telegram_alerts()
