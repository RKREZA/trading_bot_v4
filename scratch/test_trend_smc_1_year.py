import logging
import sys
import numpy as np
from datetime import datetime, timezone, timedelta
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
load_dotenv()

from core.config.loader import ConfigLoader
from core.connection import MT5Connection
from core.data.manager import DataManager
from backtesting import PortfolioBacktester
from strategies import create_strategy
from core.base_strategy import MarketData

def run_smc_test():
    config = ConfigLoader().global_config
    conn = MT5Connection()
    if not conn.connect():
        logging.error("MT5 connection failed")
        sys.exit(1)

    symbol = "XAUUSDm"
    # Last 1 year
    dt_to = datetime.now(timezone.utc)
    dt_from = dt_to - timedelta(days=365)

    logging.info(f"Fetching data for {symbol} from {dt_from.date()} to {dt_to.date()}...")
    dm = DataManager(config)
    m1 = dm.prepare_data(symbol, 'M1', dt_from)
    m5 = dm.prepare_data(symbol, 'M5', dt_from)
    m15 = dm.prepare_data(symbol, 'M15', dt_from)
    h1 = dm.prepare_data(symbol, 'H1', dt_from)
    d1 = dm.prepare_data(symbol, 'D1', dt_from)

    if m5 is None or len(m5) == 0:
        logging.error("Failed to fetch M5 data.")
        return

    logging.info(f"Data ready. M5 Bars: {len(m5)}, M15 Bars: {len(m15)}")

    # Overrides for testing
    config['backtest'] = {'debug_signals': False}
    config['ai_layer'] = {'enabled': False, 'regime_detection': False}
    
    # Single strategy override
    config['strategies'] = {
        "TrendFollowing": {
            "enabled": True,
            "allowed_sessions": ["LONDON", "NEW_YORK", "TOKYO", "ROLLOVER", "GLOBAL", "LONDON/NY"],
            "fvg_min_size": 0.3,
            "poc_window": 50,
            "displacement_threshold": 1.1,
            "volume_spike_factor": 1.2,
            "min_confidence": 0.50
        }
    }
    
    strat = create_strategy("TrendFollowing", "TrendFollowing", config)
    strat.enabled = True
    
    # Initialize Backtester
    bt = PortfolioBacktester(config)
    bt.strategies = {"TrendFollowing": strat}
    
    logging.info("Starting SMC TrendFollowing Backtest...")
    from dateutil.relativedelta import relativedelta
    
    # Let's use the backtester engine (simulating loop)
    # We will just run it manually since bt.run() might do all strategies.
    
    # We'll run the manual loop like `run_backtest.py` does internally, or just call generating signals
    # Since `PortfolioBacktester` relies on a full tick/bar stream, let's use the simplest loop:
    
    trades = 0
    wins = 0
    equity = 10000.0
    rejections = {}
    session_trades = {}
    
    from core.session_detector import SessionDetector
    
    # Simple loop simulation (M15 resolution for speed)
    for i in range(200, len(m15)):
        current_time = m15.time[i]
        
        # Adjust arrays view natively
        m15.set_limit(i+1)
        h1.set_limit(min(len(h1), int(i/4)+1))
        
        md_timestamp = datetime.fromtimestamp(m15.time[i], tz=timezone.utc)
        
        actual_session = SessionDetector.get_session(md_timestamp)
        
        md = MarketData(
            symbol=symbol,
            htf_candles=h1,
            m15_candles=m15,
            m5_candles=m5,
            d1_candles=d1,
            current_price=m15.c[i],
            bid=m15.c[i],
            ask=m15.c[i]+0.1,
            spread=0.1,
            point=0.01,
            session=actual_session,
            timestamp=md_timestamp
        )
        
        sig = strat.generate_signal(md)
        if sig:
            # We got a trade intent!
            risk = abs(sig.price - sig.stop_loss)
            reward = abs(sig.take_profit - sig.price)
            rr = reward / risk if risk > 0 else 0
            
            logging.info(f"Signal: {sig.direction} @ {sig.price:.2f} | SL: {sig.stop_loss:.2f} | TP: {sig.take_profit:.2f} | R:R: {rr:.2f}")
            
            trades += 1
            session_trades[actual_session] = session_trades.get(actual_session, 0) + 1
        else:
            reason = getattr(strat, 'last_rejection_reason', '')
            if not reason: reason = 'Silent Rejection (Missing data/cooldown)'
            rejections[reason] = rejections.get(reason, 0) + 1

    logging.info(f"--- SMC Trend Following Report ---")
    logging.info(f"Total True Institutional Signals Fired: {trades}")
    
    logging.info("--- Session Distribution ---")
    for s, c in sorted(session_trades.items(), key=lambda x: x[1], reverse=True):
        logging.info(f"{s}: {c} trades")
    
    logging.info("--- Rejections summary ---")
    for r, c in sorted(rejections.items(), key=lambda x: x[1], reverse=True)[:10]:
        logging.info(f"{r}: {c}")

if __name__ == "__main__":
    run_smc_test()
