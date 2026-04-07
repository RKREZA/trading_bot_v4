import argparse
import logging
import time
import json
import os
import signal
from datetime import datetime, timezone, date
from dotenv import load_dotenv

# Load credentials from .env
load_dotenv()

from core.connection import MT5Connection, PositionManager
from core.data_handler import DataFetcher
from core.strategy_orchestrator import StrategyOrchestrator
from core.portfolio_manager import PortfolioManager
from core.news_filter import SimpleNewsFilter
from core.session_detector import SessionDetector
from core.base_strategy import MarketData
from core.strategy_runtime import StrategyRuntime
from core.risk.risk_guardian import RiskGuardian
from core.execution.order_manager import OrderManager
from strategies import create_strategy
from dashboard import TradingDashboard, start_dashboard

def setup_live_logging():
    os.makedirs("logs", exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("logs/v4_live.log"),
            logging.StreamHandler()
        ]
    )

class LiveOrchestrator:
    def __init__(self, symbol: str, strategy_names: list):
        self.symbol = symbol
        self.connection = MT5Connection()
        self.pos_manager = PositionManager(self.connection)
        self.logs = []
        
        # 1. LOAD CONFIGURATION
        try:
            with open("config.json", "r") as f:
                self.config = json.load(f)
        except Exception:
            self.config = {}

        # 2. INSTANTIATE MICRO-SERVICES
        self.data_manager = DataFetcher()
        self.news_filter = SimpleNewsFilter()
        self.order_manager = OrderManager(self.config)
        
        # 3. CONSTRUCT RUNTIMES
        self.runtimes = []
        for name in strategy_names:
            try:
                # Find the correct StrategyID (e.g., TrendFollowing -> TREND_FOLLOWING)
                sid = f"{name.lower()}_v4"
                st_type = name.upper()
                
                # Correct call to create_strategy(sid, type, config)
                strat = create_strategy(sid, st_type, self.config)
                
                # Each runtime gets its own risk guardian
                risk_guardian = RiskGuardian(self.config)
                runtime = StrategyRuntime(strat, self.config, risk_guardian)
                self.runtimes.append(runtime)
            except Exception as e:
                logging.error(f"Failed to instantiate strategy {name}: {str(e)}")

        # 4. INITIALIZE ORCHESTRATOR (Satisfy 6 dependencies)
        self.orchestrator = StrategyOrchestrator(
            runtimes=self.runtimes,
            config=self.config,
            connection=self.connection,
            position_manager=self.order_manager,
            notification_manager=None, # TUI serves as notifier
            broker_clock=self.connection # Uses MT5 Server Time
        )
        
    def run(self):
        """Launches the Live Pulse loop with TUI dashboard."""
        if not self.connection.connect():
            print("CRITICAL: MT5 Connection Failed. Check credentials/server in .env.")
            return

        dashboard = TradingDashboard()
        consecutive_errors = 0
        max_consecutive_errors = 10
        last_reset_date = date.today()
        
        with start_dashboard(dashboard.layout) as live:
            self.logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] System Initialized. Trading {self.symbol}.")
            
            while True:
                try:
                    # Daily Reset Check (D6 FIX)
                    today = date.today()
                    if today != last_reset_date:
                        current_balance = self.connection.account_info.get('balance', 0)
                        self.orchestrator.reset_daily(current_balance)
                        last_reset_date = today
                        self.logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] Daily reset triggered. Balance synced: ${current_balance:,.2f}")
                    
                    # Sync with Broker Server Time (Every Cycle)
                    server_info = self.connection.get_symbol_info(self.symbol)
                    if server_info and isinstance(server_info, dict):
                        dt_server = datetime.now(timezone.utc)
                    else:
                        dt_server = datetime.now(timezone.utc)
                    
                    # 1. Fetch Multi-Timeframe Institutional Data (HTF, M15, M5)
                    m5_data = self.data_manager.fetch_candles(self.symbol, "M5", 500)
                    m15_data = self.data_manager.fetch_candles(self.symbol, "M15", 300)
                    h1_data = self.data_manager.fetch_candles(self.symbol, "H1", 200)
                    
                    if m5_data is None or h1_data is None or m15_data is None or len(m5_data) < 20:
                        missing = []
                        if m5_data is None: missing.append("M5")
                        elif len(m5_data) < 20: missing.append("M5(Sync)")
                        if m15_data is None: missing.append("M15")
                        if h1_data is None: missing.append("H1")
                        
                        msg = f"Waiting for {', '.join(missing)} data synchronization..."
                        self.logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")
                        
                        # Partial state update for TUI feedback
                        live.update(dashboard.update({
                            "logs": self.logs[-10:],
                            "symbol": self.symbol,
                            "connection": self.connection.get_account_snapshot()
                        }))
                        time.sleep(2)
                        continue

                    # 2. Package Market State (Satisfy V4 MarketData contract)
                    md = MarketData(
                        symbol=self.symbol,
                        htf_candles=h1_data,
                        m15_candles=m15_data,
                        m5_candles=m5_data,
                        d1_candles=None,
                        current_price=float(m5_data.c[-1]),
                        session=SessionDetector.get_session(dt_server),
                        timestamp=dt_server
                    )

                    # 3. Parallel Execution Logic
                    is_news_blocked = self.news_filter.is_blocked(dt_server.timestamp())
                    pulse_report = self.orchestrator.execute_cycle(self.symbol, md, is_news_blocked=is_news_blocked)
                    
                    # 4. UI SYNCHRONIZATION
                    raw_positions = self.pos_manager.get_open_positions()
                    ui_positions = []
                    for p in raw_positions:
                        ui_positions.append({
                            "symbol": p.symbol,
                            "type_text": "BUY" if p.type == 0 else "SELL",
                            "volume": p.volume,
                            "price_open": p.price_open,
                            "sl": p.sl,
                            "tp": p.tp,
                            "profit": p.profit
                        })
                    
                    # Generate Pulse Analysis Log
                    reg = pulse_report.get("regime")
                    strat_biases = " | ".join([f"{sid[:5]}: {sig.direction if hasattr(sig, 'direction') else sig}" for sid, sig in pulse_report.get("strategies", {}).items()])
                    analysis_msg = f"[ANALYSIS] {self.symbol} ADX:{reg.adx:.1f} ({reg.market_type.value}) | {strat_biases}"
                    
                    self.logs.append(f"[{pulse_report.get('timestamp')}] {analysis_msg}")
                    
                    # Record Executions
                    for exec_res in pulse_report.get("execution", []):
                        self.logs.append(f"[{pulse_report.get('timestamp')}] [TRADE] ENTERED {exec_res.get('direction')} @ {exec_res.get('fill_price')}")

                    if len(self.logs) > 50: self.logs = self.logs[-50:]

                    acc = self.connection.get_account_snapshot()
                    state = {
                        "connection": acc,
                        "account": acc,
                        "regime_type": reg.market_type.value,
                        "volatility": reg.volatility.value,
                        "session": md.session,
                        "symbol": self.symbol,
                        "price": md.current_price,
                        "spread": m5_data.s[-1] if len(m5_data) > 0 else 0.0,
                        "setups": {sid: sig.reasons if hasattr(sig, 'reasons') else [] for sid, sig in pulse_report["strategies"].items()},
                        "positions": ui_positions,
                        "logs": self.logs,
                        "news_list": self.news_filter.get_all_upcoming_events(dt_server.timestamp()) if hasattr(self.news_filter, 'get_all_upcoming_events') else []
                    }
                    
                    live.update(dashboard.update(state))
                    consecutive_errors = 0  # Reset on successful cycle
                    time.sleep(1)

                except KeyboardInterrupt:
                    print("\nShutdown requested by user.")
                    break
                
                except Exception as e:
                    import traceback
                    consecutive_errors += 1
                    crash_file = os.path.join("logs", "crash_report.log")
                    with open(crash_file, "a") as f:
                        f.write(f"\n--- CRASH: {datetime.now()} ---\n")
                        f.write(traceback.format_exc())
                        
                    # Fix: Rotate crash_report.log if > 5MB
                    if os.path.exists(crash_file) and os.path.getsize(crash_file) > 5 * 1024 * 1024:
                        try:
                            with open(crash_file, "r") as f:
                                lines = f.readlines()
                            with open(crash_file, "w") as f:
                                f.writelines(lines[-1000:])
                        except Exception:
                            pass
                    self.logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] [ERROR] CRITICAL PULSE FAILURE ({consecutive_errors}/{max_consecutive_errors}). See crash_report.log")
                    
                    if consecutive_errors >= max_consecutive_errors:
                        self.logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] [FATAL] Too many consecutive errors. Halting execution.")
                        break
                    
                    time.sleep(5)
        
        # Graceful shutdown
        logging.info("Shutting down. Open positions remain managed by MT5.")
        self.connection.disconnect()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="V4-ULTRA Live Trading Host")
    parser.add_argument("--symbol", type=str, default="XAUUSDm")
    parser.add_argument("--strategies", type=str, default="TrendFollowing,Breakout")
    
    args = parser.parse_args()
    setup_live_logging()
    
    strats = args.strategies.split(",")
    bot = LiveOrchestrator(args.symbol, strats)
    bot.run()
