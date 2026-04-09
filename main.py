import argparse
import logging
import time
import json
import os
import signal
import MetaTrader5 as mt5
from datetime import datetime, timezone, date
from dotenv import load_dotenv

# Load credentials from .env
load_dotenv()

from core.connection import MT5Connection, PositionManager
from core.data_handler import DataFetcher
from core.strategy_orchestrator import StrategyOrchestrator
from core.portfolio_manager import PortfolioManager
from core.news_filter import InstitutionalNewsFilter
from core.session_detector import SessionDetector
from core.base_strategy import MarketData
from core.common.types import CandleArray
from core.strategy_runtime import StrategyRuntime
from core.risk.risk_guardian import RiskGuardian
from core.execution.order_manager import OrderManager
from core.health_server import HealthServer
from strategies import create_strategy
from dashboard import TradingDashboard, start_dashboard

# Named logger for main process transparency
logger = logging.getLogger("trading_bot.main")

def setup_live_logging():
    os.makedirs("logs", exist_ok=True)
    from logging.handlers import RotatingFileHandler
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            RotatingFileHandler("logs/v4_live.log", maxBytes=20 * 1024 * 1024, backupCount=10)
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
        self.news_filter = InstitutionalNewsFilter(self.config)
        # BUG FIX: Pass self.connection so OrderManager executes LIVE trades!
        self.order_manager = OrderManager(self.config, self.connection)
        
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

        # 4. INITIALIZE ORCHESTRATOR (Satisfy dependencies)
        self.orchestrator = StrategyOrchestrator(
            runtimes=self.runtimes,
            config=self.config,
            order_manager=self.order_manager,
            position_manager=self.pos_manager,
            notification_manager=None, 
            broker_clock=self.connection, 
            news_filter=self.news_filter
        )
        
        # 5. INITIALIZE HEALTH SERVER (Production Observability)
        health_port = int(os.getenv("HEALTH_PORT", 8080))
        self.health_server = HealthServer(port=health_port, bot_ref=self)
        
    def _preprocess_indicators(self, candles: CandleArray):
        """Institutional Preprocessing Layer: Ensures all analysts have warm indicator caches."""
        if candles is None or len(candles) < 30:
            return
            
        # These calls trigger the cached calculation in CandleArray
        candles.adx(14)
        candles.atr(14)
        candles.ema(50)
        candles.ema(200)

    def run(self):
        # Singleton Guard & Terminal Verification
        lock_file = "bot.lock"
        try:
            # Singleton Guard: Attempt to create an exclusive lock file
            if os.path.exists(lock_file):
                try:
                    os.rename(lock_file, lock_file)
                except OSError:
                    print("CRITICAL: Another instance of the bot is already running.")
                    return
            
            with open(lock_file, "w") as f:
                f.write(str(os.getpid()))
            
            # Log Terminal Path for multi-install diagnostics
            ti = mt5.terminal_info()
            if ti:
                logging.info(f"MT5 Terminal: {ti.name} | Path: {ti.path} | Acc: {ti.community_id}")
            
        except Exception as e:
            print(f"CRITICAL: Identity Check failure: {e}")
            return

        if not self.connection.connect():
            print("CRITICAL: MT5 Connection Failed. Check credentials/server in .env.")
            return

        # Start Health Server
        self.health_server.start()
        logger.info(f"Health server started on port {health_port}")

        dashboard = TradingDashboard()
        consecutive_errors = 0
        max_consecutive_errors = 10
        last_reset_date = date.today()
        
        with start_dashboard(dashboard.layout) as live:
            # Force symbol activation for real-time tick feed
            mt5.symbol_select(self.symbol, True)
            
            self.logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] System Initialized. Trading {self.symbol}.")
            
            # 0. Forced Initial Update: Clears "INITIALIZING" static state immediately
            self._update_ui(live, dashboard)
            
            while True:
                try:
                    # Capture Absolute Latest Tick for UI (Tick Fix)
                    tick = mt5.symbol_info_tick(self.symbol)
                    
                    # Daily Reset Check (D6 FIX)
                    today = date.today()
                    if today != last_reset_date:
                        current_balance = self.connection.account_info.get('balance', 0)
                        self.orchestrator.reset_daily(current_balance)
                        last_reset_date = today
                        self.logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] Daily reset triggered. Balance synced: ${current_balance:,.2f}")
                    
                    # Sync with Broker Server Time (Every Cycle) - FIX (Sync Pillar)
                    dt_server = self.connection.get_broker_time(self.symbol)
                    
                    if dt_server is None:
                        # Clock sync failed, wait and retry to avoid NoneType crashes
                        msg = "Waiting for Broker Clock synchronization..."
                        self.logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")
                        logging.info(f"Handshake: {msg}")
                        self._update_ui(live, dashboard)
                        time.sleep(2)
                        continue
                    
                    # 1. FETCH MULTI-TIMEFRAME DATA (HTF, M15, M5, D1)
                    m5_data = self.data_manager.fetch_candles(self.symbol, "M5", 500)
                    m15_data = self.data_manager.fetch_candles(self.symbol, "M15", 300)
                    h1_data = self.data_manager.fetch_candles(self.symbol, "H1", 300)
                    d1_data = self.data_manager.fetch_candles(self.symbol, "D1", 100)

                    # Diagnostic Gating
                    missing = []
                    if m5_data is None or len(m5_data) < 20: missing.append(f"M5({len(m5_data) if m5_data else 0}/20)")
                    if m15_data is None or len(m15_data) < 10: missing.append(f"M15({len(m15_data) if m15_data else 0}/10)")
                    if h1_data is None or len(h1_data) < 10: missing.append(f"H1({len(h1_data) if h1_data else 0}/10)")
                    if d1_data is None or len(d1_data) < 1: missing.append(f"D1({len(d1_data) if d1_data else 0}/1)")

                    if missing:
                        msg = f"Waiting for {', '.join(missing)} synchronization..."
                        self.logs.append(f"[{dt_server.strftime('%H:%M:%S')}] {msg}")
                        logger.info(f"Handshake: {msg}")
                        # Ensure logs are flushed to disk for immediate diagnostic visibility
                        for handler in logging.getLogger().handlers: handler.flush()
                        
                        self._update_ui(live, dashboard, status=f"SYNCING ({missing[0].split('(')[0]})...", tick=tick)
                        time.sleep(2) # Reduced sync pulse frequency to save API overhead
                        continue

                    # 1.5 Data Integrity & Synchronization Guard (Sync Pillar)
                    # Detect "Ghost Candles" (intra-day gaps) before generating signals
                    m5_report = self.data_manager.validate_data_integrity(m5_data, "M5")
                    if m5_report["status"] == "CRITICAL":
                        msg = "DATA CRITICAL: Ghost candles found in M5 history. Halting for backfill..."
                        self.logs.append(f"[{dt_server.strftime('%H:%M:%S')}] {msg}")
                        logger.error(f"Handshake: {msg}")
                        time.sleep(5)
                        continue
                    
                    # Ensure HTF data is not older than M5 freshness (Temporal Drift protection)
                    # If H1 data in cache is > 5 minutes old relative to current M5 tip, force refresh.
                    if len(m5_data) > 0 and len(h1_data) > 0:
                        m5_tip = m5_data.time[-1]
                        h1_tip = h1_data.time[-1]
                        if (m5_tip - h1_tip) > 3600 + 300: # 1 hour + 5 min tolerance
                            self.logs.append(f"[{dt_server.strftime('%H:%M:%S')}] SYNC GUARD: HTF data skew detected. Refreshing H1...")
                            h1_data = self.data_manager.fetch_candles(self.symbol, "H1", 300, force_refresh=True)

                    # 1.6 Preprocess Indicators (Fix ADX NaN)
                    for tf_data in [h1_data, m15_data, m5_data]:
                        self._preprocess_indicators(tf_data)

                    # 1.7 Fetch Real-time Tick Data
                    tick = mt5.symbol_info_tick(self.symbol)
                    if tick is None:
                        logger.error(f"Failed to get tick for {self.symbol}. Skipping cycle.")
                        time.sleep(1)
                        continue

                    # 2. Package Market State (Satisfy V4 MarketData contract)
                    md = MarketData(
                        symbol=self.symbol,
                        htf_candles=h1_data,
                        m15_candles=m15_data,
                        m5_candles=m5_data,
                        d1_candles=d1_data,
                        current_price=float(tick.bid),
                        bid=float(tick.bid),
                        ask=float(tick.ask),
                        spread=float(tick.ask - tick.bid),
                        session=SessionDetector.get_session(dt_server, self.config.get("backtest", {}).get("utc_offset", 0)),
                        timestamp=dt_server
                    )

                    # 3. Proactive Risk Reduction (Step 19)
                    self.orchestrator.close_before_news(dt_server.timestamp())

                    # 4. Parallel Execution Logic
                    is_news_blocked = self.news_filter.is_blocked(self.symbol, dt_server.timestamp())
                    pulse_report = self.orchestrator.execute_cycle(self.symbol, md, is_news_blocked=bool(is_news_blocked))
                    
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
                    strat_biases = " | ".join([f"{sid[:5]}: {s_info['signal'].direction if hasattr(s_info['signal'], 'direction') else s_info['signal']}" for sid, s_info in pulse_report.get("strategies", {}).items()])
                    analysis_msg = f"[ANALYSIS] {self.symbol} ADX:{reg.adx:.1f} ({reg.market_type.value}) | {strat_biases}"
                    
                    self.logs.append(f"[{pulse_report.get('timestamp')}] {analysis_msg}")
                    
                    # Record Executions
                    for exec_res in pulse_report.get("execution", []):
                        self.logs.append(f"[{pulse_report.get('timestamp')}] [TRADE] ENTERED {exec_res.get('direction')} @ {exec_res.get('fill_price')}")
                        self.health_server.record_trade(exec_res.get('direction', 'UNKNOWN'))
                    
                    # Record Metrics
                    self.health_server.record_cycle()
                    self.health_server.record_signal()

                    if len(self.logs) > 50: self.logs = self.logs[-50:]

                    # Final Cycle Update (Pass Live Tick)
                    self._update_ui(live, dashboard, md=md, reg=reg, pulse_report=pulse_report, tick=tick)
                    consecutive_errors = 0  # Reset on successful cycle
                    time.sleep(1)

                except KeyboardInterrupt:
                    print("\nShutdown requested by user.")
                    break
                
                except Exception as e:
                    import traceback
                    consecutive_errors += 1
                    err_msg = str(e)
                    self.health_server.record_error()  # Track errors in metrics
                    
                    # Log traceback DIRECTLY to the live log file for immediate awareness
                    logging.error(f"FATAL CYCLE ERROR: {err_msg}")
                    logging.error(traceback.format_exc())
                    
                    # Fatal Crash Prevention: Protect the logging logic itself
                    try:
                        crash_file = os.path.join("logs", "crash_report.log")
                        with open(crash_file, "a") as f:
                            f.write(f"\n--- CRASH: {datetime.now()} ---\n")
                            f.write(traceback.format_exc())
                    except Exception as log_err:
                        print(f"FAILED TO WRITE CRASH LOG: {log_err}")

                    self.logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] [ERROR] {err_msg[:50]} ({consecutive_errors}/{max_consecutive_errors})")
                    print(f"ERROR: {err_msg}") # Direct terminal feedback
                    
                    if consecutive_errors >= max_consecutive_errors:
                        self.logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] [FATAL] Critical failure threshold reached.")
                        break
                    
                    time.sleep(5)
        
        # Graceful shutdown
        logging.info("Shutting down. Open positions remain managed by MT5.")
        self.health_server.stop()  # Stop health server
        if os.path.exists("bot.lock"):
            try:
                os.remove("bot.lock")
            except:
                pass
        self.connection.disconnect()

    def _update_ui(self, live, dashboard, md=None, reg=None, pulse_report=None, status=None, tick=None):
        """Unified UI state pulser with absolute tick priority."""
        acc = self.connection.get_account_snapshot()
        if not tick:
            tick = mt5.symbol_info_tick(self.symbol)
        
        # Institutional Performance: Cache static metadata to reduce API overhead (Audit Bug #8)
        if not hasattr(self, "_cached_si") or self._cached_si is None:
            self._cached_si = mt5.symbol_info(self.symbol)
        if not hasattr(self, "_cached_ai") or self._cached_ai is None:
            self._cached_ai = mt5.account_info()
        if not hasattr(self, "_cached_ti") or self._cached_ti is None:
            self._cached_ti = mt5.terminal_info()
        
        si = self._cached_si
        ai = self._cached_ai
        ti = self._cached_ti
        
        price_now = tick.bid if tick else 0.0
        ask_now = tick.ask if tick else 0.0
        bid_now = tick.bid if tick else 0.0
        spread_now = (tick.ask - tick.bid) / si.point if (tick and si) else 0.0
        digits = si.digits if si else 2
        
        # Staleness Detection & Aggressive Cache Buster
        tick_lag = 0
        if tick:
            tick_lag = int(time.time() - tick.time)
            if tick_lag > 10:
                # Aggressive Cache Buster: Toggle subscription to force refresh
                mt5.symbol_select(self.symbol, False)
                mt5.symbol_select(self.symbol, True)
                logging.warning(f"Extreme Stale Feed Detected for {self.symbol} ({tick_lag}s). Cache Buster Triggered.")
        else:
            # If no tick yet, we are still waiting for initial data
            tick_lag = 0

        state = {
            "connection": acc,
            "account": acc,
            "login": acc.get("login", "N/A"),
            "account_name": ai.name if ai else "N/A",
            "terminal_path": ti.path if ti else "N/A",
            "server": acc.get("server", "N/A"),
            "symbol": self.symbol,
            "digits": digits,
            "tick_lag": tick_lag,
            "logs": self.logs[-15:],
            "server_time": datetime.now().strftime("%d-%b-%Y %I:%M:%S %p"), # Local Pulse for UX confirmation
            "price": price_now,
            "bid": bid_now,
            "ask": ask_now,
            "spread": spread_now,
            "pips": (ask_now - bid_now) / (0.10 if "XAU" in self.symbol else 0.01 if "JPY" in self.symbol else 0.0001),
            "session": "SYNCING...",
            "regime_type": status or "SYNCING...",
            "volatility": status or "SYNCING..."
        }
        
        # Primary Data Acquisition
        if md:
            state.update({
                "session": md.session,
                "timestamp": md.timestamp,
                "server_time": md.timestamp.strftime("%d-%b-%Y %I:%M:%S %p")
            })
        else:
            # Fallback for early cycles (Waiting for candles)
            snap = self.connection.get_symbol_snapshot(self.symbol)
            if snap.get("price", 0) > 0:
                state["price"] = snap["price"]
            if snap.get("spread", 0) > 0:
                state["spread"] = snap["spread"]
            # Try to detect session using broker clock if available
            dt_server = self.connection.get_broker_time(self.symbol)
            if dt_server:
                state["session"] = SessionDetector.get_session(dt_server, self.config.get("backtest", {}).get("utc_offset", 0))
                state["server_time"] = dt_server.strftime("%d-%b-%Y %I:%M:%S %p")
        
        if reg:
            state.update({
                "regime_type": reg.market_type.value,
                "volatility": reg.volatility.value
            })
            
        if pulse_report:
            ui_positions = []
            all_positions = self.pos_manager.get_open_positions()
            for pos in all_positions:
                ui_positions.append({
                    'symbol': pos.symbol,
                    'type_text': "BUY" if pos.type == 0 else "SELL",
                    'volume': pos.volume,
                    'profit': pos.profit
                })
            
            state.update({
                "setups": pulse_report.get("strategies", {}),
                "positions": ui_positions,
                "news_list": pulse_report.get("upcoming_news_obj", []),
                "news_stale": (time.time() - os.path.getmtime(self.news_filter.cache_file)) > 86400 if os.path.exists(self.news_filter.cache_file) else True
            })

        # Add system metrics to state
        state["metrics"] = self.health_server.get_metrics()
        
        live.update(dashboard.update(state), refresh=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="V4-ULTRA Live Trading Host")
    parser.add_argument("--symbol", type=str, default="XAUUSDm")
    parser.add_argument("--strategies", type=str, default="TrendFollowing,Breakout")
    
    args = parser.parse_args()
    setup_live_logging()
    
    strats = args.strategies.split(",")
    bot = LiveOrchestrator(args.symbol, strats)
    bot.run()
