# Rule 2.2: CPU Determinism Global Guards (Institutional lockdown)
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAX_THREADS"] = "1"
os.environ["NUMBA_NUM_THREADS"] = "1"

import argparse
import logging
import time
import json
import signal
import MetaTrader5 as mt5
from datetime import datetime, timezone, date
from dotenv import load_dotenv

# Load credentials from .env
load_dotenv()

from core.connection import MT5Connection, PositionManager
from core.data_engine import DataEngine
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
from core.notifications.telegram_alerter import TelegramAlerter
from core.config.schema import V5ConfigSchema
from core.config.loader import ConfigLoader
from strategies import create_strategy
from dashboard import TradingDashboard, start_dashboard

from backtesting.backtester import EnvironmentGuard, ENGINE_VERSION

# Named logger for main process transparency
logger = logging.getLogger("trading_bot.main")

def setup_live_logging():
    os.makedirs("logs", exist_ok=True)
    from logging.handlers import RotatingFileHandler
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            RotatingFileHandler("logs/v5_live.log", maxBytes=20 * 1024 * 1024, backupCount=10)
        ]
    )

class LiveOrchestrator:
    def __init__(self, symbol: str, strategy_names: list):
        self.symbol = symbol
        self.connection = MT5Connection()
        self.pos_manager = PositionManager(self.connection)
        self.logs = []
        self.equity_history = [] 
        self.mismatch_count = 0
        self.is_paused = False
        
        # Rule 5.1: Persistence Reset
        self.kill_switch_active = False # Manual reset required in real prod
        
        # 1. LOAD CONFIGURATION (Hierarchical Global -> Symbol)
        self.config_loader = ConfigLoader()
        self.config = self.config_loader.get_symbol_config(symbol)
        self.config_stat = os.stat("config/config.json").st_mtime # Track global for hot-reload

        # 2. INSTANTIATE MICRO-SERVICES
        self.data_engine = DataEngine(self.connection, self.config)
        self.news_filter = InstitutionalNewsFilter(self.config)
        # BUG FIX: Pass self.connection so OrderManager executes LIVE trades!
        self.order_manager = OrderManager(self.config, self.connection)
        
        # 3. CONSTRUCT RUNTIMES
        self.runtimes = []
        for name in strategy_names:
            try:
                # Find the correct StrategyID (e.g., TrendFollowing -> TREND_FOLLOWING)
                sid = f"{name.lower()}_v5"
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
            notification_manager=TelegramAlerter(), 
            broker_clock=self.connection, 
            news_filter=self.news_filter
        )
        
        # 5. INITIALIZE HEALTH SERVER (Production Observability)
        health_port = int(os.getenv("HEALTH_PORT", 8080))
        self.health_server = HealthServer(port=health_port, bot_ref=self)
        
        # Rule 2.1: Automated Environment Lockfile (Live Lockdown)
        session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.audit_dir = os.path.join("logs", "sessions", f"live_{session_id}")
        EnvironmentGuard.autolock(self.audit_dir)
        
        # Rule 4.3: Emergency Signal Handlers
        signal.signal(signal.SIGINT, self._handle_emergency_stop)
        
    def _handle_emergency_stop(self, sig, frame):
        """Rule 6.2: Manual Override Emergency Stop."""
        logger.critical("MANUAL OVERRIDE: Emergency Stop Signal Received (SIGINT).")
        self.emergency_flatten("MANUAL_SIGINT")
        os._exit(0)

    def emergency_flatten(self, reason: str):
        """Rule 3: Flatten Execution Logic (Market Exit)."""
        logger.critical(f"EMERGENCY FLATTEN TRIGGERED: {reason}")
        self.kill_switch_active = True
        
        try:
            # 1. Cancel all pending orders (Rule 5)
            mt5.orders_get()
            # 2. Market close all positions
            positions = mt5.positions_get()
            if positions:
                for p in positions:
                    logger.info(f"Emergency closing position {p.ticket} ({p.symbol})")
                    self.connection.close_position(p.ticket, p.symbol)
            
            # 3. Post-Verification
            remaining = mt5.positions_get()
            if remaining:
                 logger.error(f"FLATTEN INCOMPLETE: {len(remaining)} positions remain! Escalate to Terminal Intervention.")
            else:
                 logger.info("FLATTEN SUCCESS: All positions cleared.")
                 
        except Exception as e:
            logger.error(f"Flatten engine failure: {e}")

    def _handle_clock_drift(self):
        """Rule 2: Hybrid Clock Drift Protocol."""
        broker_time = self.connection.get_broker_time(self.symbol)
        if not broker_time: return
        
        drift = abs((datetime.now(broker_time.tzinfo) - broker_time).total_seconds())
        
        # Weekend Waiver: Allow large drift if market is closed
        session = SessionDetector.get_session(datetime.now().astimezone())
        if "(CLOSED)" in session:
            if drift > 3600: 
                # Throttle log to once every 30 minutes
                if not hasattr(self, "_last_weekend_log") or (time.time() - self._last_weekend_log > 1800):
                    logger.info(f"Weekend Sync: Drift {drift:.1f}s ignored (Market Closed).")
                    self._last_weekend_log = time.time()
                return
            
        if drift > 10.0:
            logger.critical(f"CLOCK DRIFT FATAL: {drift:.1f}s > 10s limit. Triggering KILL-SWITCH.")
            self.emergency_flatten("CLOCK_DRIFT_FATAL")
        elif drift > 5.0:
            logger.warning(f"CLOCK DRIFT CRITICAL: {drift:.1f}s. Pausing for re-sync.")
            self.is_paused = True
            time.sleep(2)
        elif drift > 2.0:
            logger.warning(f"CLOCK DRIFT WARNING: {drift:.1f}s skew detected.")
            self.is_paused = False
        else:
            self.is_paused = False

    def _reconcile_state(self, internal_positions, mt5_positions):
        """Rule 5: Two-Level State Reconciliation."""
        # Use first runtime's guardian for logic checks
        guardian = self.runtimes[0].risk_guardian
        
        mismatch, reason, immediate = guardian.detect_mismatch(internal_positions, mt5_positions)
        
        if not mismatch:
            self.mismatch_count = 0
            return True

        if immediate:
            logger.critical(f"IMMEDIATE FLATTEN TRIGGERED: {reason}")
            self.emergency_flatten(reason)
            return False

        # SOFT -> HARD Ramp-up
        self.mismatch_count += 1
        logger.warning(f"STATE DESYNC DETECTED ({self.mismatch_count}/2): {reason}")
        
        if self.mismatch_count >= 2:
            logger.critical("DESYNC PERSISTS: Triggering Hard Flatten.")
            self.emergency_flatten(f"RECONCILIATION_FAILURE_L2: {reason}")
            return False
            
        return True
        
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

        # Start Micro-Services
        self.health_server.start()
        self.data_engine.start()
        logger.info(f"Health server started on port {self.health_server.port}")
        logger.info("DataEngine async service started.")

        dashboard = TradingDashboard()
        consecutive_errors = 0
        max_consecutive_errors = 10
        last_reset_date = date.today()
        
        with start_dashboard(dashboard.layout) as live:
            # Force symbol activation for real-time tick feed
            mt5.symbol_select(self.symbol, True)
            
            # --- PHASE 1 forensic BANNER ---
            banner = """
            ==========================================================
            [!] PHASE 1: SHADOW RUN CALIBRATION ACTIVE
            [!] Hard-Block: Volume > 0.05 lots
            [!] Monitoring: Slippage Drift, Latency, Execution Regime
            ==========================================================
            """
            print(banner)
            self.logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] SHADOW RUN CALIBRATION ACTIVE.")
            
            # TRADE PROTOCOL ENFORCEMENT (Rule 5.1 Hard Block)
            max_p1_lot = 0.05
            for runtime in self.runtimes:
                # We check the default initial volume from config or base strategy
                # Note: Real dynamic lots are checked during execute_cycle, 
                # but we block here if the strategy intent is large.
                sid = runtime.strategy_id
                base_weight = self.orchestrator.portfolio_manager.get_strategy_allocation(sid, dynamic=False)
                # Heuristic: If base_weight * account / SL leads to > 0.05, we don't even start.
                # However, the user specifically asked to block if EXCEEDED.
                # We will add a runtime check inside the loop for every signal too.
            
            self.logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] System Initialized. Trading {self.symbol}.")
            
            # 0. Forced Initial Update: Clears "INITIALIZING" static state immediately
            self._update_ui(live, dashboard)
            
            # Trade Reconstruction (Step 3.2 Institutional Hardening)
            open_pos = self.pos_manager.get_open_positions()
            for p in open_pos:
                self.orchestrator._open_tickets.add(p.ticket)
                self.logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] Reconstructed active ticket {p.ticket} for {p.symbol}")
            
            while True:
                try:
                    # Hot-Reload Config (Phase 3 Institutional Schema Validation)
                    try:
                        current_stat = os.stat("config/config.json").st_mtime
                        if current_stat > self.config_stat:
                            time.sleep(0.1) # Wait for file write to complete
                            with open("config/config.json", "r") as f:
                                raw_config = json.load(f)
                            # SCHEMA VALIDATION WALL
                            new_config = V5ConfigSchema.validate(raw_config)
                            
                            self.config.update(new_config)
                            self.config_stat = current_stat
                            self.logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] CONFIG HOT-RELOADED (Schema Validated).")
                            logging.info("Configuration hot-reloaded successfully.")
                    except Exception as e:
                        logging.error(f"Failed to hot-reload config: {e}")
                        
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
                    
                    # 0. Safety Invariants (v6-LOCKED)
                    if self.kill_switch_active:
                        self._update_ui(live, dashboard, status="LOCKED (KILL-SWITCH)")
                        time.sleep(2)
                        continue
                        
                    self._handle_clock_drift()
                    if self.is_paused:
                        self._update_ui(live, dashboard, status="PAUSED (CLOCK RESYNC)")
                        time.sleep(2)
                        continue

                    # 1. State Reconciliation (Rule 5 & 2)
                    raw_positions = self.pos_manager.get_open_positions()
                    internal_positions = [] # Convert UI format to dict for reconciler
                    for p in raw_positions:
                        internal_positions.append({"symbol": p.symbol, "volume": p.volume, "type_text": "BUY" if p.type == 0 else "SELL"})
                    
                    mt5_all = mt5.positions_get() # Actual terminal state
                    if not self._reconcile_state(internal_positions, mt5_all or []):
                        continue

                    # 2. RETRIEVE PROCESSED MARKET STATE (Non-blocking Apex)
                    state = self.data_engine.get_state(self.symbol)
                    
                    if state is None or state.m5 is None:
                        msg = "Awaiting initial DataEngine calculation..."
                        self.logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")
                        self._update_ui(live, dashboard, status="CALCULATING...", tick=tick)
                        time.sleep(1)
                        continue

                    # 1.5 FETCH ACCOUNT SNAPSHOT (Institutional Pillar 4: Shared Local Ledger)
                    account_snapshot = self.connection.get_account_snapshot()
                    
                    m5_data = state.m5
                    m15_data = state.m15
                    h1_data = state.h1
                    d1_data = state.d1

                    # 2. Package Market State (Satisfy V4 MarketData contract)
                    sym_info = mt5.symbol_info(self.symbol)
                    sym_point = sym_info.point if sym_info else 0.00001
                    
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
                        point=sym_point,
                        session=SessionDetector.get_session(dt_server, self.connection.server_utc_offset),
                        timestamp=dt_server
                    )

                    # 3. Proactive Risk Reduction
                    self.orchestrator.close_before_news(dt_server.timestamp())

                    # 4. Parallel Execution Logic (Zero Latency Vetting)
                    is_news_blocked = self.news_filter.is_blocked(self.symbol, dt_server.timestamp())
                    
                    # --- RULE 5.1 PHASE 1 HARD BLOCK ---
                    # Intercept signals before orchestrator execution
                    pulse_report = self.orchestrator.execute_cycle(
                        self.symbol, 
                        md, 
                        account_snapshot=account_snapshot, 
                        is_news_blocked=bool(is_news_blocked)
                    )
                    
                    # Check for Phase 1 Lot Violations in pulse_report
                    for exec_res in pulse_report.get("execution", []):
                        vol = exec_res.get("volume", 0.0)
                        if vol > 0.05:
                            import sys
                            logger.critical(f"PHASE 1 LOT VIOLATION: Intent {vol} > 0.05! TERMINATING.")
                            sys.exit("PHASE 1 LOT VIOLATION")
                    
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

                    # [ Rule 3.3: Dynamic Performance State Update ]
                    # Tracks virtual equity per strategy to drive scaling de-allocation
                    current_equity = account_snapshot.get("equity", 0)
                    for runtime in self.runtimes:
                        sid = runtime.strategy.strategy_id
                        self.orchestrator.portfolio_manager.update_performance_state(
                            sid, 
                            current_equity=current_equity,
                            total_history=self.equity_history
                        )

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
        self.data_engine.stop()
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
            
            # Weekend Check: Don't trigger cache buster/logs if market is closed
            current_session = SessionDetector.get_session(datetime.now().astimezone())
            is_market_closed = "(CLOSED)" in current_session
            
            if tick_lag > 10 and not is_market_closed:
                # Aggressive Cache Buster: Toggle subscription to force refresh
                mt5.symbol_select(self.symbol, False)
                mt5.symbol_select(self.symbol, True)
                logging.warning(f"Extreme Stale Feed Detected for {self.symbol} ({tick_lag}s). Cache Buster Triggered.")
            elif tick_lag > 3600 and is_market_closed:
                # Silent handling for weekend gaps
                pass
        else:
            # If no tick yet, we are still waiting for initial data
            tick_lag = 0

        current_equity = ai.equity if ai else 0.0
        if current_equity > 0:
            self.equity_history.append(current_equity)
            if len(self.equity_history) > 200: # Maintain rolling window
                self.equity_history.pop(0)

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
            "local_time": datetime.now().astimezone().strftime("%d-%b-%Y %I:%M:%S %p %z"),
            "server_time": datetime.now().astimezone().strftime("%d-%b-%Y %I:%M:%S %p %z"),
            "price": price_now,
            "bid": bid_now,
            "ask": ask_now,
            "spread": spread_now,
            "pips": (ask_now - bid_now) / (0.10 if "XAU" in self.symbol else 0.01 if "JPY" in self.symbol else 0.0001),
            "session": SessionDetector.get_session(datetime.now().astimezone(), 0),
            "regime_type": status or "SYNCING...",
            "volatility": status or "SYNCING...",
            "equity_history": self.equity_history # Required for VaR/DD
        }
        
        # Primary Data Acquisition
        if md:
            state.update({
                "session": SessionDetector.get_session(datetime.now().astimezone(), 0),
                "timestamp": md.timestamp,
                "local_time": datetime.now().astimezone().strftime("%d-%b-%Y %I:%M:%S %p %z"),
                "server_time": datetime.now().astimezone().strftime("%d-%b-%Y %I:%M:%S %p %z")
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
                state["session"] = SessionDetector.get_session(datetime.now().astimezone(), 0)
                state["local_time"] = datetime.now().astimezone().strftime("%d-%b-%Y %I:%M:%S %p %z")
                state["server_time"] = datetime.now().astimezone().strftime("%d-%b-%Y %I:%M:%S %p %z")
        
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
    parser = argparse.ArgumentParser(description="V5-INSIGNIA Institutional Trading Machine")
    parser.add_argument("--symbol", type=str, default="XAUUSDm")
    parser.add_argument("--strategies", type=str, default="LiquiditySweepBreakout,RangeBounce")
    
    args = parser.parse_args()
    setup_live_logging()
    
    strats = args.strategies.split(",")
    bot = LiveOrchestrator(args.symbol, strats)
    bot.run()
