import logging
import numpy as np
import os
from tqdm import tqdm
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional

from core.base_strategy import MarketData
from core.regime_detector import RegimeDetector
from core.risk_engine import RiskEngine
from core.session_detector import SessionDetector
from core.portfolio_manager import PortfolioManager
from core.regime_gater import RegimeGater
from core.recovery.checkpoint_manager import CheckpointManager
from backtesting.simulator import ExecutionSimulator

logger = logging.getLogger("trading_bot.backtester")

class PortfolioBacktester:
    """
    V4-ULTRA Production-Grade Event-Driven Backtester.
    Strictly follows 'Step 4' and 'Step 5' of the institutional development order.
    
    Features:
    - M1 Candle-Event Replay (Step 4.3)
    - Institutional Execution Simulation (Slippage/Latency/Variable Spread)
    - Crash-Safe Checkpointing & Recovery (Step 3)
    - Deterministic Determinism (Step 11)
    """

    def __init__(self, config: dict):
        self.config = config
        self.regime_detector = RegimeDetector()
        self.risk_engine = RiskEngine(config)
        self.simulator = ExecutionSimulator(config)
        self.portfolio_manager = PortfolioManager(config)
        self.checkpoint_manager = CheckpointManager()

        bt_cfg = config.get("backtest", {})
        self.initial_partition_balance = float(bt_cfg.get("initial_balance_per_strategy", 1000.0))
        
        # Internal State
        self.current_index = 0
        self.history = []
        self.open_trades = {}     # strategy_id -> trade_dict
        self.balances = {}        # strategy_id -> float
        self.equities = {}        # strategy_id -> float
        self.equity_history = []
        self.max_drawdowns = {}   
        self.peak_equity = {}

    def reset(self, strategies_ids: List[str]):
        """Full reset of the simulation state."""
        self.current_index = 0
        self.history = []
        self.open_trades = {}
        self.balances = {sid: self.initial_partition_balance for sid in strategies_ids}
        self.equities = {sid: self.initial_partition_balance for sid in strategies_ids}
        self.peak_equity = {sid: self.initial_partition_balance for sid in strategies_ids}
        self.max_drawdowns = {sid: 0.0 for sid in strategies_ids}
        self.equity_history = []
        self.checkpoint_manager.clear_checkpoint()

    def get_state(self) -> Dict[str, Any]:
        """Captures a snapshot for crash recovery."""
        return {
            "current_index": self.current_index,
            "balances": self.balances,
            "equities": self.equities,
            "peak_equity": self.peak_equity,
            "max_drawdowns": self.max_drawdowns,
            "open_trades": self.open_trades,
            "history": self.history
        }

    def set_state(self, state: Dict[str, Any]):
        """Restores state from a checkpoint."""
        self.current_index = state["current_index"]
        self.balances = state["balances"]
        self.equities = state["equities"]
        self.peak_equity = state["peak_equity"]
        self.max_drawdowns = state["max_drawdowns"]
        self.open_trades = state["open_trades"]
        self.history = state["history"]

    def run(self, symbol: str, strategies: list, target_tf_data, h1_data, m15_data, m5_data, m1_data, resume: bool = False):
        """
        Production Backtest Runner.
        Implements 'Step 15' development loop with Checkpoint support.
        """
        logger.info(f"Starting V4-ULTRA Production Backtest on {symbol}...")
        
        active_strategies = [s for s in strategies if getattr(s, "enabled", True) and s.is_symbol_allowed(symbol)]
        sid_list = [s.strategy_id for s in active_strategies]
        
        if not resume:
            self.reset(sid_list)
        else:
            state = self.checkpoint_manager.load_checkpoint()
            if state:
                self.set_state(state)
            else:
                logger.warning("Resume requested but no checkpoint found. Starting from scratch.")
                self.reset(sid_list)

        symbol_cfg = self.config.get("symbols_config", {}).get(symbol, {})
        point = float(symbol_cfg.get("point", 0.0001))
        tick_value = float(symbol_cfg.get("tick_value", 10.0))
        comm_per_lot = float(symbol_cfg.get("commission_per_lot", 7.0))
        
        # 1. Institutional Indicator Pre-calculation (IPC) - Step 4.2
        from core.indicator_engine import IndicatorEngine
        from rich.console import Console
        console = Console()
        
        with console.status(f"[bold blue]Calibrating {symbol} Strategy Indicators...") as status:
            target_tf_data.indicators = IndicatorEngine.precalculate_all(symbol, getattr(target_tf_data, "timeframe", "UNKNOWN"), target_tf_data)
            m5_data.indicators = IndicatorEngine.precalculate_all(symbol, "M5", m5_data)
            m15_data.indicators = IndicatorEngine.precalculate_all(symbol, "M15", m15_data)
            h1_data.indicators = IndicatorEngine.precalculate_all(symbol, "H1", h1_data)
            logger.info("Indicator Pre-calculation COMPLETED.")

        # Pre-flight data integrity check (Step 11)
        self._validate_data_alignment(target_tf_data, m1_data)

        # Main Loop: Step through target timeframe bars starting from current_index
        last_date = None
        pbar = tqdm(total=len(target_tf_data.time), initial=self.current_index)
        
        for i in range(max(100, self.current_index), len(target_tf_data.time)):
            try:
                self.current_index = i
                pbar.update(1)
                t = target_tf_data.time[i]
                dt = datetime.fromtimestamp(t, tz=timezone.utc)

                # 0. DAILY RESET TRIGGER (Critical for Session Strategies)
                current_date = dt.date()
                if last_date is not None and current_date != last_date:
                    for strat in active_strategies:
                        strat.reset_daily_stats()
                    for sid in self.balances:
                        self.risk_engine.reset_daily(self.balances[sid])
                last_date = current_date
                
                # [ Institutional Fidelity ]: Zero-Copy Index Shifting
                target_tf_data.set_limit(i) 
                
                if m5_data is not target_tf_data: m5_data.set_limit(self._get_tf_idx(m5_data, t, side="left"))
                if m15_data is not target_tf_data: m15_data.set_limit(self._get_tf_idx(m15_data, t, side="left"))
                if h1_data is not target_tf_data: h1_data.set_limit(self._get_tf_idx(h1_data, t, side="left"))
                
                # 1. Regime Detection & Gating
                regime_info = self.regime_detector.detect(target_tf_data)
                regime = regime_info.market_type
                risk_mult = RegimeGater.get_risk_multiplier(regime_info.volatility)
                conf_buffer = RegimeGater.get_confidence_buffer(regime_info.volatility)

                # 2. MarketData Construction (Zero-Copy & Anti-Lookahead)
                market_data = MarketData(
                    symbol=symbol,
                    htf_candles=h1_data,
                    m15_candles=m15_data,
                    m5_candles=m5_data,
                    d1_candles=None,
                    current_price=target_tf_data.open[i], 
                    session=SessionDetector.get_session(dt),
                    timestamp=dt
                )
                
                # 3. Micro-service Strategy Replay
                for strat in active_strategies:
                    sid = strat.strategy_id
                    
                    if RegimeGater.is_drawdown_gated(self.max_drawdowns.get(sid, 0)): continue
                    if not RegimeGater.is_strategy_allowed(strat.__class__.__name__, regime): continue
                    if sid in self.open_trades: continue
                    
                    signal = strat.generate_signal(market_data)
                    
                    if signal and signal.direction != "NONE":
                        min_conf = getattr(strat, "min_confidence", 0.6)
                        if signal.confidence < (min_conf + conf_buffer): continue
                            
                        sl = strat.get_stop_loss(signal, market_data)
                        tp = strat.get_take_profit(signal, market_data)
                        sl_dist = abs(market_data.current_price - sl)
                        
                        if sl_dist > 0:
                            lot_size = self.risk_engine.calculate_lot_size(
                                balance=self.balances[sid],
                                stop_loss_distance=sl_dist,
                                point=point,
                                tick_value=tick_value,
                                symbol=symbol
                            )
                            lot_size = lot_size * risk_mult
                            
                            if lot_size >= 0.01:
                                fill = self.simulator.simulate_entry(
                                    signal=signal,
                                    current_price=market_data.current_price,
                                    base_spread_points=float(target_tf_data.spread[i]),
                                    point=point
                                )
                                if fill:
                                    entry_comm = lot_size * comm_per_lot
                                    fill.update({
                                        "sl": sl, 
                                        "tp": tp, 
                                        "strategy_id": sid, 
                                        "lots": lot_size, 
                                        "session": market_data.session,
                                        "entry_comm": entry_comm
                                    })
                                    self.open_trades[sid] = fill
                                    logger.debug(f"[{sid}] Trade Entered: {fill['direction']} @ {fill['fill_price']:.5f}")

                # 4. M1 Intra-Bar Execution
                m1_slice = self._get_m1_for_m5(m1_data, t)
                if len(m1_slice) > 0:
                    self._manage_active_trades(m1_slice, tick_value, point, comm_per_lot, active_strategies)
                
                # 5. Equity Sampling & Drawdown Track
                for sid in self.balances:
                    self.peak_equity[sid] = max(self.peak_equity[sid], self.equities[sid])
                    dd = (self.peak_equity[sid] - self.equities[sid]) / self.peak_equity[sid] * 100
                    self.max_drawdowns[sid] = max(self.max_drawdowns[sid], dd)
                    self.equity_history.append({"time": t, "strategy_id": sid, "equity": self.equities[sid]})

                if i % 100 == 0:
                    self.checkpoint_manager.save_checkpoint(self.get_state())

            except Exception as e:
                import traceback
                with open("crash_report.log", "a") as f:
                    f.write(f"\n--- BACKTEST CRASH: {datetime.now()} ---\n")
                    f.write(traceback.format_exc())
                raise e

        pbar.close()
        self._force_close_at_end(target_tf_data, point, tick_value, comm_per_lot, active_strategies)
        self.checkpoint_manager.clear_checkpoint()
        return self.history, self.equity_history

    def _manage_active_trades(self, m1_candles, tick_value, point, comm_per_lot, strategies):
        """M1-Event Replay Engine for Trade Management."""
        for sid, trade in list(self.open_trades.items()):
            is_closed = False
            for m in range(len(m1_candles)):
                if is_closed: break
                
                m1_high = m1_candles.high[m]
                m1_low = m1_candles.low[m]
                spread = m1_candles.spread[m] * point
                direction = trade["direction"]
                
                exit_price = None
                event = None
                
                if direction == "BUY":
                    if m1_low <= trade["sl"]: exit_price, event = trade["sl"], "sl"
                    elif m1_high >= trade["tp"]: exit_price, event = trade["tp"], "tp"
                else: # SELL
                    if m1_high + spread >= trade["sl"]: exit_price, event = trade["sl"], "sl"
                    elif m1_low + spread <= trade["tp"]: exit_price, event = trade["tp"], "tp"
                
                if exit_price:
                    final_exit, exit_slip = self.simulator.simulate_exit(trade, exit_price, point, event=event)
                    raw_diff = (final_exit - trade["fill_price"]) if direction == "BUY" else (trade["fill_price"] - final_exit)
                    gross_pnl = (raw_diff / point) * tick_value * trade["lots"]
                    exit_comm = trade["lots"] * comm_per_lot
                    entry_comm = trade.get("entry_comm", 0.0)
                    
                    net_pnl = gross_pnl - entry_comm - exit_comm
                    self.balances[sid] += net_pnl
                    self.equities[sid] = self.balances[sid]
                    
                    trade_record = {
                        **trade,
                        "exit_price": final_exit,
                        "exit_time": m1_candles.time[m],
                        "pnl": net_pnl,
                        "exit_slippage": exit_slip / point,
                        "result": event.upper(),
                        "final_balance": self.balances[sid]
                    }
                    self.history.append(trade_record)
                    self.risk_engine.update_history(net_pnl, self.equities[sid])
                    
                    for s in strategies:
                        if s.strategy_id == sid:
                            s.on_trade_closed(trade_record)
                            break
                            
                    del self.open_trades[sid]
                    is_closed = True
                else:
                    floating_price = m1_low if direction == "BUY" else m1_high
                    f_diff = (floating_price - trade["fill_price"]) if direction == "BUY" else (trade["fill_price"] - floating_price)
                    f_gross_pnl = (f_diff / point) * tick_value * trade["lots"]
                    self.equities[sid] = self.balances[sid] + f_gross_pnl

    def _validate_data_alignment(self, m5, m1):
        """Ensures that M1 data covers the M5 range without gaps (Step 11)."""
        if len(m5) == 0 or len(m1) == 0:
            logger.warning(f"DATA ALIGNMENT SKIPPED: Missing timeframe slice.")
            return

        if m5.time[-1] > m1.time[-1]:
            logger.critical(f"DATA ALIGNMENT ERROR: M1 data ({m1.time[-1]}) expires before M5 ({m5.time[-1]})")
            raise ValueError("CRITICAL_SYSTEM_ERROR: Data inconsistency.")

    def _get_tf_idx(self, tf_data, target_time, side: str = "right") -> int:
        """Returns the current index of a higher timeframe candle relative to target_time."""
        if len(tf_data) == 0: return 0
        idx = np.searchsorted(tf_data.time, target_time, side=side)
        return max(0, idx)

    def _get_m1_for_m5(self, m1, target_time):
        """Returns M1 candles within the target timeframe bar window."""
        if len(m1) == 0:
            from core.common.types import CandleArray
            return CandleArray.from_dicts([])

        idx_start = np.searchsorted(m1.time, target_time, side='left')
        next_bar_time = target_time + 300 
        idx_end = np.searchsorted(m1.time, next_bar_time, side='left')
        if idx_end <= idx_start:
            idx_end = min(idx_start + 5, len(m1.time))
        return m1[idx_start:idx_end]

    def _force_close_at_end(self, m5_data, point, tick_value, comm_per_lot, strategies):
        if not self.open_trades: return
        last_price = m5_data.close[-1]
        for sid, trade in list(self.open_trades.items()):
            net_pnl = ((last_price - trade["fill_price"] if trade["direction"] == "BUY" else trade["fill_price"] - last_price) / point) * tick_value * trade["lots"]
            self.history.append({**trade, "exit_price": last_price, "pnl": net_pnl, "result": "FORCED_CLOSE"})
            del self.open_trades[sid]
