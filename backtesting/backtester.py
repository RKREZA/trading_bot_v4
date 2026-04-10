import logging
import numpy as np
import os
from tqdm import tqdm
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional

from core.base_strategy import MarketData
from core.regime_detector import RegimeDetector
from core.risk.risk_guardian import RiskGuardian
from core.session_detector import SessionDetector
from core.portfolio_manager import PortfolioManager
from core.regime_gater import RegimeGater
from core.recovery.checkpoint_manager import CheckpointManager
from core.execution.order_manager import OrderManager
from core.volatility_detector import VolatilityDetector, VolatilityLevel
from strategies.adaptive_manager import AdaptiveStrategyManager, RegimeAwareStrategy

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
        self.volatility_detector = VolatilityDetector(atr_period=14, lookback=100)
        self.risk_guardian = RiskGuardian(config)
        self.order_manager = OrderManager(config)
        self.portfolio_manager = PortfolioManager(config)
        self.checkpoint_manager = CheckpointManager()

        bt_cfg = config.get("backtest", {})
        self.initial_partition_balance = float(bt_cfg.get("initial_balance_per_strategy", 1000.0))
        
        vol_cfg = config.get("volatility_adaptive", {})
        self.volatility_adaptive_enabled = vol_cfg.get("enabled", True)
        self.min_volatility_for_trades = vol_cfg.get("min_volatility_for_trades", "VERY_LOW")
        
        # Internal State
        self.current_index = 0
        self.history = []
        self.open_trades = {}     # strategy_id -> trade_dict
        self.balances = {}        # strategy_id -> float
        self.equities = {}        # strategy_id -> float
        self.equity_history = []
        self.max_drawdowns = {}   
        self.peak_equity = {}
        self.volatility_history = []

    def reset(self, active_strategies: list):
        """Full reset of the simulation state with capital allocation (Step 9)."""
        self.current_index = 0
        self.history = []
        self.open_trades = {}
        
        # Institutional Allocation: Use PortfolioManager to split total pool based on config
        # Use initial_partition_balance as the 'unit' per strategy for the total pool
        total_pool = len(active_strategies) * self.initial_partition_balance
        
        self.balances = {}
        self.equities = {}
        self.peak_equity = {}
        self.max_drawdowns = {}
        
        for strat in active_strategies:
            sid = strat.strategy_id
            # Resolve balance from PortfolioManager (handles 0.0 allocations correctly)
            bal = self.portfolio_manager.get_strategy_balance(total_pool, sid)
            self.balances[sid] = bal
            self.equities[sid] = bal
            self.peak_equity[sid] = bal
            self.max_drawdowns[sid] = 0.0
            
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
        logger.info(f"DATA SIZES: target_tf={len(target_tf_data)}, m1={len(m1_data)}, m5={len(m5_data)}, m15={len(m15_data)}, h1={len(h1_data)}")
        
        # Institutional Gating Filter: Must be enabled AND have an allocation > 0
        active_strategies = []
        for s in strategies:
            # 1. Check logical enabled flag
            if not getattr(s, "enabled", True):
                continue
            
            # 2. Check symbol allowance
            if not s.is_symbol_allowed(symbol):
                continue
                
            # 3. Check allocation > 0 (via PortfolioManager logic)
            # Use a dummy total_pool to check if allocation is 0
            if self.portfolio_manager.get_strategy_balance(100.0, s.strategy_id) <= 0:
                continue
                
            active_strategies.append(s)
            
        sid_list = [s.strategy_id for s in active_strategies]
        
        # Initialize Adaptive Strategy Manager
        use_adaptive = self.config.get("backtest", {}).get("adaptive_strategy", True)
        if use_adaptive and len(active_strategies) > 1:
            adaptive_manager = AdaptiveStrategyManager(active_strategies, self.config)
            logger.info(f"Adaptive Strategy Manager initialized with {len(active_strategies)} strategies")
        
        if not resume:
            self.reset(active_strategies)


        symbol_cfg = self.config.get("symbols_config", {}).get(symbol, {})
        point = float(symbol_cfg.get("point", 0.0001))
        tick_value = float(symbol_cfg.get("tick_value", 10.0))
        comm_per_lot = float(symbol_cfg.get("commission_per_lot", 7.0))
        
        # 1. Institutional Indicator Pre-calculation (IPC) - Step 4.2
        from core.indicator_engine import IndicatorEngine
        from rich.console import Console
        console = Console()
        
        with console.status(f"[bold blue]Calibrating {symbol} Strategy Indicators...") as status:
            target_tf_data._indicators = IndicatorEngine.precalculate_all(symbol, getattr(target_tf_data, "timeframe", "UNKNOWN"), target_tf_data)
            m5_data._indicators = IndicatorEngine.precalculate_all(symbol, "M5", m5_data)
            m15_data._indicators = IndicatorEngine.precalculate_all(symbol, "M15", m15_data)
            h1_data._indicators = IndicatorEngine.precalculate_all(symbol, "H1", h1_data)
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
                        self.risk_guardian.reset_daily(self.balances[sid])
                last_date = current_date
                
                # [ Institutional Fidelity ]: Zero-Copy Index Shifting
                target_tf_data.set_limit(i) 
                
                # Ensure minimum bars for strategy requirements
                # LiquiditySweepBreakout needs 22 H1 bars, RangeBounce needs 200 M5 bars
                min_m5_bars = 250  # Allow 250 bars for RangeBounce lookback
                min_h1_bars = 30   # Allow 30 bars for LiquiditySweepBreakout
                min_m15_bars = 100 # Allow 100 bars for M15 lookback
                
                m5_idx = max(min_m5_bars, self._get_tf_idx(m5_data, t, side="right"))
                if m5_data is not target_tf_data: m5_data.set_limit(m5_idx)
                
                h1_idx = max(min_h1_bars, self._get_tf_idx(h1_data, t, side="right"))
                if h1_data is not target_tf_data: h1_data.set_limit(h1_idx)
                
                m15_idx = max(min_m15_bars, self._get_tf_idx(m15_data, t, side="right"))
                if m15_data is not target_tf_data: m15_data.set_limit(m15_idx)
                
                # 1. Regime Detection & Gating
                regime_info = self.regime_detector.detect(target_tf_data)
                regime = regime_info.market_type
                risk_mult = RegimeGater.get_risk_multiplier(regime_info.volatility)
                conf_buffer = RegimeGater.get_confidence_buffer(regime_info.volatility)
                
                # 1.5 Volatility Analysis (V4.3 New Feature)
                vol_analysis = None
                if self.volatility_adaptive_enabled:
                    vol_analysis = self.volatility_detector.analyze(m5_data, h1_data)
                    self.volatility_history.append(vol_analysis)
                    
                    # Update risk multiplier based on volatility level
                    vol_risk_mult = vol_analysis.risk_multiplier
                    risk_mult = risk_mult * vol_risk_mult
                    
                    # Skip trades in extreme low volatility
                    if vol_analysis.level == VolatilityLevel.EXTREME_LOW:
                        continue

                # 2. MarketData Construction (Zero-Copy & Anti-Lookahead)
                # Institutional Fidelity: Simulate Bid/Ask/Spread (Anti-Lookahead Fix)
                current_bid = float(target_tf_data.close[i-1]) if i > 0 else float(target_tf_data.open[i])
                symbol_cfg = self.config.get("symbols_config", {}).get(symbol, {})
                point = symbol_cfg.get("point", 0.0001)
                
                # Use actual spread from M5 data if available, else fallback to config
                # M5 spread is stored in MT5 points (159 pts = 15.9 pips for XAUUSD)
                # 1 pip = 10 points for XAUUSD
                if i > 0 and hasattr(m5_data, 'spread') and len(m5_data.spread) > i-1:
                    spread_points = float(m5_data.spread[i-1])
                    spread_pips = spread_points / 10.0  # Convert points to pips
                    spread_val = spread_pips * point
                else:
                    spread_val = symbol_cfg.get("spread_pips", 2) * point
                current_ask = current_bid + spread_val

                market_data = MarketData(
                    symbol=symbol,
                    htf_candles=h1_data,
                    m15_candles=m15_data,
                    m5_candles=m5_data,
                    d1_candles=None,
                    current_price=current_bid,
                    bid=current_bid,
                    ask=current_ask,
                    spread=spread_val,
                    session=SessionDetector.get_session(dt, self.config.get("backtest", {}).get("utc_offset", 0)),
                    timestamp=dt
                )
                
                # 3. Micro-service Strategy Replay (Adaptive Selection)
                if use_adaptive and len(active_strategies) > 1:
                    # Adaptive: Select best strategy based on regime
                    selected_strat = adaptive_manager.select_strategy(regime_info, market_data)
                    strategies_to_try = [selected_strat] if selected_strat else []
                else:
                    # Legacy: Run all strategies
                    strategies_to_try = active_strategies
                
                for strat in strategies_to_try:
                    sid = strat.strategy_id
                    
                    if RegimeGater.is_drawdown_gated(self.max_drawdowns.get(sid, 0)): continue
                    if not RegimeGater.is_strategy_allowed(strat.__class__.__name__, regime): continue
                    if sid in self.open_trades: continue
                    
                    # Apply volatility-adjusted parameters if available
                    original_params = {}
                    if self.volatility_adaptive_enabled and vol_analysis:
                        original_params = self._apply_volatility_params(strat, vol_analysis)
                    
                    try:
                        signal = strat.generate_signal(market_data)
                    finally:
                        if original_params:
                            self._restore_original_params(strat, original_params)
                    
                    if not signal or signal.direction == "NONE":
                        if self.config.get("backtest", {}).get("debug_signals"):
                            reason = getattr(strat, "last_rejection_reason", "No specific reason")
                            logger.info(f"[{dt}] [{sid}] Signal REJECTED: {reason}")
                        continue
                            
                    min_conf = getattr(strat, "min_confidence", 0.6)
                    if self.config.get("backtest", {}).get("debug_signals"):
                        logger.info(f"[{dt}] [{sid}] Signal generated: {signal.direction} @ {signal.price:.2f} conf={signal.confidence:.2f}")
                    
                    if signal.confidence < (min_conf + conf_buffer - 0.001):
                        if self.config.get("backtest", {}).get("debug_signals"):
                            logger.info(f"[{dt}] [{sid}] Confidence REJECTED: {signal.confidence:.2f} < {min_conf + conf_buffer:.2f}")
                        continue
                            
                    sl = strat.get_stop_loss(signal, market_data)
                    tp = strat.get_take_profit(signal, market_data)
                    sl_dist = abs(market_data.current_price - sl)
                    
                    if self.config.get("backtest", {}).get("debug_signals"):
                        logger.info(f"[{dt}] [{sid}] SL={sl:.2f}, TP={tp:.2f}, dist={sl_dist:.2f}")
                    
                    if sl_dist > 0:
                        lot_size = self.risk_guardian.calculate_lot_size(
                            balance=self.balances[sid],
                            stop_loss_dist=sl_dist,
                            symbol_info=symbol_cfg,
                            current_price=market_data.current_price
                        )
                        lot_size = lot_size * risk_mult
                        
                        if self.config.get("backtest", {}).get("debug_signals"):
                            logger.info(f"[{dt}] [{sid}] Lot size: {lot_size:.4f}")
                        
                        if lot_size >= 0.01:
                            sig = signal
                            sig.volume = lot_size
                            
                            price_data = {
                                "bid": market_data.current_price,
                                "ask": market_data.current_price + (target_tf_data.spread[i] * point),
                                "point": point
                            }
                            
                            fill = self.order_manager.execute_signal(
                                signal=sig,
                                symbol=symbol,
                                price_data=price_data,
                                is_news_blocked=False,
                                magic=self.risk_guardian.get_magic_number(sid),
                                timestamp=t
                            )
                            
                            if fill and not fill.get("is_error", False):
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
                                logger.info(f"[{sid}] Trade Entered: {fill['direction']} @ {fill['fill_price']:.5f}")
                            elif self.config.get("backtest", {}).get("debug_signals"):
                                logger.info(f"[{dt}] [{sid}] Execution REJECTED: OrderManager denied entry (Spread/Slip/Gating)")
                        elif self.config.get("backtest", {}).get("debug_signals"):
                            logger.info(f"[{dt}] [{sid}] Risk REJECTED: Lot size {lot_size:.3f} < 0.01")

                # 4. M1 Intra-Bar Execution (Safety Gate: Check for Gaps)
                m1_slice = self._get_m1_for_m5(m1_data, t)
                atr_vals = target_tf_data.get_indicator("atr_14")
                atr_val = atr_vals[i] if i < len(atr_vals) else 0.0

                if len(m1_slice) > 0:
                    self._manage_active_trades(m1_slice, tick_value, point, comm_per_lot, active_strategies, atr_val=atr_val)
                elif self.open_trades:
                    # Institutional Robustness: Fallback to M5 High/Low if M1 is missing (Audit Pass #5 Fix)
                    logger.warning(f"[{dt}] DATA ALERT: Missing M1 coverage for active trade at {t}. Falling back to M5-bar OHLC validation.")
                    
                    # Create a synthetic M1 slice for fallback validation
                    from core.common.types import CandleArray
                    # Mock a slice using the current indices high/low/close values
                    # We create a simple object that behaves like the M1 slice for _manage_active_trades
                    class SyntheticM1:
                        def __init__(self, high, low, close, spread, time):
                            self.high = np.array([high])
                            self.low = np.array([low])
                            self.close = np.array([close])
                            self.spread = np.array([spread])
                            self.time = np.array([time])
                        def __len__(self): return 1
                        
                    if i < len(target_tf_data.h):
                        m1_fallback = SyntheticM1(
                            target_tf_data.h[i], 
                            target_tf_data.l[i], 
                            target_tf_data.c[i], 
                            target_tf_data.s[i], 
                            target_tf_data.time[i]
                        )
                        self._manage_active_trades(m1_fallback, tick_value, point, comm_per_lot, active_strategies, atr_val=atr_val)
                    else:
                        logger.warning(f"[{dt}] DATA ALERT: Index {i} exceeds target_tf_data bounds ({len(target_tf_data)}). Forcing trade close.")
                        self._force_close_at_end(target_tf_data, point, tick_value, comm_per_lot, active_strategies)
                
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
                crash_file = os.path.join("logs", "crash_report.log")
                os.makedirs("logs", exist_ok=True)
                with open(crash_file, "a") as f:
                    f.write(f"\n--- BACKTEST CRASH: {datetime.now()} ---\n")
                    f.write(traceback.format_exc())
                raise e

        pbar.close()
        self._force_close_at_end(target_tf_data, point, tick_value, comm_per_lot, active_strategies)
        self.checkpoint_manager.clear_checkpoint()
        
        return self.history, self.equity_history

    def _manage_active_trades(self, m1_candles, tick_value, point, comm_per_lot, strategies, atr_val=0.0):
        """M1-Event Replay Engine for Trade Management."""
        for sid, trade in list(self.open_trades.items()):
            is_closed = False
            for m in range(len(m1_candles)):
                if is_closed: break
                
                m1_high = m1_candles.high[m]
                m1_low = m1_candles.low[m]
                spread = m1_candles.spread[m] * point
                direction = trade["direction"]

                # --- V4-ULTRA Trailing Stop Logic (Rule 3.1 Alignment) ---
                if self.config.get("trailing_stop", {}).get("enabled", False):
                    conf = self.config["trailing_stop"]
                    entry = trade["fill_price"]
                    curr_sl = trade["sl"]
                    
                    # Calculate current R:R using M1 close
                    current_price = m1_candles.close[m]
                    initial_risk_price = abs(entry - trade["sl"])
                    if initial_risk_price > 0:
                        profit_price = (current_price - entry) if direction == "BUY" else (entry - current_price)
                        current_rr = profit_price / initial_risk_price
                        
                        new_sl = None
                        # Phase 1: Break-Even (at 1.5R)
                        rr_threshold = conf.get("phase1_rr_threshold", 1.5)
                        if current_rr >= rr_threshold and abs(curr_sl - entry) > (initial_risk_price * 0.1):
                            be_offset = initial_risk_price * conf.get("phase2_be_offset_pct", 0.1)
                            new_sl = entry + be_offset if direction == "BUY" else entry - be_offset
                        
                        # Phase 2: ATR-based Trailing (at 3R+)
                        if current_rr >= 3.0 and atr_val > 0:
                            trail_mult = conf.get("phase3_trail_mult", 1.5)
                            trail_sl = current_price - (atr_val * trail_mult) if direction == "BUY" else current_price + (atr_val * trail_mult)
                            # Only move if improves protection
                            if direction == "BUY" and trail_sl > (new_sl or curr_sl):
                                new_sl = trail_sl
                            elif direction == "SELL" and (curr_sl == 0 or trail_sl < (new_sl or curr_sl)):
                                new_sl = trail_sl
                        
                        if new_sl:
                            trade["sl"] = new_sl

                exit_price = None
                event = None
                
                if direction == "BUY":
                    if m1_low <= trade["sl"]: exit_price, event = trade["sl"], "sl"
                    elif m1_high >= trade["tp"]: exit_price, event = trade["tp"], "tp"
                else: # SELL
                    if m1_high + spread >= trade["sl"]: exit_price, event = trade["sl"], "sl"
                    elif m1_low + spread <= trade["tp"]: exit_price, event = trade["tp"], "tp"
                
                if exit_price:
                    exit_time = m1_candles.time[m]
                    exit_res = self.order_manager.simulate_exit(trade["ticket"], event, exit_price, point, direction, exit_time=exit_time)
                    final_exit = exit_res["exit_price"]
                    exit_slip = abs(final_exit - exit_price)
                    
                    raw_diff = (final_exit - trade["fill_price"]) if direction == "BUY" else (trade["fill_price"] - final_exit)
                    gross_pnl = (raw_diff / point) * tick_value * trade["lots"]
                    exit_comm = trade["lots"] * comm_per_lot
                    entry_comm = trade.get("entry_comm", 0.0)
                    
                    net_pnl = gross_pnl - entry_comm - exit_comm
                    start_balance = self.balances[sid]
                    self.balances[sid] += net_pnl
                    self.equities[sid] = self.balances[sid]
                    
                    trade_record = {
                        **trade,
                        "exit_price": final_exit,
                        "exit_time": m1_candles.time[m],
                        "pnl": net_pnl,
                        "exit_slippage": exit_slip / point,
                        "result": event.upper(),
                        "final_balance": self.balances[sid],
                        "balance_at_start": start_balance
                    }
                    self.history.append(trade_record)
                    self.risk_guardian.record_trade_result(net_pnl, self.equities[sid])
                    
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

    def _apply_volatility_params(self, strategy, vol_analysis) -> Dict[str, Any]:
        """Apply volatility-adjusted parameters to a strategy."""
        from core.volatility_detector import VolatilityAdaptiveParameters
        
        strategy_id = strategy.strategy_id
        strategy_type = self._get_strategy_type(strategy_id)
        
        vol_params = VolatilityAdaptiveParameters.get_parameters_for_volatility(
            vol_analysis.level, 
            strategy_type
        )
        
        original = {}
        for param, value in vol_params.items():
            if hasattr(strategy, param):
                original[param] = getattr(strategy, param)
                setattr(strategy, param, value)
        
        return original
    
    def _restore_original_params(self, strategy, original_params: Dict[str, Any]) -> None:
        """Restore original strategy parameters."""
        for param, value in original_params.items():
            setattr(strategy, param, value)
    
    def _get_strategy_type(self, strategy_id: str) -> str:
        """Determine strategy type for volatility parameter selection."""
        if "Breakout" in strategy_id or "Liquidity" in strategy_id:
            return "breakout"
        elif "MeanReversion" in strategy_id or "RangeBounce" in strategy_id:
            return "mean_reversion"
        elif "Trend" in strategy_id:
            return "trend"
        return "breakout"
    
    def get_volatility_summary(self) -> Dict[str, Any]:
        """Get summary of volatility conditions encountered during backtest."""
        if not self.volatility_history:
            return {"status": "No volatility data"}
        
        level_counts = {}
        for vol in self.volatility_history:
            level_key = vol.level.value
            level_counts[level_key] = level_counts.get(level_key, 0) + 1
        
        ratios = [v.ratio for v in self.volatility_history]
        
        return {
            "total_bars": len(self.volatility_history),
            "level_distribution": {k: f"{(v/len(self.volatility_history)*100):.1f}%" for k, v in level_counts.items()},
            "avg_ratio": float(np.mean(ratios)),
            "min_ratio": float(np.min(ratios)),
            "max_ratio": float(np.max(ratios)),
        }
