import logging
import numpy as np
from tqdm import tqdm
from datetime import datetime, timezone
from core.base_strategy import MarketData
from core.regime_detector import RegimeDetector
from core.risk_engine import RiskEngine
from core.execution_engine import ExecutionEngine
from core.logger import BrokerClock
from core.session_detector import SessionDetector
from core.portfolio_manager import PortfolioManager

logger = logging.getLogger("trading_bot.backtester")

class PortfolioBacktester:
    """
    V4 Institutional Multi-Strategy Backtester.
    Runs M1-Fidelity simulation with professional cost modeling and portfolio management.
    Supports individual strategy balance partitioning ($1000 each).
    """

    def __init__(self, config: dict):
        self.config = config
        self.regime_detector = RegimeDetector()
        self.risk_engine = RiskEngine(config)
        self.execution_engine = ExecutionEngine(config)
        self.portfolio_manager = PortfolioManager(config)

        bt_cfg = config.get("backtest", {}) if isinstance(config.get("backtest", {}), dict) else {}
        costs_cfg = bt_cfg.get("costs", {}) if isinstance(bt_cfg.get("costs", {}), dict) else {}

        self.initial_partition_balance = float(bt_cfg.get("initial_balance", config.get("initial_balance", 1000.0)))
        self.deterministic = bool(bt_cfg.get("deterministic", False))
        self.random_seed = bt_cfg.get("random_seed")
        self._run_counter = 0

        self.entry_commission_sides = float(costs_cfg.get("entry_commission_sides", 1.0))
        self.exit_commission_sides = float(costs_cfg.get("exit_commission_sides", 1.0))

        # State tracking
        self.history = []
        self.open_trades = {} # strategy_id -> trade_dict
        self.balances = {}    # strategy_id -> float
        self.equities = {}    # strategy_id -> float

    def run(self, symbol: str, strategies: list, m5_data, h1_data, m15_data, m1_data):
        """
        Main simulation loop.
        Iterates over M5 candles, then drills down into M1 for trade management.
        """
        logger.info(f"Starting Multi-Strategy Backtest on {symbol}...")

        # Reset state per run
        self.history = []
        self.open_trades = {}
        self.balances = {}
        self.equities = {}

        # Re-seed execution RNG for reproducibility controls
        if self.random_seed is not None:
            seed = int(self.random_seed) if self.deterministic else int(self.random_seed) + self._run_counter
            self.execution_engine.reset_rng(seed)
        self._run_counter += 1

        # Initialize partitioned balances
        for strat in strategies:
            self.balances[strat.strategy_id] = self.initial_partition_balance
            self.equities[strat.strategy_id] = self.initial_partition_balance

        # Symbol-specific parameters for realism
        symbol_cfg = self.config.get("symbols_config", {}).get(symbol, {})
        point = float(symbol_cfg.get("point", 0.0001))
        tick_value = float(symbol_cfg.get("tick_value", 10.0))
        commission_per_lot = float(symbol_cfg.get("commission_per_lot", 7.0))
        spread_pips = float(symbol_cfg.get("spread_pips", 20.0))
        self.spread_cost = spread_pips * point
        self.execution_engine.max_spread_pips = max(self.execution_engine.max_spread_pips, spread_pips * 1.2)

        pbar = tqdm(total=len(m5_data.time) - 100)
        
        for i in range(100, len(m5_data.time)):
            pbar.update(1)
            t = m5_data.time[i]
            BrokerClock.set_time(t) # Synchronize Global Logger with Broker's Time
            dt = datetime.fromtimestamp(t, tz=timezone.utc)
            
            # 1. Regime Detection (M5-based)
            _ = self.regime_detector.detect(m5_data[:i])

            # 2. MarketData Object Construction
            h1_slice = self._slice_tf_history(h1_data, t, lookback_bars=200)
            m15_slice = self._slice_tf_history(m15_data, t, lookback_bars=200)
            market_data = MarketData(
                symbol=symbol,
                htf_candles=h1_slice,
                m15_candles=m15_slice,
                m5_candles=m5_data[:i+1],
                d1_candles=None,
                current_price=m5_data.close[i],
                session=SessionDetector.get_session(dt),
                timestamp=dt
            )
            
            # 3. Strategy Signal Generation
            signals = {}
            for strat in strategies:
                if strat.enabled:
                    sig = strat.generate_signal(market_data)
                    if sig and sig.direction != "NONE":
                        signals[strat.strategy_id] = sig
            
            # 4. Signal Resolution
            best_id_sig = self.portfolio_manager.resolve_signals(signals)
            
            # 5. Risk Check & Execution
            if best_id_sig:
                sid, sig = best_id_sig
                if not self.open_trades.get(sid):
                    # Each strategy uses its OWN $1000 partition
                    strat_balance = self.balances[sid]
                    strat_equity = self.equities[sid]
                    
                    # Daily reset (simplified for backtest)
                    if dt.hour == 0 and dt.minute == 0:
                        self.risk_engine.record_daily_close(strat_balance)
                    
                    # Check Circuit Breakers for this specific strategy partition
                    allowed, reason = self.risk_engine.check_circuit_breakers(strat_balance, strat_equity)
                    if allowed:
                        strat = next(s for s in strategies if s.strategy_id == sid)
                        sig.stop_loss = strat.get_stop_loss(sig, market_data)
                        sig.take_profit = strat.get_take_profit(sig, market_data)
                        
                        sl_dist = abs(market_data.current_price - sig.stop_loss)
                        # Position sizing based on the $1000 partition!
                        lots = self.risk_engine.calculate_lot_size(
                            balance=strat_balance, 
                            stop_loss_distance=sl_dist, 
                            point=point, 
                            tick_value=tick_value, 
                            symbol=symbol,
                            spread_points=spread_pips,
                            commission_per_lot=commission_per_lot
                        )
                        
                        if lots >= 0.01:
                            fill = self.execution_engine.execute_order(
                                sig,
                                symbol,
                                m5_data.close[i],
                                self.spread_cost,
                                point,
                                timestamp=t
                            )
                            if fill:
                                fill["lots"] = lots
                                fill["strategy_id"] = sid
                                fill["session"] = market_data.session
                                fill["entry_balance"] = strat_balance

                                # Entry cost model
                                entry_commission = lots * commission_per_lot * self.entry_commission_sides
                                fill["entry_commission"] = entry_commission
                                self.balances[sid] -= entry_commission
                                self.open_trades[sid] = fill

            # 6. M1-Fidelity Trade Management
            m1_slice = self._get_m1_for_m5(m1_data, t)
            self._manage_trades(m1_slice, tick_value, point, commission_per_lot, strategies)

        pbar.close()

        # Force-close any remaining open trades at final M5 close
        self._force_close_open_trades(m5_data, strategies, tick_value, point, commission_per_lot)
        return self.history

    def _slice_tf_history(self, tf_data, current_time, lookback_bars: int = 200):
        end_idx = np.searchsorted(tf_data.time, current_time, side="right")
        start_idx = max(0, end_idx - lookback_bars)
        return tf_data[start_idx:end_idx]

    def _get_m1_for_m5(self, m1_data, m5_time):
        idx_start = np.searchsorted(m1_data.time, m5_time)
        return m1_data[idx_start:idx_start+5]

    def _manage_trades(self, m1_candles, tick_value, point, commission_per_lot, strategies):
        for sid, trade in list(self.open_trades.items()):
            for m in range(len(m1_candles)):
                high = m1_candles.high[m]
                low = m1_candles.low[m]
                
                direction = trade["direction"]
                exit_price = None
                result = None
                
                if direction == "BUY":
                    if low <= trade["sl"]:
                        exit_price = trade["sl"]
                        result = "SL"
                    elif high >= trade["tp"]:
                        exit_price = trade["tp"]
                        result = "TP"
                else: # SELL
                    if high >= trade["sl"]:
                        exit_price = trade["sl"]
                        result = "SL"
                    elif low <= trade["tp"]:
                        exit_price = trade["tp"]
                        result = "TP"
                
                if exit_price:
                    raw_diff = (exit_price - trade["fill_price"]) if direction == "BUY" else (trade["fill_price"] - exit_price)
                    exit_event = "tp_exit" if result == "TP" else "sl_exit"
                    exit_slippage = self.execution_engine.sample_slippage_points(point, event=exit_event)

                    gross_pnl = ((raw_diff - self.spread_cost - exit_slippage) / point) * tick_value * trade["lots"]
                    exit_commission = trade["lots"] * commission_per_lot * self.exit_commission_sides
                    entry_commission = float(trade.get("entry_commission", 0.0))
                    net_trade_pnl = gross_pnl - entry_commission - exit_commission

                    # Update Strategy-Specific Balance
                    self.balances[sid] += (gross_pnl - exit_commission)
                    self.equities[sid] = self.balances[sid]

                    # Margin Call Check (Realistic Protection)
                    if self.balances[sid] <= 0:
                        self.balances[sid] = 0.0
                        logger.critical(f"MARGIN CALL: Strategy {sid} account blown. Results finalized at 0.")

                    trade_record = {
                        **trade,
                        "session": trade.get("session", "GLOBAL"),
                        "exit_price": exit_price,
                        "exit_time": m1_candles.time[m],
                        "gross_pnl": gross_pnl,
                        "entry_commission": entry_commission,
                        "exit_commission": exit_commission,
                        "exit_slippage_pips": exit_slippage / point if point > 0 else 0.0,
                        "pnl": net_trade_pnl,
                        "result": result,
                        "final_balance": self.balances[sid]
                    }
                    self.history.append(trade_record)
                    self.risk_engine.update_history(net_trade_pnl, self.equities[sid])

                    for s in strategies:
                        if s.strategy_id == sid:
                            s.on_trade_closed(trade_record)
                            break

                    del self.open_trades[sid]

                    if self.balances[sid] <= 0:
                        # Disable strategy if account is blown
                        for s in strategies:
                            if s.strategy_id == sid:
                                s.enabled = False
                    break

    def _force_close_open_trades(self, m5_data, strategies, tick_value, point, commission_per_lot):
        if not self.open_trades or len(m5_data) == 0:
            return

        final_price = float(m5_data.close[-1])
        final_time = int(m5_data.time[-1])

        for sid, trade in list(self.open_trades.items()):
            direction = trade["direction"]
            raw_diff = (final_price - trade["fill_price"]) if direction == "BUY" else (trade["fill_price"] - final_price)
            exit_slippage = self.execution_engine.sample_slippage_points(point, event="forced_exit")

            gross_pnl = ((raw_diff - self.spread_cost - exit_slippage) / point) * tick_value * trade["lots"]
            exit_commission = trade["lots"] * commission_per_lot * self.exit_commission_sides
            entry_commission = float(trade.get("entry_commission", 0.0))
            net_trade_pnl = gross_pnl - entry_commission - exit_commission

            self.balances[sid] += (gross_pnl - exit_commission)
            self.equities[sid] = self.balances[sid]

            trade_record = {
                **trade,
                "session": trade.get("session", "GLOBAL"),
                "exit_price": final_price,
                "exit_time": final_time,
                "gross_pnl": gross_pnl,
                "entry_commission": entry_commission,
                "exit_commission": exit_commission,
                "exit_slippage_pips": exit_slippage / point if point > 0 else 0.0,
                "pnl": net_trade_pnl,
                "result": "FORCED_CLOSE",
                "final_balance": self.balances[sid]
            }
            self.history.append(trade_record)
            self.risk_engine.update_history(net_trade_pnl, self.equities[sid])

            for s in strategies:
                if s.strategy_id == sid:
                    s.on_trade_closed(trade_record)
                    break

            del self.open_trades[sid]
