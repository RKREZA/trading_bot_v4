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

        bt_cfg = config.get("backtest", {})
        costs_cfg = bt_cfg.get("costs", {})

        # Micro-service Partitioning
        self.initial_partition_balance = float(bt_cfg.get("initial_balance_per_strategy", 1000.0))
        self.deterministic = bool(bt_cfg.get("deterministic", False))
        self.random_seed = bt_cfg.get("random_seed")
        self._run_counter = 0

        self.entry_commission_sides = float(costs_cfg.get("entry_commission_sides", 1.0))
        self.exit_commission_sides = float(costs_cfg.get("exit_commission_sides", 1.0))

        # State tracking
        self.history = []
        self.open_trades = {}     # strategy_id -> trade_dict
        self.balances = {}        # strategy_id -> float
        self.equities = {}        # strategy_id -> float
        self.equity_history = []  # list of dicts: {time, strategy_id, equity}
        self.max_drawdowns = {}   # strategy_id -> float

    def run(self, symbol: str, strategies: list, m5_data, h1_data, m15_data, m1_data):
        """
        Main simulation loop with Regime Gating and Partitioned Capital.
        """
        logger.info(f"Starting V4 Institutional Isolation Backtest on {symbol}...")

        active_strategies = [s for s in strategies if getattr(s, "enabled", True) and s.is_symbol_allowed(symbol)]
        if not active_strategies:
            logger.warning("No enabled strategies mapped to symbol %s", symbol)
            return [], []

        # Reset state per run
        self.history = []
        self.open_trades = {}
        self.balances = {}
        self.equities = {}
        self.equity_history = []
        self.max_drawdowns = {s.strategy_id: 0.0 for s in active_strategies}
        self.peak_equity = {s.strategy_id: self.initial_partition_balance for s in active_strategies}

        # Initialize partitioned balances
        for strat in active_strategies:
            self.balances[strat.strategy_id] = self.initial_partition_balance
            self.equities[strat.strategy_id] = self.initial_partition_balance

        symbol_cfg = self.config.get("symbols_config", {}).get(symbol, {})
        point = float(symbol_cfg.get("point", 0.0001))
        tick_value = float(symbol_cfg.get("tick_value", 10.0))
        commission_per_lot = float(symbol_cfg.get("commission_per_lot", 7.0))
        spread_pips = float(symbol_cfg.get("spread_pips", 20.0))
        self.spread_cost = spread_pips * point

        from core.regime_gater import RegimeGater
        pbar = tqdm(total=len(m5_data.time) - 100)
        
        for i in range(100, len(m5_data.time)):
            pbar.update(1)
            t = m5_data.time[i]
            BrokerClock.set_time(t)
            dt = datetime.fromtimestamp(t, tz=timezone.utc)
            
            # 1. Regime Detection & Gating
            regime_info = self.regime_detector.detect(m5_data[:i+1])
            regime = regime_info.type
            risk_mult = RegimeGater.get_risk_multiplier(regime)
            conf_buffer = RegimeGater.get_confidence_buffer(regime)

            # 2. MarketData Construction
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
            
            # 3. Micro-service Cycle
            for strat in active_strategies:
                sid = strat.strategy_id
                
                # Check Gating
                if not RegimeGater.is_strategy_allowed(strat.__class__.__name__, regime):
                    continue
                
                # Check Circuit Breakers
                allowed, _ = self.risk_engine.check_circuit_breakers(self.balances[sid], self.equities[sid])
                if not allowed: continue

                # Check if position already open for this strategy
                if sid in self.open_trades: continue

                # Generate Signal
                sig = strat.generate_signal(market_data)
                if sig and sig.direction != "NONE":
                    # Confidence check
                    if sig.confidence < (0.60 + conf_buffer): continue
                    
                    sig.stop_loss = strat.get_stop_loss(sig, market_data)
                    sig.take_profit = strat.get_take_profit(sig, market_data)
                    sl_dist = abs(market_data.current_price - sig.stop_loss)
                    
                    if sl_dist > 0:
                        lots = self.risk_engine.calculate_lot_size(
                            balance=self.balances[sid], 
                            stop_loss_distance=sl_dist, 
                            point=point, 
                            tick_value=tick_value, 
                            symbol=symbol,
                            spread_points=spread_pips,
                            commission_per_lot=commission_per_lot
                        )
                        lots = lots * risk_mult
                        
                        if lots >= 0.01:
                            fill = self.execution_engine.execute_order(sig, symbol, market_data.current_price, self.spread_cost, point, timestamp=t)
                            if fill:
                                fill.update({"lots": lots, "strategy_id": sid, "session": market_data.session, "entry_balance": self.balances[sid]})
                                entry_comm = lots * commission_per_lot * self.entry_commission_sides
                                self.balances[sid] -= entry_comm
                                self.open_trades[sid] = fill

            # 4. M1 Management & Equity Tracking
            m1_slice = self._get_m1_for_m5(m1_data, t)
            self._manage_trades(m1_slice, tick_value, point, commission_per_lot, active_strategies)
            
            # Update DD for each strategy
            for sid in self.balances:
                self.peak_equity[sid] = max(self.peak_equity[sid], self.equities[sid])
                dd = (self.peak_equity[sid] - self.equities[sid]) / self.peak_equity[sid] * 100
                self.max_drawdowns[sid] = max(self.max_drawdowns[sid], dd)
                self.equity_history.append({"time": t, "strategy_id": sid, "equity": self.equities[sid]})

        pbar.close()
        self._force_close_open_trades(m5_data, active_strategies, tick_value, point, commission_per_lot)
        return self.history, self.equity_history

    def get_summary(self) -> dict:
        """Returns isolated performance metrics for each strategy."""
        summary = {}
        for sid in self.balances:
            trades = [h for h in self.history if h["strategy_id"] == sid]
            win_rate = (len([t for t in trades if t["pnl"] > 0]) / len(trades) * 100) if trades else 0
            net_pnl = sum([t["pnl"] for t in trades])
            summary[sid] = {
                "balance": self.balances[sid],
                "net_pnl": net_pnl,
                "trades": len(trades),
                "win_rate": win_rate,
                "max_dd": self.max_drawdowns.get(sid, 0.0)
            }
        return summary

        pbar.close()

        # Force-close any remaining open trades at final M5 close
        self._force_close_open_trades(m5_data, active_strategies, tick_value, point, commission_per_lot)
        
        return self.history, self.equity_history


    def _slice_tf_history(self, tf_data, current_time, lookback_bars: int = 200):
        end_idx = np.searchsorted(tf_data.time, current_time, side="right")
        start_idx = max(0, end_idx - lookback_bars)
        return tf_data[start_idx:end_idx]

    def _get_m1_for_m5(self, m1_data, m5_time):
        idx_start = np.searchsorted(m1_data.time, m5_time)
        return m1_data[idx_start:idx_start+5]

    def _manage_trades(self, m1_candles, tick_value, point, commission_per_lot, strategies):
        for sid, trade in list(self.open_trades.items()):
            # Track if trade is closed to skip further M1 bars in this M5 cycle
            is_closed = False
            
            for m in range(len(m1_candles)):
                if is_closed: break

                high = m1_candles.high[m]
                low = m1_candles.low[m]
                spread = m1_candles.spread[m] * point
                direction = trade["direction"]
                exit_price = None
                result = None
                
                # 1. Check SL/TP with Bid/Ask Realism (Step 8)
                if direction == "BUY":
                    # Long Exit happens at BID (Lower)
                    if low <= trade["sl"]:
                        exit_price = trade["sl"]
                        result = "SL"
                    elif high >= trade["tp"]:
                        exit_price = trade["tp"]
                        result = "TP"
                else: # SELL
                    # Short Exit happens at ASK (Higher)
                    if high + spread >= trade["sl"]:
                        exit_price = trade["sl"]
                        result = "SL"
                    elif low + spread <= trade["tp"]:
                        exit_price = trade["tp"]
                        result = "TP"
                
                if exit_price:
                    # CALCULATE FINAL PNL
                    # Costs: Bid/Ask spread is already factored into hit logic.
                    # We just need to ensure the exit_price used for PnL is the correct side.
                    raw_diff = (exit_price - trade["fill_price"]) if direction == "BUY" else (trade["fill_price"] - exit_price)
                    
                    exit_event = "tp_exit" if result == "TP" else "sl_exit"
                    exit_slippage = self.execution_engine.sample_slippage_points(point, event=exit_event)

                    # Final PnL deducts slippage and commissions
                    gross_pnl = ((raw_diff - exit_slippage) / point) * tick_value * trade["lots"]
                    exit_commission = trade["lots"] * commission_per_lot * self.exit_commission_sides
                    entry_commission = float(trade.get("entry_commission", 0.0))
                    
                    # [ Institutional Capping ]: A strategy can never lose more than its partitioned capital
                    available_balance = self.balances[sid] - exit_commission
                    if gross_pnl < -available_balance:
                        gross_pnl = -available_balance
                    
                    net_trade_pnl = gross_pnl - entry_commission - exit_commission

                    # Update Account
                    self.balances[sid] += (gross_pnl - exit_commission)
                    self.equities[sid] = self.balances[sid]

                    # Margin Call Check
                    if self.balances[sid] <= 1.0: # Close to zero
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
                    is_closed = True
                
                else:
                    # 2. Update Floating Equity (For worst-case drawdown tracking)
                    # Logic: Use the 'worst' side of the M1 candle to capture max potential drawdown
                    floating_price = low if direction == "BUY" else high
                    f_diff = (floating_price - trade["fill_price"]) if direction == "BUY" else (trade["fill_price"] - floating_price)
                    f_gross_pnl = ((f_diff - self.spread_cost) / point) * tick_value * trade["lots"]
                    
                    # Update local equity snapshot
                    self.equities[sid] = self.balances[sid] + f_gross_pnl
                    
            # Sampling equity at the end of M1 cycle for this strategy
            if len(m1_candles) > 0:
                self.equity_history.append({
                    "time": m1_candles.time[-1],
                    "strategy_id": sid,
                    "equity": self.equities[sid]
                })
            else:
                self.equity_history.append({
                    "time": 0, # Placeholder or use last known
                    "strategy_id": sid,
                    "equity": self.balances[sid]
                })

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
