import os
import logging
import time
import bisect
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any, Optional
from tqdm import tqdm

from .trailing_stop import TrailingStopManager
from .lot_calculator import LotCalculator
from .risk_manager import RiskManager
from .performance import PerformanceMetrics

logger = logging.getLogger("trading_bot.backtester")

class _OpenTrade:
    def __init__(self, ticket: int, signal: Any, entry_price: float, lot: float, sl: float, tp: float, 
                 entry_time: datetime, session: str, tick_size: float, tick_value: float):
        self.ticket = ticket
        self.signal = signal
        self.direction = signal.direction
        self.entry_price = entry_price
        self.lot = lot
        self.sl = sl
        self.tp = tp
        self.entry_time = entry_time
        self.session = session
        self.best_price = entry_price
        self.partial_closed_count = 0
        self.tick_size = tick_size
        self.tick_value = tick_value
        self.total_commission_paid = 0.0

class BacktestEngine:
    """Professional-grade historical simulation engine with Anti-Lookahead protection."""
    
    def __init__(self, config: dict, strategy: Any):
        self.config = config
        self.strategy = strategy
        self.initial_balance = config.get("backtest", {}).get("initial_balance", 1000.0)
        self.balance = self.initial_balance
        self.equity = self.initial_balance
        self.trades = []
        self.risk_manager = RiskManager(config)
        self.risk_manager.silent = True
        self.utc_offset = config.get("backtest", {}).get("utc_offset", 0)
        
        # Results
        self.results_dir = "backtest_results"
        if not os.path.exists(self.results_dir): os.makedirs(self.results_dir)

    def run(self, symbol: str, htf_candles: Any, m15_candles: Any, m5_candles: Any, 
            d1_candles: Any, ticks: Optional[List[dict]] = None, quiet: bool = False):
        """
        Executes the backtest simulation.
        N-1 analyze / N entry logic to prevent Look-Ahead Bias.
        """
        self.strategy.silent = True
        self.balance = self.initial_balance
        
        symbol_cfg = self.config.get("symbols_config", {}).get(symbol, {})
        point = symbol_cfg.get("point", 0.01)
        contract_size = symbol_cfg.get("contract_size", 100)
        
        open_trade: Optional[_OpenTrade] = None
        pending_signal: Optional[Any] = None
        
        self.risk_manager.reset_daily_stats(self.balance)
        last_date = None
        daily_trades = 0; consecutive_losses = 0; notified_halts = set()
        
        m5_times = m5_candles.time
        m5_closes = m5_candles.close
        m5_highs = m5_arr_highs = m5_candles.high
        m5_lows = m5_arr_lows = m5_candles.low
        m5_opens = m5_candles.open
        
        # ATR Precomputation
        m5_tr = np.zeros_like(m5_closes)
        m5_tr[1:] = np.maximum(m5_highs[1:] - m5_lows[1:], np.maximum(np.abs(m5_highs[1:] - m5_closes[:-1]), np.abs(m5_lows[1:] - m5_closes[:-1])))
        m5_atr_series = pd.Series(m5_tr).rolling(14).mean().values
        
        pre_ctx = self.strategy.preprocess_history(htf_candles, m15_candles, m5_candles, m5_candles)
        m5_meta = pre_ctx.get("m5", [])
        
        trades_list = []
        pbar = tqdm(total=len(m5_times), desc=f"BT:{symbol}", disable=quiet)
        
        for i in range(50, len(m5_times)):
            pbar.update(1)
            t = m5_times[i]
            candle_dt = datetime.fromtimestamp(t, tz=timezone.utc)
            
            # --- Daily Boundary ---
            if last_date != candle_dt.date():
                self.risk_manager.record_daily_close(self.balance)
                self.risk_manager.reset_daily_stats(self.balance)
                daily_trades = 0; consecutive_losses = 0; last_date = candle_dt.date()

            session = self.strategy.get_session_from_hour(candle_dt.hour, self.utc_offset)
            is_enabled = self.config.get("session_config", {}).get(session, {}).get("enabled", True)
            spread = self.config.get("backtest", {}).get("session_spreads", {}).get(session, 1.5) * point

            # 1. Handle Pending Signals (Anti-Lookahead Entry at current Open)
            if pending_signal and not open_trade:
                entry_price = m5_opens[i] + (spread if pending_signal.direction == "BUY" else -spread)
                
                # Check Circuit Breakers
                allowed, _ = self.risk_manager.check_circuit_breakers(self.balance, self.balance, daily_trades, 0, consecutive_losses)
                if allowed:
                    risk_pct = self.risk_manager.calculate_scaled_risk(self.balance, session=session)
                    risk_val = self.balance * (risk_pct / 100.0)
                    sl_dist = abs(entry_price - pending_signal.stop_loss)
                    lot = LotCalculator.calculate(risk_val, sl_dist, point, symbol_cfg.get("tick_value", 1.0), 
                                                   volume_min=symbol_cfg.get("min_lot", 0.01))
                    
                    open_trade = _OpenTrade(int(t), pending_signal, entry_price, lot, pending_signal.stop_loss, pending_signal.take_profit, 
                                            candle_dt, session, point, symbol_cfg.get("tick_value", 1.0))
                    daily_trades += 1
                pending_signal = None

            # 2. Manage Open Positions
            if open_trade:
                bid_l, bid_h = m5_lows[i], m5_highs[i]
                ask_l, ask_h = bid_l + spread, bid_h + spread
                closed = False; result = "SL"; exit_p = open_trade.sl
                
                if open_trade.direction == "BUY":
                    # Partial TP Check
                    if open_trade.partial_closed_count == 0 and bid_h >= open_trade.signal.tp1_price:
                        p_lot = round(open_trade.lot * 0.25, 2)
                        p_pnl = ((open_trade.signal.tp1_price - open_trade.entry_price) / open_trade.tick_size) * open_trade.tick_value * p_lot
                        self.balance += p_pnl; open_trade.lot -= p_lot; open_trade.partial_closed_count += 1

                    if bid_l <= open_trade.sl: closed = True
                    elif bid_h >= open_trade.tp: exit_p, result, closed = open_trade.tp, "TP", True
                else:
                    if open_trade.partial_closed_count == 0 and ask_l <= open_trade.signal.tp1_price:
                        p_lot = round(open_trade.lot * 0.25, 2)
                        p_pnl = ((open_trade.entry_price - open_trade.signal.tp1_price) / open_trade.tick_size) * open_trade.tick_value * p_lot
                        self.balance += p_pnl; open_trade.lot -= p_lot; open_trade.partial_closed_count += 1

                    if ask_h >= open_trade.sl: closed = True
                    elif ask_l <= open_trade.tp: exit_p, result, closed = open_trade.tp, "TP", True
                
                if closed:
                    pnl = ((exit_p - open_trade.entry_price) / open_trade.tick_size) * open_trade.tick_value * open_trade.lot if open_trade.direction == "BUY" \
                          else ((open_trade.entry_price - exit_p) / open_trade.tick_size) * open_trade.tick_value * open_trade.lot
                    self.balance += pnl
                    trade_record = {"ticket": open_trade.ticket, "time": open_trade.entry_time, "exit_time": candle_dt, "direction": open_trade.direction,
                                    "entry": open_trade.entry_price, "exit": exit_p, "lot": open_trade.lot, "pnl": round(pnl, 2), "result": result, "session": open_trade.session}
                    trades_list.append(trade_record)
                    self.risk_manager.update_history(trade_record)
                    if pnl < 0: consecutive_losses += 1
                    else: consecutive_losses = 0
                    open_trade = None
                else:
                    # Update Trailing Stop
                    open_trade.best_price = max(open_trade.best_price, bid_h) if open_trade.direction == "BUY" else min(open_trade.best_price, ask_l)
                    risk = abs(open_trade.entry_price - open_trade.signal.stop_loss)
                    new_sl = TrailingStopManager.calculate_new_sl(open_trade.direction == "BUY", open_trade.entry_price, open_trade.sl, open_trade.best_price, 
                                                                  m5_atr_series[i], risk, self.config, {"low": m5_lows[i-1], "high": m5_highs[i-1]})
                    if new_sl: open_trade.sl = new_sl

            # 3. Analyze for new signals (Available for entry at NEXT candle O)
            if not open_trade and not pending_signal and is_enabled:
                htf_idx, m15_idx = np.searchsorted(htf_candles.time, t, side='right')-1, np.searchsorted(m15_candles.time, t, side='right')-1
                signal, _, _ = self.strategy.analyze(symbol, htf_candles[:htf_idx+1], m15_candles[:m15_idx+1], m5_candles[:i+1], m5_closes[i], session=session, preprocessed=m5_meta[i])
                if signal: pending_signal = signal

            pbar.set_postfix({"bal": f"${self.balance:.0f}", "trades": len(trades_list)})
            
        pbar.close()
        return self._finalize_results(symbol, trades_list)

    def _finalize_results(self, symbol: str, trades: List[dict]) -> dict:
        metrics = PerformanceMetrics.calculate_metrics(trades, self.initial_balance)
        metrics["trades"] = trades # Add trades for CLI display
        filename = f"{symbol}_trades_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        pd.DataFrame(trades).to_csv(os.path.join(self.results_dir, filename), index=False)
        return metrics
