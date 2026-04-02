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

logger = logging.getLogger("trading_bot.backtester")

class _OpenTrade:
    def __init__(self, ticket: int, signal: Any, entry_price: float, lot: float, sl: float, tp: float, 
                 entry_time: datetime, session: str):
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
        self.total_commission_paid = 0.0

class BacktestEngine:
    """High-fidelity backtesting engine for Price Action Scalping."""
    
    def __init__(self, config: dict, strategy: Any):
        self.config = config
        self.strategy = strategy
        self.initial_balance = config.get("backtest", {}).get("initial_balance", 1000.0)
        self.balance = self.initial_balance
        self.equity = self.initial_balance
        self.trades = []
        
        # Performance logging (CSV)
        self.results_dir = "backtest_results"
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)

    @staticmethod
    def _find_slice_index(times: np.ndarray, time_threshold: float) -> int:
        return np.searchsorted(times, time_threshold, side='right') - 1

    def run(self, symbol: str, h1_candles: Any, m15_candles: Any, m5_candles: Any, 
            d1_candles: Any, ticks: Optional[List[dict]] = None, 
            quiet: bool = False, tick_entry: bool = False):
        """
        M5 Sniper Backtest Loop.
        """
        self.strategy.silent = True
        self.balance = self.initial_balance
        
        symbol_cfg = self.config.get("symbols_config", {}).get(symbol, {})
        point = symbol_cfg.get("point", 0.01)
        contract_size = symbol_cfg.get("contract_size", 100)
        lot_step = symbol_cfg.get("lot_step", 0.01)
        min_lot = symbol_cfg.get("min_lot", 0.01)
        
        utc_offset = self.config.get("backtest", {}).get("utc_offset", 0)
        open_trade: Optional[_OpenTrade] = None
        
        # M5 is now the primary loop timeframe
        m5_arr = m5_candles
        m5_times = m5_arr.time
        m5_closes = m5_arr.close
        m5_highs = m5_arr.high
        m5_lows = m5_arr.low
        
        logger.info(f"BacktestEngine.run (M5 Sniper) started for {symbol}. M5 candles: {len(m5_times)}")
        
        # ATR Calculation for M5
        m5_tr = np.zeros_like(m5_closes)
        if len(m5_closes) > 1:
            m5_tr[1:] = np.maximum(m5_highs[1:] - m5_lows[1:], 
                            np.maximum(np.abs(m5_highs[1:] - m5_closes[:-1]), 
                                       np.abs(m5_lows[1:] - m5_closes[:-1])))
        m5_atr_series = np.zeros_like(m5_closes)
        if len(m5_tr) > 14:
            m5_atr_series[14] = np.mean(m5_tr[1:15])
            for x in range(15, len(m5_tr)):
                m5_atr_series[x] = (m5_atr_series[x-1] * 13 + m5_tr[x]) / 14
        else:
            m5_atr_series[:] = 0.1
            
        m5_avg_atr = pd.Series(m5_atr_series).rolling(window=200, min_periods=1).mean().values
        
        # --- Structured Ticks for Exit Evaluation ---
        tick_times = tick_arr = None
        if ticks:
            tick_dtype = [('time', 'i8'), ('bid', 'f8'), ('ask', 'f8')]
            tick_arr = np.array([(t['time'], t['bid'], t['ask']) for t in ticks], dtype=tick_dtype)
            tick_times = tick_arr['time']

        # --- Vectorized Preprocessing (H1/M15/M5 Hierarchy) ---
        pre_ctx = self.strategy.preprocess_history(h1_candles, m15_candles, m5_arr, m5_arr)
        m5_meta = pre_ctx.get("m5", [])
        
        trades_list = []
        pbar = tqdm(total=len(m5_times), desc=f"BT:{symbol}", disable=quiet)
        
        for i in range(50, len(m5_times)):
            pbar.update(1)
            t = m5_times[i]
            candle_dt = datetime.fromtimestamp(t, tz=timezone.utc)
            session = self.strategy.get_session_from_hour(candle_dt.hour)
            
            # --- Dynamic Session Spread (P1 Fix) ---
            # Tokyo: 3.5, London: 1.8, NY: 1.2 points
            spread_map = {"TOKYO": 3.5, "LONDON": 1.8, "LONDON/NY": 1.4, "NEW_YORK": 1.2}
            current_spread_pts = spread_map.get(session, 1.5)
            spread = current_spread_pts * point

            # 1. Trade Management (SL/TP Exits & Trailing Stops)
            if open_trade:
                bid_l, bid_h = m5_lows[i], m5_highs[i]
                ask_l, ask_h = bid_l + spread, bid_h + spread
                
                # Check for Exits on current M5 candle
                # BUY closes at BID (bid_l, bid_h)
                # SELL closes at ASK (ask_l, ask_h)
                closed = False
                if open_trade.direction == "BUY":
                    if bid_l <= open_trade.sl: exit_price, result, closed = open_trade.sl, "SL", True
                    elif bid_h >= open_trade.tp: exit_price, result, closed = open_trade.tp, "TP", True
                else:
                    if ask_h >= open_trade.sl: exit_price, result, closed = open_trade.sl, "SL", True
                    elif ask_l <= open_trade.tp: exit_price, result, closed = open_trade.tp, "TP", True
                
                if closed:
                    # Apply slippage on exit too? Usually SL/TP have some slippage.
                    # For purity, we just use the hit price level.
                    pnl = (exit_price - open_trade.entry_price) * (1 if open_trade.direction == "BUY" else -1) * contract_size * open_trade.lot
                    self.balance += pnl
                    trades_list.append({
                        "time": open_trade.entry_time, "exit_time": candle_dt, "direction": open_trade.direction,
                        "entry": open_trade.entry_price, "exit": exit_price, "lot": open_trade.lot,
                        "pnl": round(pnl, 2), "result": result, "session": open_trade.session
                    })
                    open_trade = None
                else:
                    # Update best price for trailing
                    if open_trade.direction == "BUY":
                        if bid_h > open_trade.best_price: open_trade.best_price = bid_h
                    else:
                        if ask_l < open_trade.best_price: open_trade.best_price = ask_l
                        
                    # Trailing Stop Toggle
                    if not self.config.get("trailing_stop_enabled", True):
                        continue
                        
                    # Update Trailing Stops (Synced with Live Logic)
                    risk = abs(open_trade.entry_price - open_trade.signal.stop_loss)
                    last_c = {"low": m5_lows[i-1], "high": m5_highs[i-1]}
                    
                    new_sl = TrailingStopManager.calculate_new_sl(
                        open_trade.direction == "BUY",
                        open_trade.entry_price,
                        open_trade.sl,
                        open_trade.best_price,
                        m5_atr_series[i],
                        risk,
                        last_c
                    )
                    
                    if new_sl:
                        open_trade.sl = new_sl

            # 2. Strategy Logic (Only if no open trade)
            if not open_trade:
                # Slice candles for the strategy (emulate live window)
                h1_slice = h1_candles; m15_slice = m15_candles; m5_slice = m5_arr[:i+1]
                
                signal, bias, _ = self.strategy.analyze(symbol, h1_slice, m15_slice, m5_slice, m5_closes[i], 
                                                        session=session, preprocessed=m5_meta[i])
                
                if signal:
                    # Risk calculation (simplified for BT)
                    risk_pct = 1.0; risk_val = self.balance * (risk_pct / 100.0)
                    sl_dist = abs(signal.entry_price - signal.stop_loss)
                    lot = LotCalculator.calculate(risk_val, sl_dist, point, 1.0, min_lot)
                    
                    # --- Slippage Simulation (P1 Fix) ---
                    # Add 1.5pts slippage to entry price
                    slippage = 1.5 * point
                    entry_price = signal.entry_price + (slippage if signal.direction == "BUY" else -slippage)
                    
                    open_trade = _OpenTrade(int(t), signal, entry_price, lot, signal.stop_loss, signal.take_profit, candle_dt, session)
            
            pbar.set_postfix({"balance": f"${self.balance:.0f}", "trade": len(trades_list)})
            
        pbar.close()
        return self._finalize_results(symbol, trades_list)

    def _finalize_results(self, symbol: str, trades: List[dict]) -> dict:
        df = pd.DataFrame(trades)
        net_profit = df['pnl'].sum() if not df.empty else 0
        win_rate = (len(df[df['pnl'] > 0]) / len(df) * 100) if not df.empty else 0
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{symbol}_trades_{timestamp}.csv"
        df.to_csv(os.path.join(self.results_dir, filename), index=False)
        
        return {
            "net_profit": net_profit, "final_balance": self.initial_balance + net_profit,
            "win_rate": win_rate, "total_trades": len(df), "trades": trades
        }
