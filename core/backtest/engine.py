import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timezone
import random

from core.strategy.engine import BaseStrategy, TradeSignal
from core.time.time_service import time_service

logger = logging.getLogger("trading_bot.backtest_engine")

class BacktestPosition:
    def __init__(self, signal: TradeSignal, entry_price: float, volume: float, commission: float = 0.0):
        self.signal = signal
        self.entry_price = entry_price
        self.entry_time = signal.timestamp
        self.volume = volume
        self.commission = commission
        
        self.sl = signal.stop_loss
        self.tp = signal.take_profit
        self.direction = signal.direction
        
        self.exit_price = 0.0
        self.exit_time = None
        self.pnl = 0.0
        self.is_closed = False

    def update(self, high: float, low: float, timestamp: datetime, bid: float, ask: float, spread: float):
        if self.is_closed:
            return

        if self.direction == 'BUY':
            if low <= self.sl:
                self._close(self.sl, timestamp)
            elif high >= self.tp:
                self._close(self.tp, timestamp)
        else:
            # SELL: Exit at ASK (Bid + Spread)
            if (high + spread) >= self.sl:
                self._close(self.sl, timestamp)
            elif (low + spread) <= self.tp:
                self._close(self.tp, timestamp)

    def _close(self, price: float, timestamp: datetime):
        self.exit_price = price
        self.exit_time = timestamp
        multiplier = 100000 
        if self.direction == 'BUY':
            self.pnl = (self.exit_price - self.entry_price) * self.volume * multiplier - self.commission
        else:
            self.pnl = (self.entry_price - self.exit_price) * self.volume * multiplier - self.commission
        self.is_closed = True

class BacktestEngine:
    def __init__(self, strategies: List[BaseStrategy], initial_balance: float = 10000.0):
        self.strategies = strategies
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.equity_curve = [initial_balance]
        self.history: List[BacktestPosition] = []
        self.active_positions: List[BacktestPosition] = []
        self.commission_per_lot = 7.0

    def run(self, data: Dict[str, pd.DataFrame], spread_multiplier: float = 1.0):
        """
        data: Dict mapping symbol -> DataFrame
        spread_multiplier: for Stress Testing
        """
        logger.info(f"BacktestEngine: Replaying {len(self.strategies)} clusters...")
        
        # 1. Align data (find common time range)
        # Simplified: we assume data is already aligned for this demo
        main_symbol = self.strategies[0].symbol
        df = data[main_symbol]
        
        point = 0.00001
        base_spread = 15 * point * spread_multiplier
        
        for i in range(len(df)):
            ts = df.index[i]
            bar = df.iloc[i]
            
            # Stress Test: Randomize spread slightly
            current_spread = base_spread * random.uniform(0.9, 1.3)
            
            # 2. Update existing positions
            for pos in self.active_positions:
                pos.update(bar['high'], bar['low'], ts, bar['close'], bar['close'] + current_spread, current_spread)
                if pos.is_closed:
                    self.balance += pos.pnl
                    self.history.append(pos)
            
            self.active_positions = [p for p in self.active_positions if not p.is_closed]

            # 3. Process each strategy
            for strategy in self.strategies:
                df_snapshot = df.iloc[:i+1]
                signal = strategy.on_candle(df_snapshot, ts)
                
                if signal:
                    # Apply Stress Test to entry price (slippage)
                    slippage = 2 * point * spread_multiplier if random.random() > 0.8 else 0
                    
                    volume = 0.1
                    entry_price = (bar['close'] + current_spread) if signal.direction == 'BUY' else bar['close']
                    entry_price += slippage if signal.direction == 'BUY' else -slippage
                    
                    commission = volume * self.commission_per_lot
                    pos = BacktestPosition(signal, entry_price, volume, commission)
                    self.active_positions.append(pos)

            self.equity_curve.append(self.balance)

        return self.get_report()

    def get_report(self) -> Dict[str, Any]:
        if not self.history:
            return {"status": "NO_TRADES"}

        pnls = [p.pnl for p in self.history]
        net_profit = sum(pnls)
        win_rate = len([p for p in pnls if p > 0]) / len(pnls)
        
        # Drawdown calculation
        peak = self.initial_balance
        max_dd = 0
        for val in self.equity_curve:
            if val > peak: peak = val
            dd = (peak - val) / peak
            if dd > max_dd: max_dd = dd

        return {
            "summary": {
                "initial_balance": self.initial_balance,
                "final_balance": self.balance,
                "net_profit": net_profit,
                "trades": len(self.history),
                "win_rate": win_rate,
                "max_drawdown": max_dd,
                "profit_factor": sum([p for p in pnls if p > 0]) / abs(sum([p for p in pnls if p < 0])) if any(p < 0 for p in pnls) else 99
            },
            "equity_curve": self.equity_curve,
            "monte_carlo": self.run_monte_carlo(pnls)
        }

    def run_monte_carlo(self, pnls: List[float], iterations: int = 50) -> Dict[str, Any]:
        """
        Shuffles trade order to test drawdown resilience.
        """
        mc_drawdowns = []
        for _ in range(iterations):
            shuffled = list(pnls)
            random.shuffle(shuffled)
            
            temp_balance = self.initial_balance
            temp_peak = temp_balance
            temp_max_dd = 0
            
            for pnl in shuffled:
                temp_balance += pnl
                if temp_balance > temp_peak: temp_peak = temp_balance
                dd = (temp_peak - temp_balance) / temp_peak
                if dd > temp_max_dd: temp_max_dd = dd
            mc_drawdowns.append(temp_max_dd)

        return {
            "avg_drawdown": np.mean(mc_drawdowns),
            "max_drawdown_95pc": np.percentile(mc_drawdowns, 95),
            "institutional_pass": np.percentile(mc_drawdowns, 95) < 0.15 # Pass if 95% of runs stay below 15% DD
        }
