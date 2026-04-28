import numpy as np
import logging
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List
from core.base_strategy import BaseStrategy, MarketData
from core.common.types import TradeSignal

logger = logging.getLogger("trading_bot.strategy.n_pattern_grid_v3")

class NPatternGridStrategy(BaseStrategy):
    """
    M1 N-Pattern Grid Strategy (V3: Institutional).
    Features: ATR-based grid, Session Gating, Overlap Survival Mode, and Zombie Killers.
    """

    def __init__(self, strategy_id: str, config: dict):
        super().__init__(strategy_id, config)
        # Core & Safety
        self.fixed_lot = config.get("fixed_lot", 0.05)
        self.max_grid_size = config.get("max_grid_size", 8)
        self.max_hold_bars = config.get("max_hold_bars", 45) # Reduced to 45 mins to kill zombie trades faster
        self.max_spread_pips = config.get("max_spread_pips", 0.40)
        
        # Session & Volatility
        self.trade_overlap = config.get("trade_overlap", False) # Default: Block London/NY
        self.standard_atr_multiplier = config.get("standard_atr_multiplier", 1.5)
        self.overlap_atr_multiplier = config.get("overlap_atr_multiplier", 3.0) # Double spacing if overlap trading is forced
        
        self.active_grids = {}

    def _is_big_candle(self, open_p, high, low, close, avg_body, session="GLOBAL"):
        threshold_map = {
            "TOKYO": 0.85,
            "LONDON/NY": 0.92,
            "LONDON": 0.90,
            "NEW_YORK": 0.87
        }
        
        body_ratio_thresh = next((v for k, v in threshold_map.items() if k in session), 0.87)
        body_size = abs(close - open_p)
        candle_range = high - low
        
        if candle_range == 0: return False
        
        body_ratio = body_size / candle_range
        is_large = body_size > avg_body
        
        return body_ratio > body_ratio_thresh and is_large

    def generate_signal(self, market_data: MarketData) -> Optional[TradeSignal]:
        self.last_rejection_reason = "No Pattern"
        m1 = market_data.m1_candles
        if m1 is None or len(m1) < 50: return None
        
        current_price = market_data.current_price
        current_spread = getattr(market_data, 'spread', 0.0)
        current_session = market_data.session
        
        # 1. CLEANUP & ZOMBIE EXITS
        for pid, grid in list(self.active_grids.items()):
            grid['bars_held'] += 1
            
            # Standard Exit
            if grid['direction'] == 'BUY' and (current_price >= grid['tp'] or (grid['sl'] > 0 and current_price <= grid['sl'])):
                del self.active_grids[pid]
                continue
            elif grid['direction'] == 'SELL' and (current_price <= grid['tp'] or (grid['sl'] > 0 and current_price >= grid['sl'])):
                del self.active_grids[pid]
                continue
                
            # Time-Based Exit (Kill stagnant trades)
            if grid['bars_held'] >= self.max_hold_bars:
                logger.info(f"Killing zombie grid {pid} after {self.max_hold_bars} bars.")
                del self.active_grids[pid]

        # 2. SESSION & RISK GATING
        is_friday_late = market_data.timestamp.weekday() == 4 and market_data.timestamp.hour >= 18
        if "CLOSED" in current_session or "ROLLOVER" in current_session or is_friday_late:
            self.last_rejection_reason = "Session Gated: Market Closed/Rollover"
            return None
            
        if "LONDON/NY" in current_session and not self.trade_overlap:
            self.last_rejection_reason = "Session Gated: London/NY Overlap Blocked"
            return None

        if current_spread > self.max_spread_pips:
            self.last_rejection_reason = f"Spread too high: {current_spread}"
            return None

        # 3. PATTERN DETECTION & FUSION
        avg_body = np.mean(np.abs(m1.o[-1440:-1] - m1.c[-1440:-1]))
        atr = self._calculate_atr(m1)
        
        for i in range(len(m1) - 1, len(m1) - 15, -1):
            if not self._is_big_candle(m1.o[i], m1.h[i], m1.l[i], m1.c[i], avg_body, session=current_session):
                continue
                
            is_bullish = m1.c[i] > m1.o[i]
            
            # Impulse Candle Fusion
            first_idx = i
            while first_idx > 0:
                prev = first_idx - 1
                if (m1.c[prev] > m1.o[prev]) == is_bullish and \
                   self._is_big_candle(m1.o[prev], m1.h[prev], m1.l[prev], m1.c[prev], avg_body, session=current_session):
                    first_idx = prev
                else:
                    break
            
            v_high = np.max(m1.h[first_idx:i+1])
            v_low = np.min(m1.l[first_idx:i+1])
            v_range = v_high - v_low
            
            pid = f"{'bull' if is_bullish else 'bear'}_{first_idx}_{str(m1.time[first_idx])}"
            if pid in self.active_grids: continue
            
            r_pct = max(0.50, min(0.85, self._calculate_historical_retrace(m1)))
            
            if is_bullish:
                retrace_level = v_high - (v_range * r_pct)
                history_since = m1.l[i+1:]
                
                if len(history_since) > 0 and m1.l[-1] <= retrace_level and not np.any(history_since[:-1] <= retrace_level):
                    tp = v_high 
                    self.active_grids[pid] = {
                        'direction': 'BUY', 'tp': tp, 'sl': 0.0, 'last_entry': current_price, 
                        'count': 1, 'symbol': market_data.symbol, 'bars_held': 0, 
                        'atr_at_entry': atr, 'session': current_session
                    }
                    logger.info(f"Signal: Bullish N-Fusion ({current_session}) at {current_price}, TP: {tp}")
                    return TradeSignal(direction="BUY", confidence=1.0, price=current_price, stop_loss=0.0, take_profit=tp, volume=self.fixed_lot)
            else:
                retrace_level = v_low + (v_range * r_pct)
                history_since = m1.h[i+1:]
                
                if len(history_since) > 0 and m1.h[-1] >= retrace_level and not np.any(history_since[:-1] >= retrace_level):
                    tp = v_low
                    self.active_grids[pid] = {
                        'direction': 'SELL', 'tp': tp, 'sl': 0.0, 'last_entry': current_price, 
                        'count': 1, 'symbol': market_data.symbol, 'bars_held': 0, 
                        'atr_at_entry': atr, 'session': current_session
                    }
                    logger.info(f"Signal: Bearish N-Fusion ({current_session}) at {current_price}, TP: {tp}")
                    return TradeSignal(direction="SELL", confidence=1.0, price=current_price, stop_loss=0.0, take_profit=tp, volume=self.fixed_lot)

        # 4. SESSION-AWARE ATR GRID
        for pid, grid in list(self.active_grids.items()):
            # Select multiplier based on the session the grid is currently operating in
            multiplier = self.overlap_atr_multiplier if "LONDON/NY" in current_session else self.standard_atr_multiplier
            grid_spacing = grid.get('atr_at_entry', atr) * multiplier
            
            if grid['direction'] == 'BUY' and current_price <= grid['last_entry'] - grid_spacing:
                if grid['count'] < self.max_grid_size:
                    grid['last_entry'] = current_price
                    grid['count'] += 1
                    return TradeSignal(direction="BUY", confidence=1.0, price=current_price, stop_loss=0.0, take_profit=grid['tp'], volume=self.fixed_lot)
                    
            elif grid['direction'] == 'SELL' and current_price >= grid['last_entry'] + grid_spacing:
                if grid['count'] < self.max_grid_size:
                    grid['last_entry'] = current_price
                    grid['count'] += 1
                    return TradeSignal(direction="SELL", confidence=1.0, price=current_price, stop_loss=0.0, take_profit=grid['tp'], volume=self.fixed_lot)

        return None

    def _calculate_historical_retrace(self, m1) -> float:
        try:
            depths = []
            lookback = min(300, len(m1))
            avg_body = np.mean(np.abs(m1.o[-50:-1] - m1.c[-50:-1]))
            
            for i in range(len(m1) - 50, len(m1) - lookback, -1):
                body = abs(m1.c[i] - m1.o[i])
                if body < avg_body * 1.5: continue
                
                is_bull = m1.c[i] > m1.o[i]
                v_high, v_low = m1.h[i], m1.l[i]
                v_range = v_high - v_low
                
                future = m1[i+1:i+21]
                if len(future) == 0: continue
                
                if is_bull:
                    retrace_depth = (v_high - np.min(future.l)) / v_range if v_range > 0 else 0
                else:
                    retrace_depth = (np.max(future.h) - v_low) / v_range if v_range > 0 else 0
                    
                if 0.2 < retrace_depth < 1.1:
                    depths.append(retrace_depth)
                        
            if depths: return float(np.mean(depths))
        except Exception as e:
            logger.debug(f"Retrace Analyzer failed: {e}")
            
        return 0.618

    def _calculate_atr(self, m1, period=14) -> float:
        try:
            highs, lows = m1.h[-period-1:-1], m1.l[-period-1:-1]
            closes = m1.c[-period-2:-2]
            tr = np.maximum(highs - lows, np.maximum(np.abs(highs - closes), np.abs(lows - closes)))
            return float(np.mean(tr))
        except:
            return 0.50

    def get_metrics(self, market_data: MarketData) -> Dict[str, Any]:
        """Required by BaseStrategy for dashboard reporting."""
        return {
            "grids": len(self.active_grids), 
            "pos": sum(g['count'] for g in self.active_grids.values()),
            "bars": round(float(np.mean([g['bars_held'] for g in self.active_grids.values()])), 1) if self.active_grids else 0.0
        }

    def get_thresholds(self) -> Dict[str, Any]:
        """Overrides BaseStrategy to show N-Pattern specific limits on dashboard."""
        return {
            "grids": "-",
            "pos": self.max_grid_size,
            "bars": self.max_hold_bars
        }

    def on_trade_closed(self, trade_record: dict) -> None:
        """Required by BaseStrategy for post-trade processing."""
        pass