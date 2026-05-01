import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Any
from datetime import datetime
import logging

from core.strategy.engine import BaseStrategy, TradeSignal

logger = logging.getLogger("trading_bot.smc_strategy")

class SMCStrategy(BaseStrategy):
    """
    Smart Money Concepts (SMC) Strategy.
    Focuses on BOS, CHoCH, Order Blocks, and FVG.
    """
    def __init__(self, symbol: str, timeframe: int = 5):
        super().__init__(name="SMC", symbol=symbol, timeframe=timeframe)
        self.lookback = 100
        
    def on_tick(self, tick_data: dict, timestamp: datetime) -> Optional[TradeSignal]:
        # SMC is typically candle-based. Ticks can be used for refined entry if needed.
        return None

    def on_candle(self, df: pd.DataFrame, timestamp: datetime) -> Optional[TradeSignal]:
        if len(df) < self.lookback:
            return None

        # 1. Detect Market Structure
        # Simplified: identify fractal highs/lows
        highs = df['high'].values
        lows = df['low'].values
        closes = df['close'].values
        
        # 2. Detect FVG (Fair Value Gaps)
        fvg = self._detect_fvg(df)
        
        # 3. Detect BOS / CHoCH
        # For demonstration, we'll look for a simple BOS + FVG confluence
        last_bos = self._detect_last_bos(df)
        if not last_bos:
            return None
            
        direction = last_bos['direction']
        
        # 4. Refine Entry via Order Block or FVG
        # If we have a Bullish BOS, we look for a Bullish FVG or OB below price
        if direction == 'BUY':
            # Check for bullish FVG in the impulse move
            if fvg['bullish']:
                entry = fvg['bullish'][-1]['top']
                sl = fvg['bullish'][-1]['bottom']
                tp = entry + (entry - sl) * 2.5 # 1:2.5 RR
                
                return TradeSignal(
                    execution_id=self.get_execution_id(timestamp),
                    symbol=self.symbol,
                    direction='BUY',
                    entry=entry,
                    stop_loss=sl,
                    take_profit=tp,
                    timestamp=timestamp,
                    metadata={"reason": "Bullish BOS + FVG confluence", "fvg_count": len(fvg['bullish'])}
                )
        
        elif direction == 'SELL':
            if fvg['bearish']:
                entry = fvg['bearish'][-1]['bottom']
                sl = fvg['bearish'][-1]['top']
                tp = entry - (sl - entry) * 2.5
                
                return TradeSignal(
                    execution_id=self.get_execution_id(timestamp),
                    symbol=self.symbol,
                    direction='SELL',
                    entry=entry,
                    stop_loss=sl,
                    take_profit=tp,
                    timestamp=timestamp,
                    metadata={"reason": "Bearish BOS + FVG confluence", "fvg_count": len(fvg['bearish'])}
                )

        return None

    def _detect_fvg(self, df: pd.DataFrame) -> Dict[str, List[Dict[str, float]]]:
        """Detects Fair Value Gaps in the last 10 candles."""
        bullish_fvg = []
        bearish_fvg = []
        
        # FVG is a 3-candle pattern: [i-2, i-1, i]
        # Bullish: High[i-2] < Low[i]
        # Bearish: Low[i-2] > High[i]
        
        for i in range(len(df) - 1, len(df) - 10, -1):
            if i < 2: break
            
            c1 = df.iloc[i-2]
            c3 = df.iloc[i]
            
            # Bullish FVG
            if c3['low'] > c1['high']:
                bullish_fvg.append({
                    'bottom': c1['high'],
                    'top': c3['low'],
                    'index': i-1
                })
                
            # Bearish FVG
            if c3['high'] < c1['low']:
                bearish_fvg.append({
                    'top': c1['low'],
                    'bottom': c3['high'],
                    'index': i-1
                })
                
        return {'bullish': bullish_fvg, 'bearish': bearish_fvg}

    def _detect_last_bos(self, df: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """Detects the most recent Break of Structure."""
        # Simple implementation: check if current close broke previous local high/low
        window = 20
        local_high = df['high'].iloc[-window:-1].max()
        local_low = df['low'].iloc[-window:-1].min()
        
        if df['close'].iloc[-1] > local_high:
            return {'direction': 'BUY', 'level': local_high}
        if df['close'].iloc[-1] < local_low:
            return {'direction': 'SELL', 'level': local_low}
            
        return None
