"""
TRADING BOT V5 - HYBRID BREAKOUT STRATEGY
Complete Rewrite - Different Approach

Strategy: Breakout + Pullback with Strong Trend
1. H4 EMA Trend Direction (Primary)
2. M30 Breakout Confirmation 
3. M15 Pullback Entry
4. Session Filter (London/NY only)
5. Fixed 1:2 R:R with proper SL placement

Target: 70%+ Win Rate, 15+ Trades
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class TradeSignal:
    """Trade signal data class."""
    direction: str
    entry_price: float
    stop_loss: float
    take_profit: float
    confidence: float
    confluence_score: int
    reasons: List[str] = field(default_factory=list)
    rejection_type: str = ""
    order_block: Optional[dict] = None
    fvg: Optional[dict] = None
    liquidity: Optional[dict] = None
    timestamp: datetime = None
    rr_ratio: float = 2.0


class StrategyEngine:
    """
    HYBRID BREAKOUT STRATEGY V5
    
    Different approach - breakout + pullback instead of rejection patterns
    """

    def __init__(self, config: dict, analysis_logger=None):
        self.config = config
        self.strategy_config = config.get('strategy', {})
        self.logger = analysis_logger

        # Settings
        self.min_confluence_score = self.strategy_config.get('min_confluence_score', 4)
        self.min_confidence = self.strategy_config.get('min_confidence', 50)
        
        # Track breakouts for pullback entries
        self.recent_breakouts = {}

        self._log("=" * 50)
        self._log("HYBRID BREAKOUT STRATEGY V5")
        self._log(f"Min Confluence: {self.min_confluence_score}")
        self._log(f"Min Confidence: {self.min_confidence}%")
        self._log("=" * 50)

    def _log(self, message: str, level: str = "INFO"):
        if self.logger:
            self.logger.log(message, level)
        logger.info(message)

    def analyze(self, symbol: str, h4_candles: List[dict], m30_candles: List[dict],
                m15_candles: List[dict], current_price: float) -> Optional[TradeSignal]:
        """
        Main Analysis - Breakout + Pullback Strategy
        """
        
        if len(h4_candles) < 50 or len(m30_candles) < 100 or len(m15_candles) < 100:
            return None

        # ==========================================
        # STEP 1: H4 TREND (Simple & Clear)
        # ==========================================
        h4_trend, h4_strength = self._get_h4_trend(h4_candles)
        
        if h4_trend == 'RANGING' or h4_strength < 60:
            self._log(f"H4: {h4_trend} ({h4_strength}%) - SKIP")
            return None
        
        self._log(f"H4: {h4_trend} ({h4_strength}%)")

        # ==========================================
        # STEP 2: M30 STRUCTURE CONFIRMATION
        # ==========================================
        m30_trend, m30_structure = self._get_m30_structure(m30_candles, h4_trend)
        
        if not m30_structure:
            self._log("M30 structure NOT aligned")
            return None
        
        self._log(f"M30: {m30_trend} - Structure OK")

        # ==========================================
        # STEP 3: FIND ENTRY SIGNAL
        # ==========================================
        # Method A: Breakout Entry
        breakout_signal = self._check_breakout_entry(m30_candles, m15_candles, h4_trend, current_price)
        
        # Method B: Pullback Entry
        pullback_signal = self._check_pullback_entry(m30_candles, m15_candles, h4_trend, current_price)
        
        # Choose best signal
        signal = breakout_signal or pullback_signal
        
        if not signal:
            self._log("No valid entry signal")
            return None

        # ==========================================
        # STEP 4: CALCULATE SL/TP (1:2 R:R)
        # ==========================================
        atr = self._calculate_atr(m30_candles)
        signal = self._set_sl_tp(signal, atr, m30_candles)

        # ==========================================
        # STEP 5: FINAL CONFIDENCE CHECK
        # ==========================================
        confluence, reasons = self._calculate_confluence(
            h4_trend, h4_strength, m30_trend, signal, m30_candles
        )
        
        signal.confluence_score = confluence
        signal.reasons = reasons
        
        if confluence < self.min_confluence_score:
            self._log(f"Confluence too low: {confluence}")
            return None

        confidence = self._calculate_confidence(confluence, h4_strength, signal)
        signal.confidence = confidence
        
        if confidence < self.min_confidence:
            self._log(f"Confidence too low: {confidence}%")
            return None

        self._log(f"✓ SIGNAL: {signal.direction} @ {signal.entry_price:.5f}")
        self._log(f"  SL: {signal.stop_loss:.5f} | TP: {signal.take_profit:.5f}")
        self._log(f"  R:R: {signal.rr_ratio:.2f} | Conf: {confidence:.0f}%")
        
        return signal

    def _get_h4_trend(self, candles: List[dict]) -> Tuple[str, int]:
        """
        Simple H4 trend using EMA slope and position
        Returns: (trend, strength%)
        """
        if len(candles) < 50:
            return 'RANGING', 0

        closes = np.array([c['close'] for c in candles])
        
        # EMAs
        ema20 = self._ema(closes, 20)
        ema50 = self._ema(closes, 50)
        
        price = closes[-1]
        
        # EMA Position
        above_20 = price > ema20[-1]
        above_50 = price > ema50[-1]
        below_20 = price < ema20[-1]
        below_50 = price < ema50[-1]
        
        # EMA Slope (comparing last 5 candles)
        ema20_slope = (ema20[-1] - ema20[-5]) / ema20[-5] * 100 if ema20[-5] != 0 else 0
        ema50_slope = (ema50[-1] - ema50[-5]) / ema50[-5] * 100 if ema50[-5] != 0 else 0
        
        # Bullish conditions
        bull_score = 0
        if above_20 and above_50:
            bull_score += 30
        if ema20_slope > 0.1:
            bull_score += 25
        if ema50_slope > 0.05:
            bull_score += 15
        if ema20[-1] > ema50[-1]:
            bull_score += 20
        # Higher highs and higher lows
        highs = [c['high'] for c in candles[-10:]]
        lows = [c['low'] for c in candles[-10:]]
        if highs[-1] > highs[-3] and lows[-1] > lows[-3]:
            bull_score += 10
        
        # Bearish conditions
        bear_score = 0
        if below_20 and below_50:
            bear_score += 30
        if ema20_slope < -0.1:
            bear_score += 25
        if ema50_slope < -0.05:
            bear_score += 15
        if ema20[-1] < ema50[-1]:
            bear_score += 20
        # Lower highs and lower lows
        if highs[-1] < highs[-3] and lows[-1] < lows[-3]:
            bear_score += 10
        
        # Decision
        if bull_score >= 60 and bull_score > bear_score + 15:
            return 'BULLISH', bull_score
        if bear_score >= 60 and bear_score > bull_score + 15:
            return 'BEARISH', bear_score
        
        return 'RANGING', max(bull_score, bear_score)

    def _get_m30_structure(self, candles: List[dict], h4_trend: str) -> Tuple[str, bool]:
        """
        Check M30 structure aligns with H4
        """
        if len(candles) < 50:
            return 'RANGING', False
        
        # Recent swing points
        highs = [c['high'] for c in candles[-20:]]
        lows = [c['low'] for c in candles[-20:]]
        
        # Find recent swing high and low
        swing_high = max(highs[-10:])
        swing_low = min(lows[-10:])
        
        current_price = candles[-1]['close']
        
        # For bullish: price should be above recent swing low, making higher lows
        if h4_trend == 'BULLISH':
            # Check for higher lows
            recent_lows = lows[-15:]
            hl_count = sum(1 for i in range(1, len(recent_lows)) if recent_lows[i] >= recent_lows[i-1] * 0.999)
            if hl_count >= 8:
                return 'BULLISH', True
            # Or price above midpoint
            midpoint = (swing_high + swing_low) / 2
            if current_price > midpoint:
                return 'BULLISH', True
        
        # For bearish: price should be below recent swing high, making lower highs
        if h4_trend == 'BEARISH':
            recent_highs = highs[-15:]
            lh_count = sum(1 for i in range(1, len(recent_highs)) if recent_highs[i] <= recent_highs[i-1] * 1.001)
            if lh_count >= 8:
                return 'BEARISH', True
            # Or price below midpoint
            midpoint = (swing_high + swing_low) / 2
            if current_price < midpoint:
                return 'BEARISH', True
        
        return 'RANGING', False

    def _check_breakout_entry(self, m30_candles: List[dict], m15_candles: List[dict],
                               trend: str, current_price: float) -> Optional[TradeSignal]:
        """
        Check for breakout entry
        Entry on breakout of recent swing high/low
        """
        # Find key level
        lookback = 20
        recent = m30_candles[-lookback:]
        
        if trend == 'BULLISH':
            # Find resistance to break
            resistance = max(c['high'] for c in recent[:-1])
            
            # Check if current candle breaks above
            if current_price > resistance:
                # Check M15 for confirmation (strong close above)
                m15_recent = m15_candles[-5:]
                strong_close = all(c['close'] > resistance for c in m15_recent[-2:])
                
                if strong_close:
                    return TradeSignal(
                        direction='BUY',
                        entry_price=current_price,
                        stop_loss=0,
                        take_profit=0,
                        confidence=0,
                        confluence_score=0,
                        rejection_type='BREAKOUT',
                        timestamp=datetime.now()
                    )
        
        elif trend == 'BEARISH':
            # Find support to break
            support = min(c['low'] for c in recent[:-1])
            
            # Check if current candle breaks below
            if current_price < support:
                m15_recent = m15_candles[-5:]
                strong_close = all(c['close'] < support for c in m15_recent[-2:])
                
                if strong_close:
                    return TradeSignal(
                        direction='SELL',
                        entry_price=current_price,
                        stop_loss=0,
                        take_profit=0,
                        confidence=0,
                        confluence_score=0,
                        rejection_type='BREAKOUT',
                        timestamp=datetime.now()
                    )
        
        return None

    def _check_pullback_entry(self, m30_candles: List[dict], m15_candles: List[dict],
                               trend: str, current_price: float) -> Optional[TradeSignal]:
        """
        Check for pullback entry
        Entry when price pulls back to EMA or key level
        """
        closes = np.array([c['close'] for c in m30_candles])
        ema20 = self._ema(closes, 20)
        
        ema_value = ema20[-1]
        
        if trend == 'BULLISH':
            # Price pulls back to EMA20 from above
            if current_price > ema_value:
                distance_pct = (current_price - ema_value) / current_price * 100
                
                # Within 0.5% of EMA (pullback zone)
                if distance_pct < 0.5:
                    # Check M15 for bullish candle forming
                    m15_recent = m15_candles[-3:]
                    bullish_candles = sum(1 for c in m15_recent if c['close'] > c['open'])
                    
                    if bullish_candles >= 2:
                        return TradeSignal(
                            direction='BUY',
                            entry_price=current_price,
                            stop_loss=0,
                            take_profit=0,
                            confidence=0,
                            confluence_score=0,
                            rejection_type='PULLBACK_EMA',
                            timestamp=datetime.now()
                        )
        
        elif trend == 'BEARISH':
            # Price pulls back to EMA20 from below
            if current_price < ema_value:
                distance_pct = (ema_value - current_price) / current_price * 100
                
                if distance_pct < 0.5:
                    m15_recent = m15_candles[-3:]
                    bearish_candles = sum(1 for c in m15_recent if c['close'] < c['open'])
                    
                    if bearish_candles >= 2:
                        return TradeSignal(
                            direction='SELL',
                            entry_price=current_price,
                            stop_loss=0,
                            take_profit=0,
                            confidence=0,
                            confluence_score=0,
                            rejection_type='PULLBACK_EMA',
                            timestamp=datetime.now()
                        )
        
        return None

    def _set_sl_tp(self, signal: TradeSignal, atr: float, m30_candles: List[dict]) -> TradeSignal:
        """
        Set Stop Loss and Take Profit with 1:2 R:R
        SL placed at recent swing low/high
        """
        recent = m30_candles[-20:]
        
        if signal.direction == 'BUY':
            # SL at recent swing low - 1 ATR
            swing_low = min(c['low'] for c in recent)
            signal.stop_loss = swing_low - atr * 0.5
            
            # Calculate risk
            risk = signal.entry_price - signal.stop_loss
            
            # TP at 2x risk
            signal.take_profit = signal.entry_price + (risk * 2)
            
        else:  # SELL
            # SL at recent swing high + 1 ATR
            swing_high = max(c['high'] for c in recent)
            signal.stop_loss = swing_high + atr * 0.5
            
            # Calculate risk
            risk = signal.stop_loss - signal.entry_price
            
            # TP at 2x risk
            signal.take_profit = signal.entry_price - (risk * 2)
        
        # Calculate actual R:R
        actual_risk = abs(signal.entry_price - signal.stop_loss)
        actual_reward = abs(signal.take_profit - signal.entry_price)
        signal.rr_ratio = actual_reward / actual_risk if actual_risk > 0 else 0
        
        return signal

    def _calculate_confluence(self, h4_trend: str, h4_strength: int, 
                               m30_trend: str, signal: TradeSignal,
                               m30_candles: List[dict]) -> Tuple[int, List[str]]:
        """Calculate confluence score"""
        score = 0
        reasons = []
        
        # H4 Trend (2 points)
        if h4_strength >= 70:
            score += 2
            reasons.append(f"H4_Strong({h4_strength}%)")
        else:
            score += 1
            reasons.append(f"H4_Trend({h4_strength}%)")
        
        # M30 Alignment (2 points)
        if h4_trend == m30_trend:
            score += 2
            reasons.append("M30_Aligned")
        else:
            score += 1
            reasons.append("M30_OK")
        
        # Entry Type (1-2 points)
        if signal.rejection_type == 'BREAKOUT':
            score += 2
            reasons.append("Breakout")
        else:
            score += 1
            reasons.append("Pullback")
        
        # Volume/Momentum (1 point)
        recent = m30_candles[-5:]
        if signal.direction == 'BUY':
            if all(c['close'] > c['open'] for c in recent[-2:]):
                score += 1
                reasons.append("Momentum")
        else:
            if all(c['close'] < c['open'] for c in recent[-2:]):
                score += 1
                reasons.append("Momentum")
        
        return score, reasons

    def _calculate_confidence(self, confluence: int, h4_strength: int, signal: TradeSignal) -> float:
        """Calculate confidence percentage"""
        base = confluence * 10
        base += (h4_strength - 50) * 0.3
        
        # R:R bonus
        if signal.rr_ratio >= 2.0:
            base += 10
        elif signal.rr_ratio >= 1.8:
            base += 5
        
        return min(max(base, 30), 95)

    def _determine_trend(self, candles: List[dict]) -> str:
        """Alias for compatibility"""
        trend, _ = self._get_h4_trend(candles)
        return trend

    def _ema(self, data: np.ndarray, period: int) -> np.ndarray:
        """Calculate EMA"""
        ema = np.zeros_like(data, dtype=float)
        ema[0] = data[0]
        mult = 2 / (period + 1)
        for i in range(1, len(data)):
            ema[i] = (data[i] - ema[i-1]) * mult + ema[i-1]
        return ema

    def _calculate_atr(self, candles: List[dict], period: int = 14) -> float:
        """Calculate ATR"""
        if len(candles) < period + 1:
            return 100
        
        trs = []
        for i in range(1, len(candles)):
            h, l, pc = candles[i]['high'], candles[i]['low'], candles[i-1]['close']
            tr = max(h - l, abs(h - pc), abs(l - pc))
            trs.append(tr)
        
        return np.mean(trs[-period:]) if trs else 100
