import numpy as np
import logging
from typing import Optional, Dict, Any, List
from core.base_strategy import BaseStrategy, MarketData
from core.common.types import TradeSignal
from core.volatility_detector import VolatilityDetector, VolatilityLevel, VolatilityAdaptiveParameters

logger = logging.getLogger("trading_bot.volatility_adaptive_strategy")


class VolatilityAdaptiveStrategy:
    """
    V5-INSIGNIA Volatility-Adaptive Strategy Wrapper.
    
    Dynamically selects and adjusts strategy parameters based on detected
    market volatility conditions. Wraps existing strategies with intelligent
    parameter optimization.
    
    Features:
    - Real-time volatility detection across multiple timeframes
    - Dynamic strategy selection (breakout vs mean reversion vs trend)
    - Volatility-adjusted parameters (confidence, SL/TP, frequency)
    - Risk management based on volatility regime
    """
    
    def __init__(self, wrapped_strategies: List[BaseStrategy], config: dict):
        self.config = config
        self.wrapped_strategies = {s.strategy_id: s for s in wrapped_strategies}
        self.volatility_detector = VolatilityDetector(atr_period=14, lookback=100)
        
        self.current_volatility: Optional[VolatilityLevel] = None
        self.current_strategy: Optional[BaseStrategy] = None
        self.volatility_history = []
        
        self._saved_params = {}
        self._cooldown_bars = 0
        self._vol_threshold_for_trades = VolatilityLevel.VERY_LOW
        
    def analyze_and_select(self, market_data: MarketData) -> Dict[str, Any]:
        """
        Analyze volatility and select best strategy.
        
        Returns:
            Dict with volatility analysis and recommended action
        """
        vol_analysis = self.volatility_detector.analyze(
            market_data.m5_candles,
            h1_candles=market_data.htf_candles
        )
        
        self.current_volatility = vol_analysis.level
        self._volatility_history.append(vol_analysis)
        
        recommended_action = self._determine_action(vol_analysis)
        selected_strategy = self._select_strategy_for_volatility(vol_analysis)
        
        return {
            "volatility": vol_analysis,
            "action": recommended_action,
            "selected_strategy": selected_strategy,
            "risk_multiplier": vol_analysis.risk_multiplier,
        }
    
    def _determine_action(self, vol_analysis) -> str:
        """Determine trading action based on volatility."""
        if vol_analysis.level == VolatilityLevel.EXTREME_LOW:
            return "NO_TRADE"
        elif vol_analysis.level == VolatilityLevel.VERY_LOW:
            return "REDUCE_EXPOSURE"
        elif vol_analysis.level == VolatilityLevel.LOW:
            return "SELECTIVE_TRADING"
        elif vol_analysis.level == VolatilityLevel.NORMAL:
            return "FULL_TRADING"
        elif vol_analysis.level == VolatilityLevel.HIGH:
            return "AGGRESSIVE_TRADING"
        elif vol_analysis.level == VolatilityLevel.VERY_HIGH:
            return "MOMENTUM_TRADING"
        else:
            return "PROTECT_PROFITS"
    
    def _select_strategy_for_volatility(self, vol_analysis) -> Optional[BaseStrategy]:
        """Select and configure the best strategy for current volatility."""
        recommended_type = vol_analysis.recommended_strategy
        
        strategy_map = {
            "NO_TRADE": None,
            "MEAN_REVERSION_CONSERVATIVE": "RangeBounce",
            "MEAN_REVERSION": "RangeBounce",
            "MOMENTUM_BREAKOUT": "LiquiditySweepBreakout",
            "MOMENTUM_BREAKOUT_AGGRESSIVE": "LiquiditySweepBreakout",
            "TREND_CATCHING": "TrendFollowing",
            "TREND_CATCHING_REVERSED": "RangeBounce",
        }
        
        preferred_strategy = strategy_map.get(recommended_type)
        
        if preferred_strategy is None:
            return None
            
        strategy = self.wrapped_strategies.get(preferred_strategy)
        
        if strategy is None:
            for sid, strat in self.wrapped_strategies.items():
                if getattr(strat, "enabled", False):
                    strategy = strat
                    break
        
        self.current_strategy = strategy
        return strategy
    
    def apply_volatility_parameters(self, strategy: BaseStrategy, vol_analysis) -> None:
        """
        Temporarily apply volatility-optimized parameters to a strategy.
        
        This modifies strategy parameters dynamically based on volatility conditions.
        """
        strategy_id = strategy.strategy_id
        
        strategy_type = self._get_strategy_type(strategy_id)
        vol_params = VolatilityAdaptiveParameters.get_parameters_for_volatility(
            vol_analysis.level,
            strategy_type
        )
        
        self._saved_params[strategy_id] = {
            "min_confidence": getattr(strategy, "min_confidence", 0.6),
            "min_bars_between_signals": getattr(strategy, "min_bars_between_signals", 20),
            "sl_atr": getattr(strategy, "sl_atr", 1.5),
            "tp_atr": getattr(strategy, "tp_atr", 6.0),
        }
        
        if hasattr(strategy, "min_confidence"):
            strategy.min_confidence = vol_params.get("min_confidence", 0.6)
        if hasattr(strategy, "min_bars_between_signals"):
            strategy.min_bars_between_signals = vol_params.get("min_bars_between_signals", 20)
        if hasattr(strategy, "sl_atr"):
            strategy.sl_atr = vol_params.get("sl_atr", 1.5)
        if hasattr(strategy, "tp_atr"):
            strategy.tp_atr = vol_params.get("tp_atr", 6.0)
        
        if hasattr(strategy, "body_thresh"):
            strategy.body_thresh = vol_params.get("body_thresh", 0.55)
        if hasattr(strategy, "h1_strength_thresh"):
            strategy.h1_strength_thresh = vol_params.get("h1_strength_thresh", 0.45)
        
        if hasattr(strategy, "rsi_oversold"):
            strategy.rsi_oversold = vol_params.get("rsi_oversold", 30)
        if hasattr(strategy, "rsi_overbought"):
            strategy.rsi_overbought = vol_params.get("rsi_overbought", 70)
        if hasattr(strategy, "bb_std"):
            strategy.bb_std = vol_params.get("bb_std", 2.0)
        
        logger.debug(f"[VolatilityAdaptive] Applied {vol_analysis.level.value} params to {strategy_id}")
    
    def restore_parameters(self, strategy: BaseStrategy) -> None:
        """Restore original parameters after trading."""
        strategy_id = strategy.strategy_id
        
        if strategy_id in self._saved_params:
            saved = self._saved_params[strategy_id]
            for param, value in saved.items():
                setattr(strategy, param, value)
            del self._saved_params[strategy_id]
    
    def _get_strategy_type(self, strategy_id: str) -> str:
        """Determine strategy type for parameter selection."""
        if "Breakout" in strategy_id or "Liquidity" in strategy_id:
            return "breakout"
        elif "MeanReversion" in strategy_id or "RangeBounce" in strategy_id:
            return "mean_reversion"
        elif "Trend" in strategy_id:
            return "trend"
        else:
            return "breakout"
    
    def should_trade(self, vol_analysis) -> bool:
        """Determine if trading should occur based on volatility."""
        if vol_analysis.level == VolatilityLevel.EXTREME_LOW:
            return False
        if vol_analysis.level == VolatilityLevel.VERY_LOW:
            return vol_analysis.signal_frequency_boost >= 0.3
        return True
    
    def get_trade_frequency_adjustment(self, vol_analysis) -> float:
        """Get multiplier for signal frequency (for cooldown adjustments)."""
        base_cooldown = 20
        adjusted_cooldown = int(base_cooldown * (1.0 / vol_analysis.signal_frequency_boost))
        return adjusted_cooldown
    
    def get_volatility_summary(self) -> Dict[str, Any]:
        """Get summary of volatility conditions over time."""
        if not self._volatility_history:
            return {"status": "No data"}
        
        levels = [v.level for v in self._volatility_history]
        level_counts = {}
        for level in levels:
            level_counts[level.value] = level_counts.get(level.value, 0) + 1
        
        ratios = [v.ratio for v in self._volatility_history]
        
        return {
            "total_bars": len(self._volatility_history),
            "level_distribution": {k: f"{(v/len(levels)*100):.1f}%" for k, v in level_counts.items()},
            "avg_ratio": np.mean(ratios),
            "current_level": self.current_volatility.value if self.current_volatility else "UNKNOWN",
            "current_ratio": ratios[-1] if ratios else 0.0,
        }


class MultiVolatilityStrategy:
    """
    Enhanced strategy that combines multiple strategies with volatility-based selection.
    
    Automatically switches between:
    - Breakout strategies in high volatility
    - Mean reversion in low volatility
    - Trend following in normal conditions
    """
    
    def __init__(self, strategies: List[BaseStrategy], config: dict):
        self.config = config
        self.adaptive_wrapper = VolatilityAdaptiveStrategy(strategies, config)
        self.strategies = strategies
        
    def generate_signal(self, market_data: MarketData) -> Optional[TradeSignal]:
        """Generate signal with volatility-adaptive strategy selection."""
        analysis = self.adaptive_wrapper.analyze_and_select(market_data)
        vol_analysis = analysis["volatility"]
        
        if not self.adaptive_wrapper.should_trade(vol_analysis):
            return None
        
        strategy = analysis["selected_strategy"]
        
        if strategy is None:
            return None
        
        self.adaptive_wrapper.apply_volatility_parameters(strategy, vol_analysis)
        
        try:
            signal = strategy.generate_signal(market_data)
            
            if signal:
                confidence_adjustment = vol_analysis.signal_frequency_boost
                signal.confidence = min(0.98, signal.confidence * confidence_adjustment)
                
            return signal
            
        finally:
            self.adaptive_wrapper.restore_parameters(strategy)
    
    def get_stop_loss(self, signal: TradeSignal, market_data: MarketData) -> float:
        """Get stop loss with volatility adjustment."""
        for strat in self.strategies:
            if strat.strategy_id == signal.strategy_id if hasattr(signal, 'strategy_id') else False:
                return strat.get_stop_loss(signal, market_data)
        
        if self.strategies:
            return self.strategies[0].get_stop_loss(signal, market_data)
        return market_data.current_price * 0.99
    
    def get_take_profit(self, signal: TradeSignal, market_data: MarketData) -> float:
        """Get take profit with volatility adjustment."""
        for strat in self.strategies:
            if strat.strategy_id == signal.strategy_id if hasattr(signal, 'strategy_id') else False:
                return strat.get_take_profit(signal, market_data)
        
        if self.strategies:
            return self.strategies[0].get_take_profit(signal, market_data)
        return market_data.current_price * 1.02
    
    @property
    def strategy_id(self) -> str:
        return "MultiVolatilityStrategy"
    
    @property
    def enabled(self) -> bool:
        return True
    
    def is_symbol_allowed(self, symbol: str) -> bool:
        return True
    
    def reset_daily_stats(self):
        for strat in self.strategies:
            if hasattr(strat, "reset_daily_stats"):
                strat.reset_daily_stats()
