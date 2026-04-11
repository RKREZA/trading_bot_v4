import numpy as np
import logging
from typing import Optional, List, Dict, Any
from core.common.types import MarketRegime, VolatilityStatus, TradeSignal

logger = logging.getLogger("trading_bot.adaptive_manager")

class AdaptiveStrategyManager:
    """
    V5-INSIGNIA Adaptive Strategy Selector.
    Dynamically selects the best strategy based on detected market regime.
    
    Strategy -> Regime Mapping:
    - LiquiditySweepBreakout: BEST for HIGH VOLATILITY / TREND breaks
    - TrendFollowing: BEST for TREND markets with NORMAL/LOW volatility
    - SmartMeanReversion: BEST for RANGE markets
    """
    
    def __init__(self, strategies: list, config: dict):
        self.strategies = {s.strategy_id: s for s in strategies}
        self.config = config
        self.active_strategy = None
        self.strategy_performance = {sid: {"wins": 0, "losses": 0, "profit": 0} for sid in self.strategies}
        self.regime_history = []
        
    def select_strategy(self, regime_info, market_data) -> Optional[Any]:
        """
        Select the best strategy based on market regime.
        Returns the selected strategy or None if no suitable strategy.
        """
        regime = regime_info.market_type
        volatility = regime_info.volatility
        adx = regime_info.adx
        atr = regime_info.atr
        
        selected_id = None
        
        # Priority-based strategy selection based on regime
        if regime == MarketRegime.TREND:
            if volatility == VolatilityStatus.HIGH:
                # High volatility + Trend = Breakout trades well
                selected_id = "LiquiditySweepBreakout"
            else:
                # Normal/Low volatility + Trend = TrendFollowing
                selected_id = "TrendFollowing"
                
        elif regime == MarketRegime.RANGE:
            # Range market = Mean reversion works best
            if self.strategies.get("SmartMeanReversion"):
                selected_id = "SmartMeanReversion"
            else:
                # Fallback to breakout for range breaks
                selected_id = "LiquiditySweepBreakout"
                
        else:  # UNCERTAIN
            # Uncertain regime = Conservative, use breakout
            selected_id = "LiquiditySweepBreakout"
        
        # Check if selected strategy is enabled
        if selected_id and selected_id in self.strategies:
            strat = self.strategies[selected_id]
            if getattr(strat, "enabled", True):
                self.active_strategy = strat
                self._record_regime_selection(regime, volatility, selected_id)
                return strat
        
        # Fallback to first enabled strategy
        for sid, strat in self.strategies.items():
            if getattr(strat, "enabled", True):
                self.active_strategy = strat
                return strat
        
        return None
    
    def _record_regime_selection(self, regime, volatility, strategy_id):
        """Track regime-strategy mappings for analysis."""
        self.regime_history.append({
            "regime": regime,
            "volatility": volatility,
            "strategy": strategy_id
        })
    
    def record_trade_result(self, strategy_id: str, pnl: float, is_win: bool):
        """Update strategy performance tracking."""
        if strategy_id in self.strategy_performance:
            self.strategy_performance[strategy_id]["profit"] += pnl
            if is_win:
                self.strategy_performance[strategy_id]["wins"] += 1
            else:
                self.strategy_performance[strategy_id]["losses"] += 1
    
    def get_strategy_for_regime(self, regime: MarketRegime, volatility: VolatilityStatus) -> str:
        """Returns the recommended strategy ID for a given regime."""
        mapping = {
            (MarketRegime.TREND, VolatilityStatus.HIGH): "LiquiditySweepBreakout",
            (MarketRegime.TREND, VolatilityStatus.NORMAL): "TrendFollowing",
            (MarketRegime.TREND, VolatilityStatus.LOW): "TrendFollowing",
            (MarketRegime.RANGE, VolatilityStatus.HIGH): "SmartMeanReversion",
            (MarketRegime.RANGE, VolatilityStatus.NORMAL): "SmartMeanReversion",
            (MarketRegime.RANGE, VolatilityStatus.LOW): "SmartMeanReversion",
            (MarketRegime.UNCERTAIN, VolatilityStatus.HIGH): "LiquiditySweepBreakout",
            (MarketRegime.UNCERTAIN, VolatilityStatus.NORMAL): "LiquiditySweepBreakout",
            (MarketRegime.UNCERTAIN, VolatilityStatus.LOW): "LiquiditySweepBreakout",
        }
        return mapping.get((regime, volatility), "LiquiditySweepBreakout")
    
    def get_performance_summary(self) -> Dict:
        """Returns performance metrics for each strategy."""
        summary = {}
        for sid, perf in self.strategy_performance.items():
            total = perf["wins"] + perf["losses"]
            win_rate = (perf["wins"] / total * 100) if total > 0 else 0
            summary[sid] = {
                "total_trades": total,
                "wins": perf["wins"],
                "losses": perf["losses"],
                "win_rate": win_rate,
                "profit": perf["profit"]
            }
        return summary
    
    def get_regime_distribution(self) -> Dict:
        """Returns distribution of detected regimes."""
        if not self.regime_history:
            return {}
        
        dist = {}
        for entry in self.regime_history:
            key = f"{entry['regime'].value}_{entry['volatility'].value}"
            dist[key] = dist.get(key, 0) + 1
        
        total = len(self.regime_history)
        return {k: f"{(v/total*100):.1f}%" for k, v in dist.items()}


class RegimeAwareStrategy:
    """
    Wrapper that adds regime-awareness to existing strategies.
    Dynamically adjusts parameters based on market conditions.
    """
    
    def __init__(self, strategy, config: dict):
        self.strategy = strategy
        self.config = config
        self.current_regime = None
        
    def generate_signal(self, market_data, regime_info) -> Optional[TradeSignal]:
        """Generate signal with regime-aware parameter adjustment."""
        self.current_regime = regime_info
        
        # Adjust confidence threshold based on regime
        base_conf = getattr(self.strategy, "min_confidence", 0.6)
        
        if regime_info.market_type == MarketRegime.TREND:
            # Lower confidence requirement for trends (easier to find setups)
            adjusted_conf = base_conf * 0.90
        elif regime_info.market_type == MarketRegime.RANGE:
            # Higher confidence for range (mean reversion is trickier)
            adjusted_conf = base_conf * 1.05
        else:
            adjusted_conf = base_conf
        
        # Apply temporary confidence adjustment
        original_conf = getattr(self.strategy, "min_confidence", 0.6)
        self.strategy.min_confidence = min(0.95, adjusted_conf)
        
        try:
            signal = self.strategy.generate_signal(market_data)
            return signal
        finally:
            # Restore original confidence
            self.strategy.min_confidence = original_conf
    
    def get_stop_loss(self, signal, market_data) -> float:
        return self.strategy.get_stop_loss(signal, market_data)
    
    def get_take_profit(self, signal, market_data) -> float:
        return self.strategy.get_take_profit(signal, market_data)
    
    @property
    def strategy_id(self):
        return self.strategy.strategy_id
    
    @property
    def enabled(self):
        return self.strategy.enabled
    
    @enabled.setter
    def enabled(self, value):
        self.strategy.enabled = value
    
    def is_symbol_allowed(self, symbol: str) -> bool:
        return self.strategy.is_symbol_allowed(symbol)
    
    def reset_daily_stats(self):
        if hasattr(self.strategy, "reset_daily_stats"):
            self.strategy.reset_daily_stats()


def create_adaptive_manager(strategies: list, config: dict) -> AdaptiveStrategyManager:
    """Factory function to create adaptive strategy manager."""
    return AdaptiveStrategyManager(strategies, config)
