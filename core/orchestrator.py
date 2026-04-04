"""
core/orchestrator.py
Orchestrates signal generation across multiple strategies.
"""
import logging
from typing import List, Optional, Any
from .base_strategy import MarketData

logger = logging.getLogger("trading_bot.orchestrator")

class StrategyOrchestrator:
    def __init__(self, strategies: List[Any], data_fetcher: Any):
        """
        Args:
            strategies: List of enabled strategy instances.
            data_fetcher: Instance of DataFetcher.
        """
        self.strategies = strategies
        self.data_fetcher = data_fetcher

    def run_cycle(self, symbol: str, session: str, current_price: float, broker_clock: Any) -> list:
        """
        Fetch market data and run all strategies once.
        Returns a list of standardized signal dictionaries.
        """
        # Fetch multi-timeframe candles once per cycle
        h1 = self.data_fetcher.fetch_candles(symbol, "H1", 200)
        m15 = self.data_fetcher.fetch_candles(symbol, "M15", 200)
        m5 = self.data_fetcher.fetch_candles(symbol, "M5", 500)
        d1 = self.data_fetcher.fetch_candles(symbol, "D1", 50)
        
        if len(m5) < 60:
            logger.warning(f"Insufficient data for {symbol} cycle.")
            return []

        # Prepare immutable market data container
        market_data = MarketData(
            symbol=symbol,
            htf_candles=h1,
            m15_candles=m15,
            m5_candles=m5,
            d1_candles=d1,
            current_price=current_price,
            session=session,
            timestamp=broker_clock.now()
        )

        signals = []
        for strategy in self.strategies:
            if not strategy.enabled:
                continue
                
            try:
                signal = strategy.generate_signal(market_data)
                if signal:
                    signals.append(signal)
            except Exception as e:
                logger.error(f"Error in strategy {strategy.strategy_id}: {e}", exc_info=True)

        return signals
