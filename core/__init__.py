"""Trading Bot V4 - Institutional Core Package"""

from .connection import MT5Connection
from .data_handler import DataFetcher
from .risk_engine import RiskEngine
from .execution_engine import ExecutionEngine
from .portfolio_manager import PortfolioManager
from .performance_tracker import PerformanceTracker
from .regime_detector import RegimeDetector, MarketRegime
from .logger import setup_logging
from .types import CandleArray, TradeSignal, BotConfig

# Multi-Strategy Framework
from .base_strategy import BaseStrategy, MarketData

__all__ = [
    "MT5Connection",
    "DataFetcher",
    "RiskEngine",
    "ExecutionEngine",
    "PortfolioManager",
    "PerformanceTracker",
    "RegimeDetector",
    "MarketRegime",
    "setup_logging",
    "CandleArray",
    "TradeSignal",
    "BotConfig",
    "BaseStrategy",
    "MarketData",
]