"""Trading Bot V4 - Institutional Core Package"""

# Common Infrastructure
from .common.types import CandleArray, TradeSignal, BotConfig, MarketRegime

# Institutional Packages
from .risk.risk_guardian import RiskGuardian
from .data.source_handler import SourceHandler
from .execution.order_manager import OrderManager

# Core Logic (To be refactored further)
from .connection import MT5Connection
from .performance_tracker import PerformanceTracker
from .regime_detector import RegimeDetector
from .logger import setup_logging
from .base_strategy import BaseStrategy, MarketData

__all__ = [
    "MT5Connection",
    "SourceHandler",
    "RiskGuardian",
    "OrderManager",
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