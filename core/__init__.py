"""Trading Bot V3 - Core Module"""
from .strategy_engine import StrategyEngine, TradeSignal
from .connection import MT5Connection
from .data_fetcher import DataFetcher
from .backtester import BacktestEngine
from .risk_manager import RiskManager
from .performance import PerformanceMetrics
from .ai_advisor import AIAdvisor
from .validation import ValidationSuite
from .logger import setup_logging
from .broker_clock import BrokerClock

# Multi-Strategy Framework
from .base_strategy import BaseStrategy, MarketData, TaggedSignal
from .strategy_runtime import StrategyRuntime
from .strategy_orchestrator import StrategyOrchestrator
from .position_tracker import PositionTracker
from .performance_tracker import PerformanceTracker
from .order_tagger import OrderTagger

__all__ = [
    "StrategyEngine",
    "TradeSignal",
    "MT5Connection",
    "DataFetcher",
    "BacktestEngine",
    "RiskManager",
    "PerformanceMetrics",
    "AIAdvisor",
    "ValidationSuite",
    "setup_logging",
    "BrokerClock",
    # Multi-Strategy Framework
    "BaseStrategy",
    "MarketData",
    "TaggedSignal",
    "StrategyRuntime",
    "StrategyOrchestrator",
    "PositionTracker",
    "PerformanceTracker",
    "OrderTagger",
]